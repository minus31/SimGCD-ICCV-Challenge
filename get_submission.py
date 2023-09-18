import os 
import sys 


import argparse
import math

from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

import SSB
from SSB.get_datasets.get_gcd_datasets_funcs import get_gcd_datasets
from SSB.utils import load_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from data.cub import get_cub_datasets

# Best at Any Dataset 
model_paths = {
    "fgvc_aircraft" : './dev_outputs/simgcd/log/{}/checkpoints/model.pt'.format("aircraft_simgcd_(07.09.2023_|_33.607)"),
    "cub" :  './dev_outputs/simgcd/log/{}/checkpoints/model.pt'.format(         "cub_simgcd_(15.09.2023_|_58.804)"),
    "scars" : './dev_outputs/simgcd/log/{}/checkpoints/model.pt'.format(        "scars_simgcd_(15.09.2023_|_58.803)"),
}

# Best at 1magenet1k
# model_paths = {
#     "fgvc_aircraft" : './dev_outputs/simgcd/log/{}/checkpoints/model.pt'.format("aircraft_simgcd_(14.09.2023_|_32.896)"),
#     "cub" :  './dev_outputs/simgcd/log/{}/checkpoints/model.pt'.format(         "cub_simgcd_(15.09.2023_|_37.482)"),
#     "scars" : './dev_outputs/simgcd/log/{}/checkpoints/model.pt'.format(        "scars_simgcd_(14.09.2023_|_32.878)"),
# }


template_submission = {
    'fgvc_aircraft' : "./pred_aircraft.csv",
    'cub' : "./pred_cub.csv",
    'scars' : "./pred_scars.csv"
}

sample_submission = {
    'fgvc_aircraft' : "starting_k/pred_aircraft.csv",
    'cub' : "starting_k/pred_cub.csv",
    'scars' : "starting_k/pred_scars.csv"
}

# backbone = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

# backbone_out_dim = 512
backbone_out_dim = 768


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
class SubmissionImageFolder():

    """
    Base ImageFolder
    """

    def __init__(self, paths, transform):

        self.paths = paths
        self.transform = transform
        self.target_transform = None      

    def __len__(self):
        return len(self.paths)
    
    
    def __getitem__(self, item):

        img = pil_loader(self.paths[item])
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    
def get_img_paths(dataset_name):
    """
    Arguments:
    dataset_name = {"scars", "fgvc_aircraft", "cub"}
    """
    
    df_submission = pd.read_csv(sample_submission[dataset_name])
    glob_format = {
            "scars" : "./dataset/Stanford_Cars/car_data/car_data/*/*/*.jpg",
            "fgvc_aircraft" : "./dataset/FGVC_Aircraft/fgvc-aircraft-2013b/data/images/*.jpg",
            "cub" : "./dataset/CUB/CUB_200_2011/images/*/*.jpg"
        }
    whole_data_list = glob(glob_format[dataset_name])
    name_extract = [x.split("/")[-1] for x in whole_data_list]
    
    img_paths = []
    for sample in df_submission["img"]:

        idx = name_extract.index(sample)

        if idx is not None:
            img_paths.append(whole_data_list[idx])

        else:
            print(sample)
            return None

    return img_paths


def predict(model, test_loader):

    model.eval()

    preds = []

    for batch_idx, images in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())

    preds = np.concatenate(preds)

    return preds



def main(dataset_name, args):
    
    args.dataset_name = dataset_name
    print(f"{dataset_name} STARTED")
    
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
# 
    args.interpolation = 3
    args.crop_pct = 0.875

    args.image_size = 224
    # args.image_size = 448
    # args.feat_dim = 768
    args.feat_dim = backbone_out_dim
    args.num_mlp_layers = 3

    class_splits = load_class_splits(dataset_name)

    args.train_classes = class_splits['known_classes']
    args.mlp_out_dim = len(class_splits['known_classes']) \
                    + len(class_splits['unknown_classes']['Easy']) \
                    + len(class_splits['unknown_classes']['Medium']) \
                    + len(class_splits['unknown_classes']['Hard'])

    for m in backbone.parameters():
        m.requires_grad = False

    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    
    # --------------------
    # DATASETS
    # --------------------
    img_paths = get_img_paths(dataset_name)
    submission_dataset = SubmissionImageFolder(img_paths, train_transform)


    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    model_path = model_paths[dataset_name]

    print("LOAD MODEL : ", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])

    submission_loader = DataLoader(submission_dataset, 
                            num_workers=args.num_workers, 
                            batch_size=256, 
                            shuffle=False, 
                            pin_memory=False)

    predictions = predict(model, submission_loader)

    df_submission = pd.read_csv(sample_submission[dataset_name])
    df_submission['pred'] = predictions
    
    df_submission.to_csv(template_submission[dataset_name], index=False)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    args = parser.parse_args()
    
    for name in ["cub", "scars", "fgvc_aircraft"]:
        main(name, args)
        
    print("DONE three files")