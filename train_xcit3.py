import os 
import sys 

import argparse

import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

import SSB
from SSB.get_datasets.get_gcd_datasets_funcs import get_gcd_datasets
from SSB.utils import load_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, CustomDINOHead

from pytorch_metric_learning import losses

# random_seed = 777
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)

torch.cuda.empty_cache()


def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    # optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = AdamW(params_groups, lr=args.lr)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    update_cnt = 0
    max_all_acc = 0.
    
    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):

                student_proj, student_out = student(images)
                teacher_out = student_out.detach()
                
                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                if epoch < args.only_representation_epochs:
                    cls_loss = torch.tensor(0.).to('cuda:0')
                    cluster_loss = torch.tensor(0.).to('cuda:0')
                else: 
                    # clustering, sup
                    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                    
                    # clustering, unsup
                    cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                    cluster_loss += args.memax_weight * me_max_loss

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                    
                loss = 0
                loss += 1.3 * ((1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss)
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)

        args.logger.info('{}, Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(args.dataset_name, all_acc, old_acc, new_acc))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        update_cnt += 1
        if all_acc > max_all_acc:
            update_cnt = 0 
            max_all_acc = all_acc
            
            torch.save(save_dict, args.model_path)
            args.logger.info("model saved to {}.".format(args.model_path))
            
            with open(os.path.dirname(os.path.dirname(args.model_path)) + "/stat_{}.txt".format(args.dataset_name), 'w') as f: 
                f.write('epochs: {} \nTrain Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(epoch, all_acc, old_acc, new_acc))

        if update_cnt > 190:
            print("[ EARLY STOP THE SCORE HAS NOT CHANEGE OVER 100 EPOCHS ]")
            break
            
def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)
    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=23)
    parser.add_argument('--only_representation_epochs', type=int, default=0)
    
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    
    parser.add_argument('--rep_weight', type=float, default=None)
    
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    
    ### SOMETHING I ADDED 
    parser.add_argument('--head_type', default=None, type=str)


    args = parser.parse_args()
    device = torch.device('cuda:0')

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    
    # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16')
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')
    
    args.image_size = 224
    args.feat_dim = 512
    args.num_mlp_layers = 3
    class_splits = load_class_splits(args.dataset_name)
    args.train_classes = class_splits['known_classes']
    args.mlp_out_dim = len(class_splits['known_classes']) \
                     + len(class_splits['unknown_classes']['Easy']) \
                     + len(class_splits['unknown_classes']['Medium']) \
                     + len(class_splits['unknown_classes']['Hard'])

    for m in backbone.parameters():
        m.requires_grad = False

    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block: #gardient from block 보다 노드가 많으면 학습해야 한단 거가? 
                m.requires_grad = True

    args.logger.info('model build')

    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_gcd_datasets(args.dataset_name, 
                                                                                            train_transform,
                                                                                            test_transform,)

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)


    # ---------------------
    # FINAL MODEL
    # ---------------------
    if args.head_type == "custom":
        projector = CustomDINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
        
    else: 
        projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
        
    model = nn.Sequential(backbone, projector)
    
    print("==== MODEL SPEC ====")
    print(model)
    print("====================")
    
    
    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu')["model"])
    
    model = model.to(device)

    train(model, train_loader, None, test_loader_unlabelled, args)
