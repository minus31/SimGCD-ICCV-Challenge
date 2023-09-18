import argparse
from esvit.config import config
from esvit.config import update_config
from esvit.config import save_config

import esvit.utils as utils
from esvit.models import build_model

def load_model():
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')

    parser.add_argument('--cfg',
                        default="/conor/SimGCD-ICCV-Challenge/esvit/experiments/imagenet/swin/swin_base_patch4_window14_224.yaml",
                        help='experiment configure file name',
                        type=str)

    parser.add_argument('--arch', default='swin_base', type=str,
        choices=['cvt_tiny', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'swin', 'vil', 'vil_1281', 'vil_2262', 'deit_tiny', 'deit_small', 'vit_base'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")

    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')

    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')

    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")

    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')

    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)

    # Dataset
    parser.add_argument('--zip_mode', type=utils.bool_flag, default=False, help="""Whether or not
        to use zip file.""")


    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)    

    args = parser.parse_args(args=[])
    
    update_config(config, args)
    model = build_model(config, is_teacher=True)

    # model = model.load_state_dict(ckpt["teacher"], strict=False)
    utils.load_pretrained_weights(model, 
                                "/conor/SimGCD-ICCV-Challenge/checkpoint_best.pth", 
                                "teacher", 
                                "swin_base", 4)
    
    return model