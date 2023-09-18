import os
# -----------------
# DATASET ROOTS
# -----------------
DATASET_DIR = "/conor/SimGCD-ICCV-Challenge/dataset"

cifar_10_root = '${DATASET_DIR}/cifar10'
cifar_100_root = '${DATASET_DIR}/cifar100'
# cub_root = '${DATASET_DIR}/CUB'
# aircraft_root = '${DATASET_DIR}/fgvc-aircraft-2013b'
# car_root = '${DATASET_DIR}/cars'
herbarium_dataroot = '${DATASET_DIR}/herbarium_19'
imagenet_root = '${DATASET_DIR}/ImageNet'

cub_root = os.path.join(DATASET_DIR, 'CUB')
aircraft_root = os.path.join(DATASET_DIR, "FGVC_Aircraft", 'fgvc-aircraft-2013b')
car_root = os.path.join(DATASET_DIR, 'Stanford_Cars')

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
exp_root = 'dev_outputs' # All logs and checkpoints will be saved here