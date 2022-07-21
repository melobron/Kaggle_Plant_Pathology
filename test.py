import argparse

# Arguments
parser = argparse.ArgumentParser(description='Test Plant Pathology Classification')

parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=100)

# Data parameters
parser.add_argument('--data_dir', type=str, default='./plant_dataset')
parser.add_argument('--n_classes', type=int, default=6)
parser.add_argument('--classes', type=list, default=['rust', 'complex', 'healthy', 'powdery_mildew', 'scab', 'frog_eye_leaf_spot'])

# Training parameters
parser.add_argument('--threshold', type=int, default=0.4)

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=224)
parser.add_argument('--shift_scale_rotate', type=bool, default=True)
parser.add_argument('--rotate', type=bool, default=True)
parser.add_argument('--brightness_contrast', type=bool, default=True)
parser.add_argument('--rgb_shift', type=bool, default=True)
parser.add_argument('--random_shadow', type=bool, default=True)
parser.add_argument('--random_fog', type=bool, default=True)
parser.add_argument('--gauss_noise', type=bool, default=True)
parser.add_argument('--coarse_dropout', type=bool, default=True)
parser.add_argument('--normalize', type=bool, default=True)

args = parser.parse_args()

