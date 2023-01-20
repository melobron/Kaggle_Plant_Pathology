import albumentations as A
import torch.nn
import torchvision.models
from albumentations.pytorch import ToTensorV2

import numpy as np
import os
from sklearn.metrics import f1_score, accuracy_score


################################# Path & Directory #################################
def make_exp_dir(main_dir):
    dirs = os.listdir(main_dir)
    dir_nums = []
    for dir in dirs:
        dir_num = int(dir[3:])
        dir_nums.append(dir_num)
    if len(dirs) == 0:
        new_dir_num = 1
    else:
        new_dir_num = max(dir_nums) + 1
    new_dir_name = 'exp{}'.format(new_dir_num)
    new_dir = os.path.join(main_dir, new_dir_name)
    return {'new_dir': new_dir, 'new_dir_num': new_dir_num}


################################# Transforms #################################
def get_transforms(args):
    transform_list = []
    if args.resize:
        transform_list.append(A.Resize(args.patch_size, args.patch_size))
    if args.shift_scale_rotate:
        transform_list.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.6))
    if args.rotate:
        transform_list.append(A.Rotate(p=0.1, limit=(-68, 178), border_mode=0, value=(0, 0, 0)))
    if args.brightness_contrast:
        transform_list.append(A.RandomBrightnessContrast(p=0.3))
    if args.rgb_shift:
        transform_list.append(A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3))
    if args.random_shadow:
        transform_list.append(A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=3, shadow_roi=(0, 0.6, 1, 1), p=0.4))
    if args.random_fog:
        transform_list.append(A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.2, alpha_coef=0.2, p=0.3))
    if args.gauss_noise:
        transform_list.append(A.GaussNoise(var_limit=(50, 70), always_apply=False, p=0.3))
    if args.coarse_dropout:
        transform_list.append(A.CoarseDropout(max_holes=5, max_height=5, max_width=5, min_holes=3, min_height=5, min_width=5, p=0.2))
    if args.normalize:
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform_list.append(ToTensorV2())
    return transform_list


################################# Metrics #################################
class MetricMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.accuracies = []
        self.scores = []
        self.metrics = dict({
            'loss': self.losses,
            'acc': self.accuracies,
            'f1': self.scores
        })

    def update(self, metric_name, value):
        self.metrics[metric_name] += [value]


def get_metrics(pred_prob, target, threshold):
    y_pred = np.where(pred_prob > threshold, 1, 0).astype(np.float)
    y_target = target.astype(np.float)

    f1 = f1_score(y_pred, y_target, average='weighted')
    acc = accuracy_score(y_pred, y_target, normalize=True)

    return acc, f1


################################# Model #################################
def create_model(args, pretrained=True):
    model = torchvision.models.resnet50(pretrained=pretrained)

    for param in model.layer1.parameters():
        param.requires_grad = False

    for param in model.layer2.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Sequential(*[
        torch.nn.Linear(in_features=model.fc.in_features, out_features=args.n_classes),
        torch.nn.Sigmoid()
    ])
    return model


################################# ETC #################################
def get_statistics(imgs, max_pixel=255.):
    linear_sum, square_sum = 0, 0
    count = 0

    for index, img in enumerate(imgs):
        img = img / max_pixel
        h, w = img.shape[0], img.shape[1]
        count += h * w
        linear_sum += img.sum(axis=(0, 1))
        square_sum += np.square(img).sum(axis=(0, 1))

    mean = linear_sum / count
    var = square_sum / count - (mean ** 2)
    std = np.sqrt(var)
    print('mean:{}, std:{}'.format(mean, std))
    return {'mean': mean, 'std': std}


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
