from torch.utils.data import Dataset

import os
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


class Plant(Dataset):
    def __init__(self, args, mode='train', transform=None, target_transform=None):
        super(Plant, self).__init__()

        self.transform = transform
        self.target_transform = target_transform

        self.train_img_path = os.path.join(args.data_dir, 'train_images')
        train_df_path = os.path.join(args.data_dir, 'train.csv')

        df_train = pd.read_csv(train_df_path)
        df_train['labels'] = df_train['labels'].apply(lambda string: string.split(' '))

        mlb = MultiLabelBinarizer()
        one_hot = pd.DataFrame(mlb.fit_transform(df_train['labels']), columns=mlb.classes_, index=df_train.index)
        data = pd.concat([df_train, one_hot], axis=1)

        x_train, x_valid, y_train, y_valid = train_test_split(data['image'], data[args.classes], test_size=args.test_size,
                                                              shuffle=True, random_state=args.seed)

        if mode == 'train':
            self.img_ids = np.array(x_train)
            self.targets = np.array(y_train)
        elif mode == 'valid':
            self.img_ids = np.array(x_valid)
            self.targets = np.array(y_valid)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_path = os.path.join(self.train_img_path, img_id)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']

        target = self.targets[index]
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_ids)
