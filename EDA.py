import os
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

train_img_path = './plant_dataset/train_images'
test_img_path = './plant_dataset/test_images'
train_df_path = './plant_dataset/train.csv'

df_train = pd.read_csv(train_df_path)

# # 1. Basic Values
# print(df_train.head())
# print(df_train['labels'].value_counts())

# # 2. Label Distributions Visualization
# fig, axes = plt.subplots(figsize=(20, 12), ncols=2, nrows=1)
#
# source = df_train['labels'].value_counts()
#
# sns.histplot(ax=axes[0], x=source.values)
# axes[0].set_title('Histplot')
#
# sns.barplot(ax=axes[1], x=source.index, y=source.values)
# axes[1].set_title('Barplot')
# plt.xticks(rotation=45)
#
# plt.show()

# # 3. Plant Images Visualization
# df_sample = df_train.sample(n=9)
# img_ids = df_sample['image'].values
# labels = df_sample['labels'].values
#
# fig = plt.figure(figsize=(15, 12))
# rows, cols = 3, 3
#
# for index, (img_id, label) in enumerate(zip(img_ids, labels)):
#     plt.subplot(rows, cols, index+1)
#     img = cv2.imread(os.path.join(train_img_path, img_id))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     plt.imshow(img)
#     plt.title('Class:{}, Size:{}'.format(label, img.shape))
#     plt.axis('off')
# plt.show()

# 4. Separate Multi Labels
df_train['labels'] = df_train['labels'].apply(lambda string: string.split(' '))

mlb = MultiLabelBinarizer()
trainx = pd.DataFrame(mlb.fit_transform(df_train['labels']), columns=mlb.classes_, index=df_train.index)

result = pd.concat([df_train, trainx], axis=1)
# print(result)

classes = ['healthy', 'rust', 'complex', 'powdery_mildew', 'scab', 'frog_eye_leaf_spot']

X_train, X_valid, y_train, y_valid = train_test_split(result['image'], result[classes], test_size=2000, shuffle=True, random_state=100)
print(len(np.array(X_train)))

# labels = list(trainx.sum().keys())
# label_counts = list(trainx.sum().values)

# fig, axes = plt.subplots(figsize=(20, 12), ncols=1, nrows=1)
#
# source = df_train['labels'].value_counts()
#
# sns.barplot(ax=axes, x=labels, y=label_counts)
# axes.set_title('Barplot')
#
# plt.show()

# unique_labels = df_train.labels.unique()
# single_labels = []
# for label in unique_labels:
#     single_labels += label.split()
# single_labels = list(set(single_labels))
#
# df_train[single_labels] = 0
# print(unique_labels)
#
# for label in unique_labels:
#     label_indices = df_train[df_train['labels'] == label].index
#     print(label_indices)




########################################
# print(df_train.head())
#
# unique_labels = df_train.labels.unique()
# single_labels = []
# for label in unique_labels:
#     single_labels += label.split()
# single_labels = list(set(single_labels))
# print(single_labels)
#
# df_train['labels'] = df_train['labels'].apply(lambda string: string.split(' '))
# print(df_train.head())


