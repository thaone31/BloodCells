import numpy as np
import pandas as pd
import os
import random

from tqdm import tqdm
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./detr/')

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import matplotlib.patches as patches
import matplotlib.text as text

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

seed = 42
num_classes = 6
num_queries = 20
null_class_coef = 0.2
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 20
ANNOT_DIR = './DataImage/annotation'
device = torch.device('cuda')

def get_objects(xml_file):
    annotation = ET.parse(xml_file)
    root = annotation.getroot()
    
    name = os.path.basename(xml_file).replace('.xml','')
    size = root.find('size')
    
    objects = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        new_object = {}
        new_object['image_id'] = name
        new_object['labels'] = obj.find('name').text
        new_object['width'] = int(size.find('width').text)
        new_object['height'] = int(size.find('height').text)
        new_object['x'] = int(bbox.find('xmin').text)
        new_object['y'] = int(bbox.find('ymin').text)
        new_object['w'] = int(bbox.find('xmax').text)-int(bbox.find('xmin').text)
        new_object['h'] = int(bbox.find('ymax').text)-int(bbox.find('ymin').text)
        objects.append(new_object)
    return objects

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(seed)

annots = []
for xml in os.listdir(ANNOT_DIR):
    annots += get_objects(os.path.join(ANNOT_DIR,xml))
df = pd.DataFrame(annots)

# Adjust w, h
for i in df.index:
    exceed_w =  df.iloc[i].x + df.iloc[i].w - df.iloc[i].width
    exceed_h =  df.iloc[i].y + df.iloc[i].h - df.iloc[i].height
    if exceed_w > 0:
        df.loc[df.index == i,'w'] -= exceed_w
    if exceed_h > 0:
        df.loc[df.index == i,'h'] -= exceed_h

encoder = LabelEncoder()
df.labels = encoder.fit_transform(df.labels)

df.head()
# g = sns.countplot(x='labels',data=df)
# g.set_xticklabels(encoder.classes_);


df_split = df[['image_id']].copy()
df_split = pd.concat([df_split,pd.get_dummies(df.labels).astype('int32')],axis=1)
df_split['bbox_count'] = 1
df_split = df_split.groupby('image_id').sum()

# g = plt.figure(figsize=(15,4))
# g = sns.countplot(x='bbox_count',data=df_split)

df_split

train_split = df_split[:int(len(df_split)*.9)]
val_split = df_split[int(len(df_split)*.9):]

# fig, axs = plt.subplots(1,2,figsize=(13,4))
# sns.countplot(x='labels',data=df[df['image_id'].isin(train_split.index)],ax=axs[0])
# axs[0].set_title('Labels distribution in training set')
# sns.countplot(x='labels',data=df[df['image_id'].isin(val_split.index)],ax=axs[1])
# axs[1].set_title('Labels distribution in validation set')


def train_transform():
    return A.Compose([
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.RandomBrightnessContrast(),
        A.Resize(300, 300),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco',label_fields=['labels']))

def valid_trainsform():
    return A.Compose([
        A.Resize(300,300),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco',label_fields=['labels']))

IMG_DIR = './DataImage/image'

class MaskDataset(Dataset):
    def __init__(self, image_ids, dataframe, transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms
    def __len__(self) -> int:
        return self.image_ids.shape[0]
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id']==image_id]
        
        image = cv2.imread(f'{IMG_DIR}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        boxes = records[['x', 'y', 'w', 'h']].values
        area = boxes[:,2]*boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)
        
        labels =  records['labels'].values
        
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']
            
        _, h, w = image.shape
        boxes = A.core.bbox_utils.denormalize_bboxes
        
        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        
        return image, target, image_id

def collate_fn(batch):
    return tuple(zip(*batch))


train_dataset = MaskDataset(image_ids=train_split.index.values,
                           dataframe=df,
                           transforms=train_transform())
train_data_loader = DataLoader(train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=2,
                             collate_fn=collate_fn)
val_dataset = MaskDataset(image_ids=val_split.index.values,
                           dataframe=df,
                           transforms=valid_trainsform())
val_data_loader = DataLoader(val_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=2,
                             collate_fn=collate_fn)


matcher = HungarianMatcher()
weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
losses = ['labels', 'boxes', 'cardinality']

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DETRModel(nn.Module):
    def __init__(self, num_classes, num_queries):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries
    def forward(self, images):
        return self.model(images)

def train_fn(data_loader, model, criterion, optimizer, device, epoch):
    model.train()
    criterion.train()
    
    summary_loss = AverageMeter()
    
    for step, (images, targets, image_ids) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        output = model(images)
        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k]*weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        summary_loss.update(losses.item(), BATCH_SIZE)
    return summary_loss

def eval_fn(data_loader, model, criterion, device, epoch):
    model.eval()
    criterion.eval()
    
    summary_loss = AverageMeter()
    
    with torch.no_grad():
        for step, (images, targets, image_ids) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            output = model(images)
            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k]*weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            summary_loss.update(losses.item(), BATCH_SIZE)
    return summary_loss


os.environ['WANDB_CONSOLE'] = 'off'

def run():
    model = DETRModel(num_classes=num_classes, num_queries=num_queries)
    model = model.to(device)
    criterion = SetCriterion(num_classes-1, matcher, weight_dict, eos_coef=null_class_coef, losses=losses)
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_loss = 10**5
    for epoch in range(1,EPOCHS+1):
        train_loss = train_fn(train_data_loader, model, criterion, optimizer, device, epoch=epoch)
        valid_loss = eval_fn(val_data_loader, model, criterion, device, epoch)
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), f'detr_best.pth')
        if epoch%1 == 0:
            print(f'Epoch {epoch+0:03}: | Train Loss: {train_loss.avg:.5f} | Val Loss: {valid_loss.avg:.5f}')


model = run()

def show_sample(index):
    images, targets, image_ids = next(iter(val_data_loader))
    images = list(img.to(device) for img in images)
    image = images[index].permute(1,2,0).cpu().numpy()
    original_image = image.copy()

    boxes = targets[index]['boxes'].numpy()
    boxes = [np.array(box).astype(np.int32) for box in A.augmentations.bbox_utils.denormalize_bboxes(boxes,image.shape[0],image.shape[1])]

    model.eval()
    model.to(device)
    with torch.no_grad():
        outputs = model(images)
    pred_classes = outputs['pred_logits'][index].softmax(1).cpu().numpy()
    pred_boxes = outputs['pred_boxes'][index].cpu().numpy()
    pred_boxes = [np.array(box).astype(np.int32) for box in A.augmentations.bbox_utils.denormalize_bboxes(pred_boxes,image.shape[0],image.shape[1])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2]+box[0], box[3]+box[1]),
                      (255, 0, 0), 1)
    class_color = {0:(255, 255, 0),1:(0, 0, 255),2:(0, 255, 0)}
    for pred_class, pred_box in zip(pred_classes, pred_boxes):
        c = pred_class.argmax(0)
        if c != 3:
            pred_prob = pred_class.max(0)
            cv2.rectangle(image,
                  (pred_box[0], pred_box[1]),
                  (pred_box[2]+pred_box[0], pred_box[3]+pred_box[1]),
                  class_color[c], 1)
            cv2.rectangle(image,
                         (pred_box[0]+33,pred_box[1]),
                         (pred_box[0], pred_box[1]-15),
                          class_color[c], -1)
            ax2.text(pred_box[0], pred_box[1]-8, f'Class:{c}',color='w',fontsize=9)
            ax2.text(pred_box[0], pred_box[1], f'Prob:{int(pred_prob*100)}%',color='w',fontsize=9)
    ax1.set_axis_off()
    ax1.imshow(original_image)
    ax2.set_axis_off()
    ax2.imshow(image)


model = DETRModel(num_classes=num_classes,num_queries=num_queries)
model.load_state_dict(torch.load("./detr_best.pth"))
show_sample(0)
