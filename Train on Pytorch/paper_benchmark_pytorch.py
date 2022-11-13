from mobile_vit_pytorch import MobileViT
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
import os
import pandas as pd
from PIL import Image
import cv2
import tqdm
import time
import wandb
import timm

# run with lower GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#  MODELS
def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=257, expansion=2)

def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)



DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
WEIGHTS_PATH = '/home/angepapa/PycharmProjects/Mobile-vit/weights'

# Confid File
CFG = {'name': 'Mobile-ViT',
       'implementation': 'Pytorch',
       'epochs': 300,
       'batch_size': 128,
       'initial_lr': 1e-03,
       'seed': 42,
       'scheduler': 'None',
       'wandb': True}

# show CFG
for i in CFG:
    print(f'{i}: {CFG[i]}')

# Activate weights and biases
if CFG['wandb']:
    wandb.init(project='mobile-vit', entity='angepapa', config=CFG)


# count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Set randomSeed for reproducability
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG['seed'])

print(f'Device: {DEVICE}')


# Dataset creation for Dataloader
class caltech256train(Dataset):
    def __init__(self, filename, labels, transform):
        self.filename = filename
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, item):
        image = self.filename[item]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        augmented = self.transform(image=image)
        image = augmented['image']
        label = self.labels[item]
        return image, label


class caltech256valid(Dataset):
    def __init__(self, filename, labels, transform):
        self.filename = filename
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, item):
        image = self.filename[item]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        augmented = self.transform(image=image)
        image = augmented['image']
        label = self.labels[item]
        return image, label


data_transforms = {
    "train": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Normalize(),
        ToTensorV2()], p=1.),
    "valid": A.Compose([
        A.Normalize(),
        ToTensorV2()], p=1.)
}

# Main
if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING = 1
    # Load data
    df = pd.read_csv('/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/caltech_data.csv')
    train_paths, train_labels = df[df['execution'] == 'train'][['full_paths', 'labels']].values.T
    valid_paths, valid_labels = df[df['execution'] == 'eval'][['full_paths', 'labels']].values.T

    # Create Dataloaders
    trainset = caltech256train(filename=train_paths, labels=train_labels, transform=data_transforms['train'])
    validset = caltech256valid(filename=valid_paths, labels=valid_labels, transform=data_transforms['valid'])
    trainloader = DataLoader(trainset, shuffle=True, num_workers=4, pin_memory=True,
                             batch_size=CFG['batch_size'])
    validloader = DataLoader(validset, shuffle=True, num_workers=4, pin_memory=True,
                             batch_size=CFG['batch_size'])

    # Create model
    model = timm.create_model('efficientformer_l1', pretrained=True, num_classes=257)
    print(count_parameters(model))
    # model = mobilevit_xxs()
    model.to(DEVICE)

    def criterion(outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['initial_lr'])
    # scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8,
                                                           min_lr=1e-5, patience=5,
                                                           mode='min', verbose=True)

    max_acc = 0
    # train
    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        losses = []
        train_running_loss = 0.0
        valid_running_loss = 0.0
        acc = 0.0
        start = time.time()
        for i, element in enumerate(trainloader):
            images, labels = element[0].to(DEVICE), element[1].to(DEVICE)

            with torch.cuda.amp.autocast():
                pred = model(images)
                loss = criterion(pred, labels)

            losses.append(loss.item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            train_running_loss += loss.item()

        mean_loss = sum(losses) / len(losses)

        # watch lr
        lr = optimizer.param_groups[0]['lr']
        print(f'lr: {lr}')
        print(f'Epoch: {epoch}')
        print(f'Training Loss: {mean_loss}\n')

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i, element in enumerate(validloader):
                    images, labels = element[0].to(DEVICE), element[1].to(DEVICE)
                    pred = model(images)
                    loss = criterion(pred, labels)
                    valid_running_loss += loss.item()
                    temp = torch.sum(torch.argmax(pred, 1).float() == labels)
                    acc += temp / len(labels)


        print(f'Validation Loss: {valid_running_loss /(i + 1)}, Accuracy: {acc /(i + 1)}')

        scheduler.step(valid_running_loss /(i + 1))
        print(f'time elapsed: {(time.time() - start) / 60} min')
        if CFG['wandb']:
            wandb.log({
                'Epoch': epoch,
                'lr': lr,
                'Training Loss': (train_running_loss / (i + 1)),
                'Validation Loss': (valid_running_loss / (i + 1)),
                'Accuracy': (acc / (i + 1)),
                'time': (time.time() - start) / 60
            })

        if acc > max_acc:
            torch.save(model.state_dict(), WEIGHTS_PATH+f'/_{CFG["name"]}_{CFG["implementation"]}_{CFG["batch_size"]}_{CFG["initial_lr"]}.pth')







