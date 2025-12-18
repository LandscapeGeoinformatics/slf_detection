import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms import ToTensor

#from utils.loss import TverskyLoss
from utils.metrics import calculate_metrics
from utils.trainer import train_and_evaluate
#from utils.model import UNet

# Ensure reproducibility
seed_value = 0

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
np.random.seed(seed_value)

# Optional: for further ensuring reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# new dataset

class mydataset(Dataset):
    def __init__(self, dir, season="summer", split="train", dem=False):
        super(mydataset, self).__init__()
        self.rgb_dir = os.path.join(dir, f'{split}_image')
        self.mask_dir = os.path.join(dir, f'{split}_label')
        self.dem_dir = os.path.join(dir, 'dem')
        
        self.concat = dem
        
        self.target_size = (512, 512)
        
        self.images = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.tif')])
    
        #season_stats = {
        #    "spring": {
        #        "mean": [0.219, 0.231, 0.179],
        #        "std": [0.248, 0.256, 0.202]
        #    },
        #    "summer": {
        #        "mean": [0.198, 0.248, 0.168],
        #        "std": [0.224, 0.269, 0.188]
        #    }
        #}
    
        #stats = season_stats.get(season, season_stats["summer"])
        #self.rgb_mean = stats["mean"]
        #self.rgb_std = stats["std"]
    
        #self.dem_mean = 53.184
        #self.dem_std = 43.774
        self.dem_min = 0.017
        self.dem_max = 267.818
    
        #self.rgb_transform = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize(self.rgb_mean, self.rgb_std)
        #])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        dem_path = os.path.join(self.dem_dir, self.images[idx])
    
        # load rgb
        image = Image.open(rgb_path).convert("RGB")
        #image = self.rgb_transform(image) #if using transforms.Normalize
        image = image.resize(self.target_size, resample=Image.BILINEAR) #resize
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.clip(image, 0.0, 1.0)
        image = torch.from_numpy(image.transpose(2, 0, 1))
    
        # load mask
        mask = Image.open(mask_path)
        mask = mask.resize(self.target_size, resample=Image.NEAREST) #resize
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) # use long for multi-class task

        # load dem if it exists and concat if enabled
        if self.concat:
            if os.path.exists(dem_path):
                dem = Image.open(dem_path).convert("F")
                dem = dem.resize(self.target_size, resample=Image.NEAREST) #resize
                dem = np.array(dem, dtype=np.float32)
                dem = torch.from_numpy(dem).unsqueeze(0)
                #dem = (dem - self.dem_mean) / self.dem_std
                dem = (dem - self.dem_min) / (self.dem_max - self.dem_min)
                dem = torch.clamp(dem, 0.0, 1.0)
            else:
                # If DEM is missing, use a zero tensor of appropriate shape
                dem = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
                # Optionally log or warn about the missing file
                print(f"Warning: DEM file not found at {dem_path}, using zero tensor instead.")

            return torch.cat((image, dem), dim=0), mask
        return image, mask

train_set = mydataset(dir='/landscape_elements/working/patches/train_aug_new/train_multi/', split='train', dem=False)
val_set = mydataset(dir='/landscape_elements/working/patches/train_aug_new/train_multi/', split='val', dem=False)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

# If using GPU
print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')

patch_size = 512
batch_size = 16
n_channel = 3
n_class = 1
base_lr = 1e-5
min_epochs = 0
max_epochs = 50
patience = 5

import segmentation_models_pytorch as smp

model = smp.Unet(
    in_channels=3,
    classes=1,
)

model = model.to(device)


bce_loss = nn.BCEWithLogitsLoss()
criterion = bce_loss
optimizer = optim.Adam(model.parameters(), lr=base_lr)
#optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4) #1e-4

print(batch_size, base_lr, criterion)

num_epochs = max_epochs

log_dict = {
    'loss_per_epoch': [],
    'val_loss_per_epoch': [],
    'iou_per_epoch': [],
    'val_iou_per_epoch': [],
    'precision_per_epoch': [],
    'val_precision_per_epoch': [],
    'recall_per_epoch': [],
    'val_recall_per_epoch': [],
    'accuracy_per_epoch': [],
    'val_accuracy_per_epoch': [],
    'f1_per_epoch': [],
    'val_f1_per_epoch': [],
    'lr_per_epoch': []  # Add learning rate to the log dictionary
}
best_loss = 9999999
epochs_without_improvement = 0

# unet
train_and_evaluate(model, train_loader, val_loader, 
                   criterion, 
                   optimizer, 
                   num_epochs, 
                   device, 
                   calculate_metrics, 
                   log_dict, 
                   min_epochs, 
                   best_loss, 
                   epochs_without_improvement, 
                   patience, 
                   scheduler=None, 
                   early_stopping=True)
