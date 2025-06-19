import os
import torch
import lightning as L
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from PIL import Image
from torch.nn import Conv2d, MaxPool2d, ReLU, Sequential, ConvTranspose2d, Sigmoid, BatchNorm2d, Upsample
from pytorch_msssim import ssim
from enum import Enum

# dataset: https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification

class TransformType(Enum):
    RGB_TO_HSV = 1
    HSV_TO_RGB = 2
    
TYPE = TransformType.HSV_TO_RGB

RESIZE_SIZE = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_MAX_EPOCHS = 10

def ssim_loss(pred, target):
    # ssim() returns similarity [0, 1], so loss = 1 - similarity
    return 1 - ssim(pred, target, data_range=1.0, size_average=True)

class PredictDataSet(Dataset):
    def __init__(self, dataset_dir, transform = None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        path = os.path.join(dataset_dir)
        self.images = [img for img in os.listdir(path) if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg')]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.images[idx])
        if TYPE == TransformType.RGB_TO_HSV:
            rgb = Image.open(img_path).convert('RGB')
            hsv = rgb.convert('HSV')
        elif TYPE == TransformType.HSV_TO_RGB:
            rgb = Image.open(img_path).convert('HSV')
            hsv = rgb.convert('RGB')
        if self.transform:
            rgb = self.transform(rgb)
            hsv = self.transform(hsv)
        return rgb, hsv

class LightingData(L.LightningDataModule):
    def __init__(self, dataset_dir, batch_size, num_workers, transform = transforms.ToTensor(), train_ratio = 0.8, val_ratio = 0.2):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        dataset = PredictDataSet(self.dataset_dir, transform=transform)
        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size
        val_size = int(train_size * val_ratio)
        train_size -= val_size
        generator = torch.Generator()
        generator.manual_seed(42)  # for split reproducibility
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

def calculate_num_of_nodes(size):
    value = size - 2
    value //= 2
    value -= 2
    value //= 2
    return value*value*64*3

class LightingModel(L.LightningModule):
    def __init__(self, lr = 0.0002137, loss_fn = torch.nn.MSELoss()):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn', 'model'])
        
        self.encoder = Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1),  # 3x128x128 → 64x128x128
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2),                             # 64x64x64
            Conv2d(64, 128, kernel_size=3, padding=1),# 128x64x64
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2),                             # 128x32x32
        )

        self.decoder = Sequential(
            # Upsample
            # ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 64x64x64
            # BatchNorm2d(64),
            # ReLU(),
            # ConvTranspose2d(64, 3, kernel_size=2, stride=2),     # 3x128x128
            # Sigmoid(),  # HSV in [0,1]
            
            # Upsample to remove checkboard artifacts
            # można nearest -> bilinear: lepsza jakość bo patrzy w sąsiedztwie a nie tylko najbliższy ale wolniejsze
            Upsample(scale_factor=2, mode='nearest'),
            Conv2d(128, 64, kernel_size=3, padding=1),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),
            Conv2d(64, 3, kernel_size=3, padding=1),
            Sigmoid(), # HSV in [0,1]
        )
        
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def training_step(self, batch):
        rgb, hsv = batch
        outputs = self(rgb)
        loss = self.loss_fn(outputs, hsv)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        rgb, hsv = batch
        outputs = self(rgb)
        loss = self.loss_fn(outputs, hsv)
        self.log('val_loss', loss)

    def test_step(self, batch):
        rgb, hsv = batch
        outputs = self(rgb)
        loss = self.loss_fn(outputs, hsv)
        self.log('test_loss', loss)

    def predict_step(self, batch):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    