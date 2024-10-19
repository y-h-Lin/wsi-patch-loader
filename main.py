import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.transforms import Compose, RandRotate90, RandFlip, RandZoom, RandAdjustContrast, ToTensor
from cucim import CuImage

class WSILoader:
    def __init__(self, file_path, level=0, patch_size=1024):
        self.wsi = CuImage(file_path)
        self.level = level
        self.patch_size = patch_size

    def get_patch(self, x, y):
        patch = self.wsi.read_region((x, y), self.level, (self.patch_size, self.patch_size))
        return np.array(patch)

class WSIDataset(Dataset):
    def __init__(self, wsi_files, patch_size=1024, target_patches=1000):
        self.wsi_files = wsi_files
        self.patch_size = patch_size
        self.target_patches = target_patches
        self.patches_per_wsi = self._calculate_patches_per_wsi()

    def _calculate_patches_per_wsi(self):
        patches_per_wsi = []
        for file_path in self.wsi_files:
            loader = WSILoader(file_path, patch_size=self.patch_size)
            wsi_width, wsi_height = loader.wsi.dimensions
            stride = self._calculate_stride(wsi_width, wsi_height)
            x_patches = (wsi_width - self.patch_size) // stride + 1
            y_patches = (wsi_height - self.patch_size) // stride + 1
            patches_per_wsi.append(x_patches * y_patches)
        return patches_per_wsi

    def _calculate_stride(self, wsi_width, wsi_height):
        area = wsi_width * wsi_height
        stride = int(np.sqrt(area / self.target_patches))
        return max(stride, self.patch_size // 2)

    def __len__(self):
        return sum(self.patches_per_wsi)

    def __getitem__(self, idx):
        wsi_idx = 0
        while idx >= self.patches_per_wsi[wsi_idx]:
            idx -= self.patches_per_wsi[wsi_idx]
            wsi_idx += 1
        
        file_path = self.wsi_files[wsi_idx]
        loader = WSILoader(file_path, patch_size=self.patch_size)
        wsi_width, wsi_height = loader.wsi.dimensions
        stride = self._calculate_stride(wsi_width, wsi_height)
        
        x_patches = (wsi_width - self.patch_size) // stride + 1
        x = (idx % x_patches) * stride
        y = (idx // x_patches) * stride
        
        x = min(x, wsi_width - self.patch_size)
        y = min(y, wsi_height - self.patch_size)
        
        patch = loader.get_patch(x, y)
        return {"image": patch, "name": f"{os.path.basename(file_path)}_{x}_{y}"}

def get_transforms(is_strong=False):
    if is_strong:
        return Compose([
            RandRotate90(prob=0.5),
            RandFlip(prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            RandAdjustContrast(prob=0.5),
            ToTensor()
        ])
    else:
        return Compose([
            RandRotate90(prob=0.5),
            RandFlip(prob=0.5),
            ToTensor()
        ])

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Set up dataset
    wsi_dir = "/path/to/wsi/files"
    wsi_files = [os.path.join(wsi_dir, f) for f in os.listdir(wsi_dir) if f.endswith('.ndpi')]
    
    dataset = WSIDataset(wsi_files, transform=get_transforms(is_strong=True))
    
    # Set up DataLoader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        sampler=sampler
    )
    
    # Set up model (replace this with your actual model)
    model = YourModel().to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            images = batch['image'].to(rank)
            # Assume you have labels
            # labels = batch['label'].to(rank)
            
            optimizer.zero_grad()
            outputs = model(images)
            # Calculate loss
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} completed")
    
    cleanup()

if __name__ == "__main__":
    world_size = 24  # 3 DGX H100 machines, 8 GPUs each
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
