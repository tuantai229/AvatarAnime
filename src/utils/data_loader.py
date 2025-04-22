# Xử lý dữ liệu
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None, mode='train'):
        """
        Dataset class cho CycleGAN
        
        Args:
            root_A: đường dẫn đến thư mục chứa ảnh domain A (người thật)
            root_B: đường dẫn đến thư mục chứa ảnh domain B (anime)
            transform: các biến đổi áp dụng cho ảnh
            mode: 'train', 'val' hoặc 'test'
        """
        self.transform = transform
        
        # Lấy đường dẫn tới tất cả các ảnh
        self.files_A = sorted([os.path.join(root_A, x) for x in os.listdir(root_A) 
                        if x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg')])
        self.files_B = sorted([os.path.join(root_B, x) for x in os.listdir(root_B) 
                        if x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg')])
        
        # Đảm bảo số ảnh trong cả hai domain đều hợp lệ
        assert len(self.files_A) > 0, f"No images found in {root_A}"
        assert len(self.files_B) > 0, f"No images found in {root_B}"
        
        print(f"Domain A: {len(self.files_A)} images")
        print(f"Domain B: {len(self.files_B)} images")
    
    def __getitem__(self, index):
        # Đảm bảo chọn một ảnh ngẫu nhiên từ domain B nếu có nhiều hơn domain A
        if len(self.files_A) < len(self.files_B):
            img_A = Image.open(self.files_A[index % len(self.files_A)])
            img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            img_A = Image.open(self.files_A[random.randint(0, len(self.files_A) - 1)])
            img_B = Image.open(self.files_B[index % len(self.files_B)])
        
        # Chuyển đổi ảnh grayscale sang RGB nếu cần
        if img_A.mode != 'RGB':
            img_A = img_A.convert('RGB')
        if img_B.mode != 'RGB':
            img_B = img_B.convert('RGB')
        
        # Áp dụng transform
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        
        return {'A': img_A, 'B': img_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

def get_transforms(load_size=160, crop_size=128):
    """
    Tạo các transform cho dữ liệu
    
    Args:
        load_size: kích thước để resize ảnh
        crop_size: kích thước để crop ảnh
    
    Returns:
        transform: chuỗi các biến đổi
    """
    transform = transforms.Compose([
        transforms.Resize(load_size, transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
    ])
    
    return transform

def create_dataloaders(person_root, anime_root, batch_size=1, num_workers=4, 
                      load_size=160, crop_size=128, mode='train'):
    """
    Tạo dataloaders cho training và validation
    
    Args:
        person_root: thư mục chứa ảnh người
        anime_root: thư mục chứa ảnh anime
        batch_size: kích thước batch
        num_workers: số worker để load dữ liệu
        load_size: kích thước để resize ảnh
        crop_size: kích thước để crop ảnh
        mode: 'train' hoặc 'val' hoặc 'test'
    
    Returns:
        dataloader: DataLoader object
    """
    transform = get_transforms(load_size, crop_size)
    dataset = ImageDataset(person_root, anime_root, transform, mode)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader