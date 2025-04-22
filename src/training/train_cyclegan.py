# Script huấn luyện
import torch
import torch.nn as nn
import torchvision.utils as vutils
import time
import os
import datetime
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.cyclegan.model import CycleGANModel
from src.utils.data_loader import create_dataloaders

def train(args):
    # Tạo thư mục để lưu kết quả
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(args.output_dir, f"cyclegan_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Đặt device
    # Đặt device
    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:0")
        print(f"Using device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    
    # Tạo dataloaders
    train_dataloader = create_dataloaders(
        args.person_root, args.anime_root, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        mode='train'
    )
    
    val_dataloader = create_dataloaders(
        args.person_val_root, args.anime_val_root, 
        batch_size=1,  # Validation luôn dùng batch_size=1
        num_workers=args.num_workers,
        mode='val'
    )
    
    # Khởi tạo mô hình
    model = CycleGANModel(
        device=device,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity
    )
    
    # Tạo fixed batch để visualization
    fixed_samples = next(iter(val_dataloader))
    fixed_A = fixed_samples['A']
    fixed_B = fixed_samples['B']
    
    # Training loop
    total_steps = 0
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Training phase
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            real_A = batch['A']
            real_B = batch['B']
            
            model.set_input(real_A, real_B)
            model.optimize_parameters()
            
            total_steps += 1
            
            # Hiển thị thông tin loss
            if i % args.print_freq == 0:
                print(f"[{epoch+1}/{args.epochs}][{i}/{len(train_dataloader)}] "
                      f"Loss_D_A: {model.loss_D_A:.4f} Loss_D_B: {model.loss_D_B:.4f} "
                      f"Loss_G_A2B: {model.loss_G_A2B:.4f} Loss_G_B2A: {model.loss_G_B2A:.4f} "
                      f"Loss_cycle_A: {model.loss_cycle_A:.4f} Loss_cycle_B: {model.loss_cycle_B:.4f}")
        
        # Tạo và lưu ảnh từ fixed samples
        with torch.no_grad():
            model.set_input(fixed_A, fixed_B)
            model.forward()
            fake_B = model.fake_B
            rec_A = model.rec_A
            fake_A = model.fake_A
            rec_B = model.rec_B
            
            # Đảm bảo tất cả tensor đều trên cùng device (CPU) trước khi nối
            fixed_A_cpu = fixed_A.cpu()
            fake_B_cpu = fake_B.cpu()
            rec_A_cpu = rec_A.cpu()
            fixed_B_cpu = fixed_B.cpu()
            fake_A_cpu = fake_A.cpu()
            rec_B_cpu = rec_B.cpu()
            
            # Tạo grid ảnh
            images = [fixed_A_cpu, fake_B_cpu, rec_A_cpu, fixed_B_cpu, fake_A_cpu, rec_B_cpu]
            image_grid = vutils.make_grid(torch.cat(images, 0), nrow=args.batch_size, padding=2, normalize=True)
            vutils.save_image(image_grid, os.path.join(output_dir, "images", f"epoch_{epoch+1}.png"))
        
        # Lưu model checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            model.save_models(os.path.join(output_dir, "checkpoints"), epoch + 1)
        
        # Tính thời gian cho mỗi epoch
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
    
    # Lưu đồ thị loss
    plt.figure(figsize=(10, 5))
    plt.plot(model.losses['D_A'], label='D_A')
    plt.plot(model.losses['D_B'], label='D_B')
    plt.plot(model.losses['G_A2B'], label='G_A2B')
    plt.plot(model.losses['G_B2A'], label='G_B2A')
    plt.legend()
    plt.title('Discriminator and Generator Losses')
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    
    plt.figure(figsize=(10, 5))
    plt.plot(model.losses['cycle_A'], label='cycle_A')
    plt.plot(model.losses['cycle_B'], label='cycle_B')
    plt.plot(model.losses['identity_A'], label='identity_A')
    plt.plot(model.losses['identity_B'], label='identity_B')
    plt.legend()
    plt.title('Cycle and Identity Losses')
    plt.savefig(os.path.join(output_dir, "cycle_identity_loss_plot.png"))
    
    print(f"Training completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CycleGAN model")
    
    # Data related arguments
    parser.add_argument('--person_root', type=str, default="data/processed/CelebA/train", 
                        help='Root directory for person training images')
    parser.add_argument('--anime_root', type=str, default="data/processed/AnimeFace/train", 
                        help='Root directory for anime training images')
    parser.add_argument('--person_val_root', type=str, default="data/processed/CelebA/val", 
                        help='Root directory for person validation images')
    parser.add_argument('--anime_val_root', type=str, default="data/processed/AnimeFace/val", 
                        help='Root directory for anime validation images')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Weight for cycle loss')
    parser.add_argument('--lambda_identity', type=float, default=5.0, help='Weight for identity loss')
    
    # Other options
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--print_freq', type=int, default=100, help='Frequency of printing training info')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency of saving models')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save results')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    train(args)