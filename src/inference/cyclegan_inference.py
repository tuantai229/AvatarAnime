# Script để chạy inference
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import sys
import argparse
from PIL import Image

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.cyclegan.networks import Generator

def load_image(image_path, size=256):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0)

def inference(args):
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
    
    # Tạo mô hình Generator
    generator = Generator().to(device)
    
    # Load weights
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.eval()
    
    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Kiểm tra nếu input_path là thư mục
    if os.path.isdir(args.input_path):
        image_paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        image_paths = [args.input_path]
    
    # Xử lý từng ảnh
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        # Load ảnh
        input_image = load_image(img_path, args.size).to(device)
        
        # Chạy inference
        with torch.no_grad():
            output_image = generator(input_image)
        
        # Lưu output
        save_image(output_image, os.path.join(args.output_dir, f"{base_name}_anime.png"), normalize=True)
        if args.save_input:
            save_image(input_image, os.path.join(args.output_dir, f"{base_name}_input.png"), normalize=True)
        
        print(f"Processed {filename}")
    
    print(f"All images processed and saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with CycleGAN")
    
    parser.add_argument('--input_path', type=str, required=True, 
                        help='Path to input image or directory')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to generator model checkpoint')
    parser.add_argument('--output_dir', type=str, default="results/inference", 
                        help='Directory to save results')
    parser.add_argument('--size', type=int, default=256, 
                        help='Size of input image')
    parser.add_argument('--save_input', action='store_true', 
                        help='Save input image alongside output')
    parser.add_argument('--cpu', action='store_true', 
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    inference(args)