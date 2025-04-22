import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def process_image(image_path, output_path, target_size=(256, 256)):
    """
    Xử lý một ảnh: resize và chuẩn hóa
    """
    try:
        # Đọc ảnh với PIL
        img = Image.open(image_path)
        
        # Đảm bảo ảnh là RGB (chuyển đổi nếu là grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize ảnh
        img = img.resize(target_size, Image.LANCZOS)
        
        # Lưu ảnh đã xử lý
        img.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Lỗi xử lý ảnh {image_path}: {e}")
        return False

def process_dataset(input_dir, output_dir, target_size=(256, 256), max_workers=8):
    """
    Xử lý tất cả ảnh trong một thư mục
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách tất cả ảnh
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    successful = 0
    failed = 0
    
    # Xử lý song song
    print(f"Xử lý {len(image_files)} ảnh từ {input_dir}...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for img_file in image_files:
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.jpg')
            futures.append(executor.submit(process_image, input_path, output_path, target_size))
        
        # Theo dõi tiến trình
        for future in tqdm(futures):
            if future.result():
                successful += 1
            else:
                failed += 1
    
    print(f"Đã xử lý {successful} ảnh thành công, {failed} ảnh thất bại")

if __name__ == "__main__":
    # Định nghĩa kích thước mục tiêu - điều chỉnh tùy theo yêu cầu của mô hình
    TARGET_SIZE = (256, 256)
    
    # Xử lý dataset CelebA
    dataset_dirs = [
        ("data/CelebA_subset_split/train", "data/processed/CelebA/train"),
        ("data/CelebA_subset_split/val", "data/processed/CelebA/val"),
        ("data/CelebA_subset_split/test", "data/processed/CelebA/test"),
        ("data/AnimeFace_subset_split/train", "data/processed/AnimeFace/train"),
        ("data/AnimeFace_subset_split/val", "data/processed/AnimeFace/val"),
        ("data/AnimeFace_subset_split/test", "data/processed/AnimeFace/test")
    ]
    
    for input_dir, output_dir in dataset_dirs:
        process_dataset(input_dir, output_dir, TARGET_SIZE)