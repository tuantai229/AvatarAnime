import os
import random
import shutil
from tqdm import tqdm

# Thiết lập đường dẫn
anime_dir = "data/AnimeFace"  # Thư mục gốc
output_dir = "data/AnimeFace_subset"
target_samples = 1000

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Lấy danh sách tất cả ảnh trực tiếp từ thư mục AnimeFace (không có thư mục images)
all_images = [f for f in os.listdir(anime_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Chọn ngẫu nhiên 1000 ảnh
selected_images = random.sample(all_images, min(target_samples, len(all_images)))

# Sao chép ảnh đã chọn vào thư mục đầu ra
print(f"Đang sao chép {len(selected_images)} ảnh từ AnimeFace...")
for img_name in tqdm(selected_images):
    src_path = os.path.join(anime_dir, img_name)
    dst_path = os.path.join(output_dir, img_name)
    shutil.copy(src_path, dst_path)

print(f"Đã tạo tập dữ liệu con với {len(selected_images)} ảnh từ AnimeFace tại {output_dir}")

# Chia tập dữ liệu thành train/val/test
def split_dataset(dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Chia một thư mục ảnh thành các tập train/val/test
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Tổng các tỷ lệ phải bằng 1"
    
    # Tạo thư mục đầu ra
    train_dir = os.path.join(dataset_dir + "_split", "train")
    val_dir = os.path.join(dataset_dir + "_split", "val")
    test_dir = os.path.join(dataset_dir + "_split", "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Lấy tất cả ảnh
    all_images = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Chia tập dữ liệu
    train_val, test = train_test_split(all_images, test_size=test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
    
    # Sao chép ảnh vào các thư mục tương ứng
    for img in train:
        shutil.copy(os.path.join(dataset_dir, img), os.path.join(train_dir, img))
    
    for img in val:
        shutil.copy(os.path.join(dataset_dir, img), os.path.join(val_dir, img))
    
    for img in test:
        shutil.copy(os.path.join(dataset_dir, img), os.path.join(test_dir, img))
    
    print(f"Chia tập dữ liệu hoàn tất:")
    print(f"  Train: {len(train)} ảnh")
    print(f"  Validation: {len(val)} ảnh")
    print(f"  Test: {len(test)} ảnh")

# Thêm import
from sklearn.model_selection import train_test_split

# Chia tập dữ liệu
if __name__ == "__main__":
    split_dataset(output_dir)