import os
import random
import shutil
import pandas as pd
from tqdm import tqdm

# Thiết lập đường dẫn
celeba_dir = "data/CelebA"
output_dir = "data/CelebA_subset"
target_samples = 10000

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Đọc file thuộc tính (nếu có)
try:
    attr_file = os.path.join(celeba_dir, "list_attr_celeba.csv")
    attributes = pd.read_csv(attr_file)
    has_attr = True
except:
    print("Không tìm thấy file thuộc tính, sẽ chọn mẫu ngẫu nhiên đơn thuần")
    has_attr = False

# Lấy danh sách tất cả ảnh từ thư mục img_align_celeba
img_dir = os.path.join(celeba_dir, "img_align_celeba/img_align_celeba")
all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

if has_attr:
    # Chọn mẫu có phân tầng dựa trên giới tính (đảm bảo cân bằng)
    male_attr = 'Male'  # Cột thuộc tính giới tính trong file
    males = attributes[attributes[male_attr] == 1]['image_id'].tolist()
    females = attributes[attributes[male_attr] == -1]['image_id'].tolist()
    
    # Chọn 5000 nam và 5000 nữ
    selected_males = random.sample(males, min(5000, len(males)))
    selected_females = random.sample(females, min(5000, len(females)))
    selected_images = selected_males + selected_females
else:
    # Nếu không có file thuộc tính, chọn ngẫu nhiên
    selected_images = random.sample(all_images, min(target_samples, len(all_images)))

# Sao chép ảnh đã chọn vào thư mục đầu ra
print(f"Đang sao chép {len(selected_images)} ảnh từ CelebA...")
for img_name in tqdm(selected_images):
    src_path = os.path.join(img_dir, img_name)
    dst_path = os.path.join(output_dir, img_name)
    shutil.copy(src_path, dst_path)

print(f"Đã tạo tập dữ liệu con với {len(selected_images)} ảnh từ CelebA tại {output_dir}")

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