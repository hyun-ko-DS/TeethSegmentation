import os

# 경로 설정
print("\n" + "="*60)
print("Setting Up Paths")
print("="*60)

extract_dir = '/data/alphadent_preprocessed'

TRAIN_IMAGES_PATH = os.path.join(extract_dir, 'images', 'train')
VALID_IMAGES_PATH = os.path.join(extract_dir, 'images', 'valid')
TEST_IMAGES_PATH = os.path.join(extract_dir, 'images', 'test')
TRAIN_LABELS_PATH = os.path.join(extract_dir, 'labels', 'train')
VALID_LABELS_PATH = os.path.join(extract_dir, 'labels', 'valid')

OUTPUT_DIR = './alphadent_yolo'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 경로 확인
paths_info = {
    'Train Images': TRAIN_IMAGES_PATH,
    'Valid Images': VALID_IMAGES_PATH,
    'Test Images': TEST_IMAGES_PATH,
    'Train Labels': TRAIN_LABELS_PATH,
    'Valid Labels': VALID_LABELS_PATH
}

print("\nPath verification:")
for name, path in paths_info.items():
    exists = os.path.exists(path)
    status = "OK" if exists else "NOT FOUND"
    print(f"  {name}: {status}")
    if not exists:
        raise FileNotFoundError(f"{name} not found: {path}")

# 파일 개수 확인
train_count = len([f for f in os.listdir(TRAIN_IMAGES_PATH) if f.endswith('.png')])
valid_count = len([f for f in os.listdir(VALID_IMAGES_PATH) if f.endswith('.png')])
test_count = len([f for f in os.listdir(TEST_IMAGES_PATH) if f.endswith('.png')])

print(f"\nDataset Statistics:")
print(f"  Training images: {train_count}")
print(f"  Validation images: {valid_count}")
print(f"  Test images: {test_count}")