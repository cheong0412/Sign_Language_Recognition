# ------------------------------------------------------------------------
# 모듈 로딩
# ------------------------------------------------------------------------
import pandas as pd
import torch
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import matplotlib.pyplot as plt

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torchvision.transforms import functional as F


# A100 80GB의 절반인 0.5로 제한
# VRAM 메모리량 제한 설정
# torch.cuda.set_per_process_memory_fraction(0.5, device=0)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ✅ 디바이스 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 디바이스 확인
print(f"Using device: {device}", flush=True)

# # ------------------------------------------------------------------------
# # [1] 라벨 로딩 및 인덱싱 + 엑셀 저장
# # ------------------------------------------------------------------------
# # 원본 엑셀 로딩 (파일명, 한국어, 영어)
# label_df = pd.read_excel(r'G:\Team5\DATA\MP\Video_Labels_Mapped.xlsx')

# # 고유 한국어 라벨에 인덱스 부여
# unique_labels = sorted(label_df['Korean'].unique())
# label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

# # 라벨 인덱스 컬럼 추가
# label_df['Label_Index'] = label_df['Korean'].map(label_to_idx)

# # 엑셀로 저장: 파일명, 한국어, 영어, 라벨인덱스 포함
# label_df.to_excel(r'G:\Team5\DATA\MP\Video_Labels_Index.xlsx', index=False)
# print("-- 라벨 인덱스 포함된 엑셀 저장 완료: final_label_with_index.xlsx --")


# ------------------------------------------------------------------------
# [2] 엑셀에서 라벨 매핑 정보 불러오기
# ------------------------------------------------------------------------
label_df = pd.read_excel(r'G:\Team5\DATA\New_Final_Video_Label_Index.xlsx')

# 파일명 → 한국어 라벨 매핑
label_dict = {row['File_name']: row['Korean'] for _, row in label_df.iterrows()}

# 한국어 → 라벨 인덱스 매핑
label_to_idx = {row['Korean']: row['Label_Index'] for _, row in label_df.iterrows()}

print("-- 학습용 라벨 매핑 정보 불러오기 완료 --")



# ------------------------------------------------------------------------
# [2] 커스텀 데이터셋 정의
# ------------------------------------------------------------------------
class SignLanguageDataset(Dataset):
    def __init__(self, frames_dir, keypoints_dir, label_dict, label_to_idx, transform=None, seq_len=30):
        self.samples = []
        self.transform = transform
        self.seq_len = seq_len

        for video_name in os.listdir(frames_dir):
            frame_folder = os.path.join(frames_dir, video_name)
            kp_folder = os.path.join(keypoints_dir, video_name)

            if not os.path.isdir(frame_folder) or not os.path.isdir(kp_folder):
                continue

            label_name = label_dict.get(video_name)
            if label_name is None:
                continue

            frame_files = sorted(os.listdir(frame_folder))[:seq_len]
            keypoint_files = sorted(os.listdir(kp_folder))[:seq_len]

            if len(frame_files) < seq_len or len(keypoint_files) < seq_len:
                continue

            self.samples.append({
                'frames': [os.path.join(frame_folder, f) for f in frame_files],
                'keypoints': [os.path.join(kp_folder, f) for f in keypoint_files],
                'label': label_to_idx[label_name]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        images = []
        for img_path in sample['frames']:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = F.to_tensor(img)
            # if img.shape[2] != 1920:
            #     print(f'img => {img.shape},  {img_path}')
            images.append(img)

        keypoints = [np.load(p) for p in sample['keypoints']]
        keypoints = torch.tensor(np.array(keypoints), dtype=torch.float32)

        img_seq = torch.stack(images)
        label = sample['label']
        # print(f'keypoints.shape : {keypoints.shape}')
        # print(f'img_seq.shape : {img_seq.shape}')
        return img_seq, keypoints, label

# ------------------------------------------------------------------------
# [2-1] 이미지 전처리 정의 (정규화 포함) ----------> 성능 개선을 위해 추가
# ------------------------------------------------------------------------
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((360, 640)),  # 필요한 경우만 사용
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ------------------------------------------------------------------------
# [3] 데이터셋 로드
# ------------------------------------------------------------------------
print("Loading dataset...", flush=True)
frames_dir = r'G:\Team5\DATA\New_Final_Crop_Rotate_Resize'
keypoints_dir = r'G:\Team5\DATA\New_Final_npy'

dataset = SignLanguageDataset(
    frames_dir=frames_dir,
    keypoints_dir=keypoints_dir,
    label_dict=label_dict,
    label_to_idx=label_to_idx,
    transform = image_transform, # 성능 개선을 위해 추가함.
    seq_len=30
)
print(f"Dataset loaded. Total samples: {len(dataset)}", flush=True)

# ------------------------------------------------------------------------
# [4] 라벨별로 7:2:1로 나눔
# ------------------------------------------------------------------------
print("Splitting dataset by label (train/valid/test)...", flush=True)

label_to_indices = defaultdict(list)
for idx, sample in enumerate(dataset.samples):  # ⚠️ sample = {'frames': ..., 'label': ...}
    label = sample['label']
    label_to_indices[label].append(idx)

train_indices, valid_indices, test_indices = [], [], []

# 라벨별로 분할
for label, indices in label_to_indices.items():
    train, temp = train_test_split(indices, test_size=0.3, random_state=42)
    valid, test = train_test_split(temp, test_size=2/3, random_state=42)

    train_indices.extend(train)
    valid_indices.extend(valid)
    test_indices.extend(test)

    # # 🔽 각 샘플이 어느 split에 들어갔는지 출력
    # for i in train:
    #     video_name = os.path.basename(dataset.samples[i]['frames'][0].split(os.sep)[-2])
    #     print(f"→ {video_name} → train", flush=True)
    # for i in valid:
    #     video_name = os.path.basename(dataset.samples[i]['frames'][0].split(os.sep)[-2])
    #     print(f"→ {video_name} → valid", flush=True)
    # for i in test:
    #     video_name = os.path.basename(dataset.samples[i]['frames'][0].split(os.sep)[-2])
    #     print(f"→ {video_name} → test", flush=True)

print(f"Split complete. Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}", flush=True)

# ------------------------------------------------------------------------
# [5] DataLoader 생성
# ------------------------------------------------------------------------
print("Creating DataLoaders...", flush=True)
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=8, shuffle=True)
valid_loader = DataLoader(Subset(dataset, valid_indices), batch_size=8, shuffle=False)
test_loader  = DataLoader(Subset(dataset, test_indices),  batch_size=8, shuffle=False)
print("DataLoaders created.", flush=True)


# # ------------------------------------------------------------------------
# # [6] 시각화 및 저장
# # ------------------------------------------------------------------------
# print("Saving preview images...", flush=True)
# save_base = r'G:\Team5\DATA\MP\split_preview_no_transform'
# os.makedirs(save_base, exist_ok=True)

# def save_preview(indices, split_name, max_per_class=5):
#     saved = defaultdict(int)
#     for idx in indices:
#         img_seq, _, label = dataset[idx]
#         label_name = list(label_to_idx.keys())[list(label_to_idx.values()).index(label)]
#         save_dir = os.path.join(save_base, split_name, label_name)
#         os.makedirs(save_dir, exist_ok=True)

#         if saved[label_name] >= max_per_class:
#             continue

#         img = img_seq[0].permute(1, 2, 0).numpy()
#         save_path = os.path.join(save_dir, f"sample_{saved[label_name]}.png")
#         plt.imsave(save_path, img)
#         saved[label_name] += 1

# save_preview(train_indices, "train")
# print("Train preview saved.")
# save_preview(valid_indices, "valid")
# print("Valid preview saved.")
# save_preview(test_indices, "test")
# print("Test preview saved.")

print("All processing completed successfully.")
# for idx in range(10):
#     print(f'\n[{idx}]--------------')
#     dataset[idx]
    
# for img_seq, keypoints, labels in train_loader:
#     print(img_seq.shape, labels.shape)
#     break