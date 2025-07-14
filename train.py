# train.py

from CNNLSTM_3D import CNNLSTMModel_3D
from Video_Dataset import SignLanguageDataset  # <- 데이터셋 클래스만 가져옴

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os

# 모델 저장 폴더 생성
save_dir = './2_NEW_3D_Layer6_Batch11_CNN_LSTM'
os.makedirs(save_dir, exist_ok=True)


# # A100 80GB의 절반인 0.5로 제한
# # VRAM 메모리량 제한 설정
# torch.cuda.set_per_process_memory_fraction(0.5, device=0)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 디바이스 확인
# print(f"Using device: {device}", flush=True)

# ------------------------------------------------------------------------
# (transform 추가함.)
# ------------------------------------------------------------------------
from torchvision import transforms  
# 이미지 전처리 정의
image_transform = transforms.Compose([
    # transforms.Resize((720, 1280)),  # 모델 입력에 맞게 resize
    # transforms.Resize((360, 640)), # 이미지 줄이되 비율은 유지
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # 정규화
])

# ------------------------------------------------------------------------
# [1] 라벨 불러오기 (엑셀에서)
# ------------------------------------------------------------------------
label_df = pd.read_excel(r'G:\Team5\DATA\New_Final_Video_Label_Index.xlsx')
label_dict = label_df.set_index('File_name')['Korean'].to_dict()
label_to_idx = label_df.set_index('Korean')['Label_Index'].to_dict()

# ------------------------------------------------------------------------
# [2] Dataset 생성 (transform 추가함.)
# ------------------------------------------------------------------------
dataset = SignLanguageDataset(
    frames_dir=r'G:\Team5\DATA\New_Final_Crop_Rotate_Resize',
    keypoints_dir=r'G:\Team5\DATA\New_Final_npy',
    label_dict=label_dict,
    label_to_idx=label_to_idx,
    transform = image_transform
)

# ------------------------------------------------------------------------
# [3] Split: 라벨별로 7:2:1
# ------------------------------------------------------------------------
label_to_indices = defaultdict(list)
for idx, sample in enumerate(dataset.samples):
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
    
print(f"Split complete. Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}", flush=True)

# ------------------------------------------------------------------------
# [4] DataLoader 생성
# ------------------------------------------------------------------------
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=11, shuffle=True)
valid_loader = DataLoader(Subset(dataset, valid_indices), batch_size=11, shuffle=False)
test_loader  = DataLoader(Subset(dataset, test_indices),  batch_size=11, shuffle=False)
print("DataLoaders created.", flush=True)
# ------------------------------------------------------------------------
# [5] 모델 준비
# ------------------------------------------------------------------------
# torch.cuda.set_per_process_memory_fraction(0.5, device=0)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ✅ 디바이스 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = len(label_to_idx)
model = CNNLSTMModel_3D(num_classes=num_classes).to(device)

### ✅ 저장한 모델 로딩 ✅  - 학습시킬 때 끊겼을 경우!!!!!
state_dict = torch.load(r'G:\Team5\NEW_3D_Layer6_Batch11_CNN_LSTM\best_weight_ep17_trainL1.11_trainA53.01_valL1.18_valA50.62.pt')
model.load_state_dict(state_dict)

# ------------------------------------------------------------------------
# [6] 손실함수, 옵티마이저
# ------------------------------------------------------------------------
from collections import Counter

# 클래스별 샘플 수 계산
class_counts = Counter([sample['label'] for sample in dataset.samples])

# 가중치 계산 (전체 샘플 수 / 각 클래스 샘플 수)
num_samples = sum(class_counts.values())
weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# CrossEntropyLoss에 클래스 가중치 적용
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# ------------------------------------------------------------------------
# [7] 학습 루프 + 성능 출력 + Best 모델 저장
# ------------------------------------------------------------------------ 
num_epochs = 100
best_valid_loss = float('inf')  # Best 모델 판단 기준

for epoch in range(num_epochs):
    print(f"\n🔁 [Epoch {epoch+1}/{num_epochs}] ----------------------------")
    
    # ------- Train -------
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (img_seq, keypoints, labels) in enumerate(train_loader):
        print(f"🌀 [Train] Epoch {epoch+1}, Step {i+1}/{len(train_loader)} - Loading batch...", flush=True)

        img_seq, labels = img_seq.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(img_seq, keypoints)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 🔹 스텝별 출력
        print(f"  ▶ Step {i+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}", flush=True)

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"📊 Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%", flush=True)

    # ------- Validation -------
    model.eval()
    valid_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for img_seq, keypoints, labels in valid_loader:
            img_seq, labels = img_seq.to(device), labels.to(device)
            outputs = model(img_seq, keypoints)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_valid_loss = valid_loss / len(valid_loader)
    valid_acc = 100 * val_correct / val_total
    print(f"✅ Valid Loss: {avg_valid_loss:.4f} | Accuracy: {valid_acc:.2f}%", flush=True)

    # ------- Best 모델 저장 -------
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        filename = (
            f'best_weight_ep{epoch+1}_'
            f'trainL{train_loss:.2f}_trainA{train_acc:.2f}_'
            f'valL{avg_valid_loss:.2f}_valA{valid_acc:.2f}.pt'
        )
        save_path = os.path.join(save_dir, filename)
        torch.save(model.state_dict(), save_path)
        print(f"📦 Best model saved at epoch {epoch+1}! [{filename}]", flush=True)