import torch
import torch.nn as nn

# A100 80GB의 절반인 0.5로 제한
# VRAM 메모리량 제한 설정
# torch.cuda.set_per_process_memory_fraction(0.5, device=0)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ✅ 디바이스 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"🖥 사용 디바이스: {device}", flush=True)

# 디바이스 확인
print(f"Using device: {device}", flush=True)


# ------------------------------------------------------------------------
#  [모델]
# ------------------------------------------------------------------------
class CNNLSTMModel_3D(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTMModel_3D, self).__init__()
        self.cnn = nn.Sequential(            #[Batch_size, Channel, Time stemps, Height, Width]
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(3, 3, 3), padding=1),  # (B*T, 3, 720, 1280)
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # (360, 640)
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2)),  # (180, 320)

            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2)),  # (90, 160)

            nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2)),  # (45, 80)
            
            nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2)),  # (45, 80)
            
            nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2)),  # (45, 80)
        )
        
        # self.flat_size = 32 * 180 * 320
        # self.flat_size = 32 * 90 * 160
        # 자동 계산 방식 사용
        self.flat_size = None
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 16, 480, 640)  # 입력 이미지와 동일한 사이즈
            dummy_out = self.cnn(dummy)
            B, C, T, H, W = dummy_out.size()
            self.flat_size = C * H * W
        
        self.lstm = nn.LSTM(input_size=self.flat_size + 63, hidden_size=256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x, keypoints):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # → [B, C, T, H, W]
        x = self.cnn(x)  # Conv3D 통과 → [B, C, T, H, W]

        # CNN 출력 flatten → [B, T, feature_dim]
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, -1)

        # 🔽 keypoints와 결합
        keypoints = keypoints.to(x.device)  # [B, T, 63]
        x = torch.cat([x, keypoints], dim=-1)  # [B, T, flat_size + 63]

        _, (h_n, _) = self.lstm(x)  # LSTM 출력
        h_n = h_n.permute(1, 0, 2).contiguous().view(B, -1)  # [B, 256]
        out = self.fc(h_n)  # [B, num_classes]
        return out
