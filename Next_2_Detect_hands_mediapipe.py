import cv2
import os
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import torch

# ✅ 디바이스 확인
torch.cuda.set_per_process_memory_fraction(0.5, device=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ✅ 입력 및 출력 경로 설정
FRAMES_DIR = r'G:\Team5\DATA\Final_Data\Word\New_Final_Crop_Rotate_Resize'  # 프레임 이미지가 있는 루트 폴더
KEYPOINTS_DIR = r'G:\Team5\DATA\Final_Data\Word\Final_npy'               # 관절 좌표 저장 폴더
os.makedirs(KEYPOINTS_DIR, exist_ok=True)

# ✅ MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# ✅ 영상 폴더 리스트
video_names = sorted(os.listdir(FRAMES_DIR))
print(f"🎞 총 {len(video_names)}개 영상 폴더 처리 시작", flush=True)

# ✅ 이어서 실행하면서 실시간 로그 출력
for video_name in video_names:
    frame_folder = os.path.join(FRAMES_DIR, video_name)
    keypoint_output_dir = os.path.join(KEYPOINTS_DIR, video_name)
    os.makedirs(keypoint_output_dir, exist_ok=True)

    print(f'\n📂 현재 처리 중: {video_name}')
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.lower().endswith('.jpg')])

    for frame_file in frame_files:
        npy_expected = frame_file.replace('.jpg', f'_hand0.npy')  # 최소 하나의 손 파일 기준
        npy_path = os.path.join(keypoint_output_dir, npy_expected)

        if os.path.exists(npy_path):
            continue  # 이미 처리된 프레임은 건너뛰기

        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"⚠️ 이미지 로드 실패: {frame_path}")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                npy_name = frame_file.replace('.jpg', f'_hand{hand_idx}.npy')
                np_file = os.path.join(keypoint_output_dir, npy_name)
                np.save(npy_file := np_file, np.array(coords))
                print(f"✅ 저장 완료: {npy_file}")
        else:
            print(f"❌ 손 미검출: {frame_path}")

hands.close()
print("\n✅ 중단 지점부터 이어서 관절 추출 완료")
