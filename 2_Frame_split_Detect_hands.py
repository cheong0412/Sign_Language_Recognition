import cv2
import os
from tqdm import tqdm

# ✅ 경로 설정
PROJECT_ROOT = r'G:\Team5\DATA'
VIDEO_DIR = os.path.join(PROJECT_ROOT, 'new_add_word')  # 증상별 폴더 포함
FRAMES_DIR = os.path.join(PROJECT_ROOT, 'MP', 'New_frames')  # 저장 위치

# ✅ 출력 폴더 생성
os.makedirs(FRAMES_DIR, exist_ok=True)

# ✅ 처리할 모든 영상 수 파악
all_video_paths = []
for root, _, files in os.walk(VIDEO_DIR):
    for file in files:
        if file.lower().endswith('.mp4'):
            all_video_paths.append(os.path.join(root, file))

print(f"🎞 총 {len(all_video_paths)}개 영상 처리 시작", flush=True)

for video_path in tqdm(all_video_paths, desc="🎬 영상 처리 중"):
    filename = os.path.basename(video_path)
    basename = os.path.splitext(filename)[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 열기 실패: {video_path}", flush=True)
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"⚠️ 프레임 없음: {video_path}", flush=True)
        continue

    # 출력 폴더 (영상별로 구분)
    frame_output_dir = os.path.join(FRAMES_DIR, basename)
    os.makedirs(frame_output_dir, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_save_path = os.path.join(frame_output_dir, f'frame_{frame_idx:04d}.jpg')
        cv2.imwrite(frame_save_path, frame)

        frame_idx += 1

    cap.release()

print("✅ 모든 영상 프레임 저장 완료")