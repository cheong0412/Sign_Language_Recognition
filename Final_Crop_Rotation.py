import cv2
import os
from tqdm import tqdm

# 경로 설정
input_dir = r"G:\Team5\DATA\New_frames"  # 원본 프레임 폴더 (하위 폴더 포함)
output_dir = r"G:\Team5\DATA\ALL_New_Final_Crop_Rotate_Resize"  # 출력 폴더
os.makedirs(output_dir, exist_ok=True)

# crop 크기 및 리사이즈 설정
CROP_WIDTH = 1180
CROP_HEIGHT = 1000
RESIZE_TO = (640, 480)

def center_crop_fixed(img, crop_w=CROP_WIDTH, crop_h=CROP_HEIGHT):
    h, w = img.shape[:2]
    x1 = max((w - crop_w) // 2, 0)
    y1 = max((h - crop_h) // 2, 0)
    return img[y1:y1+crop_h, x1:x1+crop_w]

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

# 모든 하위 폴더 탐색
for foldername in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, foldername)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n📁 처리 중: {foldername}")

    # 저장할 폴더 준비
    orig_save_dir = os.path.join(output_dir, foldername)
    rot1_save_dir = os.path.join(output_dir, f"{foldername}_Rot_1")
    rot2_save_dir = os.path.join(output_dir, f"{foldername}_Rot_2")
    os.makedirs(orig_save_dir, exist_ok=True)
    os.makedirs(rot1_save_dir, exist_ok=True)
    os.makedirs(rot2_save_dir, exist_ok=True)

    # 각 프레임 이미지 처리
    for filename in tqdm(sorted(os.listdir(folder_path))):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 이미지 로딩 실패: {img_path}")
            continue

        # crop + resize (원본)
        cropped = center_crop_fixed(img)
        resized = cv2.resize(cropped, RESIZE_TO)

        # 파일 저장 (원본)
        save_path = os.path.join(orig_save_dir, filename)
        cv2.imwrite(save_path, resized)

        # 회전 및 저장
        base_name, ext = os.path.splitext(filename)

        # 왼쪽 회전
        rotated_left = rotate_image(cropped, -10)
        resized_left = cv2.resize(rotated_left, RESIZE_TO)
        cv2.imwrite(os.path.join(rot1_save_dir, f"{base_name}_Rot_1{ext}"), resized_left)

        # 오른쪽 회전
        rotated_right = rotate_image(cropped, 10)
        resized_right = cv2.resize(rotated_right, RESIZE_TO)
        cv2.imwrite(os.path.join(rot2_save_dir, f"{base_name}_Rot_2{ext}"), resized_right)

print("\n✅ 모든 프레임 → crop → 회전 → 폴더 분리 저장 완료!")
