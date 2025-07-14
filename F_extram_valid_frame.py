## -----------------------------------------------------------
## 흐릿한 프레임, 중복 프레임, 손 모양 변화 없는 프레임 등 필터링 
## - is_atention_pose()     : 차렷 자세 여부 판별
## - extract_valid_frames() : 유사 위치 키포인트 이미지 제거 
## - extract_all_frames()   : 
## - playing_files()        : 테스트용 함수, 유효 프레임 재생
## - change_filename()      : 파일명 "KETI_SL_" => "SL" 변경
## -----------------------------------------------------------

## -----------------------------------------------------------
## 모듈로딩
## -----------------------------------------------------------
import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from F_config import VIDEO_DIR, MEDI_FRAMES


## -----------------------------------------------------------
## 전역변수
## -----------------------------------------------------------
# Mediapipe 설정
MP_HANDS = mp.solutions.hands
HANDS    = MP_HANDS.Hands( static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5 )

LOG_FILE = './log.txt'

## -----------------------------------------------------------
## 함수기능 :  차렷 자세 체크 
## 함수이름 :  is_atention_pose
## 매개변수 :  landmarks       - 손 랜드마크 즉, 키포인트
##            threshold=0.8   - 손 랜드마크 임계값
## 함수결과 :  차렷 여부 True/False
## -----------------------------------------------------------
def is_atention_pose(landmarks, threshold=0.8):
    ##- 손의 평균 y값이 threshold보다 크면 차렷 자세로 판단
    if landmarks is None or len(landmarks) == 0:
        return True
    mean_y = np.mean([pt.y for pt in landmarks])
    return mean_y > threshold

## -----------------------------------------------------------
## 함수기능 :  수어 동영상에서 유효한 프레임만 추출해서 저장 
##            차렷자세 제외, 이전과 동일 위치 프레임 제외 
## 함수이름 :  extract_valid_frames
## 매개변수 :  video_path         - 수어 동영상 경로
##            save_dir           - 유효 프레임 저장 폴더경로    
##            video_name         - 수어 동영상 이름
##            min_movement=0.01  - 최소 감지 정도 임계값 
## 함수결과 :  유효한 프레임 저장 
## -----------------------------------------------------------
def extract_valid_frames(video_path, save_dir, video_name, min_movement=0.01):
    
    print(f'video_path: {video_path}\nsave_dir: {save_dir}\nvideo_name: {video_name}')

    ##- 수어 동영상 로딩 
    cap = cv2.VideoCapture(str(video_path))
    prev_kp = None
    count = 0
    valid_count = 0

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ##- 수어 동영상에서 유효한 프레임 추출
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            print(f'[{video_name}] 로딩 실패')
            break

        # Mediapipe 처리를 위해 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = HANDS.process(image_rgb)

        if not results.multi_hand_landmarks:
            count += 1
            continue

        # 각 손의 keypoints 평균
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            if is_atention_pose(hand_landmarks.landmark):
                continue  # 차렷 자세 skip
            keypoints.extend([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

        if len(keypoints) == 0:
            count += 1
            continue

        # 움직임 변화 체크
        if prev_kp is not None:
            #- 이전과 현재 키포인트 데이터 추출
            prev_np, curr_np = np.array(prev_kp), np.array(keypoints)

            #- 이전과 현재 키포트인 변화 여부 체크 
            if prev_np.shape == curr_np.shape:
                diff = np.linalg.norm(curr_np - prev_np, axis=1).mean()
                if diff < min_movement:
                    count += 1
                    continue
        prev_kp = keypoints

        # 유효한 프레임을 이미지로 저장 
        save_path = save_dir / f"{video_path.stem}_frame_{count:04d}.jpg"
        print(f'save_path: {save_path}')
        cv2.imwrite(str(save_path), frame)
        valid_count += 1
        count += 1

    cap.release()
    with open(LOG_FILE, mode='a', encoding='utf-8') as logFile:
        print(f"[{video_path.name}] 저장된 유효 프레임 수: {valid_count}개")
        logFile.write(f"[ VIDEO_PATH ] {video_path}\t")
        logFile.write(f"저장된 유효 프레임 수: {valid_count}개\n")


## -----------------------------------------------------------
## 함수기능 : 동일 수어 동영상에 해당하는 프레임만 추출하여 재생 
## 함수이름 : playing_files
## 매개변수 : 없음
## -----------------------------------------------------------
def playing_files():
    frame_root = Path(MEDI_FRAMES) 

    for sign_folder in frame_root.iterdir():
            
        for frame_file in sign_folder.glob("*.jpg"):
            img = cv2.imread(str(frame_file))
            cv2.imshow("SIGN", img)
            key = cv2.waitKey(int(300))    # 300ms 즉, 0.3sec
            if key == ord('x'):
                break
        cv2.destroyAllWindows()
        break


## -----------------------------------------------------------
## 함수기능 : 파일명 변경 "KETI_SL_" => "SL"
## 함수이름 : change_filename
## 매개변수 : 없음
## 함수결과 : 없음
## -----------------------------------------------------------
def change_filename():
    video_root = Path(VIDEO_DIR) 

    for sign_folder in video_root.iterdir():
        print(f'sign_folder : {sign_folder}')
        if sign_folder.is_dir():
            for video_file in sign_folder.glob("*.avi"):
                print(f' video_file : {type(video_file)} {video_file}')
                video_file = str(video_file)
                video_newfile = video_file.replace("KETI_SL_", "SL")
                print(f' video_file : {type(video_file)} {video_file}')
                os.rename(video_file, video_newfile)

## -----------------------------------------------------------
## 함수기능 : 모든 수어 동영상에서 유효 프레임 추출 및 저장 
## 함수이름 : extract_all_frames
## 매개변수 : 없음
## 함수결과 : 유효 프레임을 저장 
## -----------------------------------------------------------
def extract_all_frames():
    video_root = Path(VIDEO_DIR) 
    frame_root = Path(MEDI_FRAMES)

    for sign_folder in video_root.iterdir():
        print(f'sign_folder : {sign_folder}\nframe_root  : {frame_root}')

        if sign_folder.is_dir():
            with open(LOG_FILE, mode='a', encoding='utf-8') as logFile:
                logFile.write(f"\n[{sign_folder.name}] 영상별 저장된 유효 프레임 수 확인----------------\n")

            for video_file in sign_folder.glob("*.avi"):
                output_dir = frame_root / sign_folder.name
                print(f'output_dir  : {output_dir}')
                extract_valid_frames(video_file, output_dir, video_file.stem)

## -----------------------------------------------------------
## 함수기능 : 모듈 시작 지점 함수
## -----------------------------------------------------------
if __name__ == "__main__":
    extract_all_frames()
    playing_files()
    print("유효한 프레임만 추출 완료")
