## ------------------------------------------------------------
## 파일이름 : config.py
## 모듈기능 : 공통으로 사용되는 경로 정의된 모듈
## ------------------------------------------------------------
import os

# 기본 프로젝트 경로
BASE_DIR            = "D:/CO_PROJECT/SIGN2TEXT_MDEIAPIPE"
MEDI_BASE           = f"{BASE_DIR}/media_SLR"
VIDEO_DIR           = f"{BASE_DIR}/videos"

ANNOT_DIR           = f"{MEDI_BASE}/annotation"
ANNOT_FILE          = f"{ANNOT_DIR}/SL_Annotation.xlsx"

# MDEIAPIPE 관련 경로
MEDI_FRAMES         = f"{MEDI_BASE}/frames"
MEDI_IMAGES         = f"{MEDI_BASE}/images"
MEDI_KEYPOINTS      = f"{MEDI_BASE}/keypoints"
MEDI_LABELS_FILE    = f"{MEDI_BASE}/labels.csv"
MEDI_SEQUENCE       = f"{MEDI_BASE}/sequences"

## 디버깅 설정
DEBUG              = True

## 공통 설정
SEQUENCE_LENGTH = 16