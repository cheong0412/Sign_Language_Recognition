import os
import pandas as pd

# 🔹 기존 라벨 엑셀 파일 경로
label_excel_path = r'G:\Team5\DATA\Final_Data\Word\Final_Video_Labels_Index.xlsx'
df = pd.read_excel(label_excel_path)

# 🔹 중복 제거 (File_name 기준)
df_unique = df.drop_duplicates(subset='File_name')

# 🔹 라벨 딕셔너리 (기준: 원본 폴더 이름)
folder_to_label = df_unique.set_index('File_name')[['Korean', 'English', 'Label_Index']].to_dict('index')

# 🔹 증강 이미지가 저장된 루트 폴더
augmented_root = r'G:\Team5\DATA\Final_Data\Word\New_Final_Crop_Rotate_Resize'

# 🔹 최종 저장할 라벨 리스트
augmented_data = []

# 🔹 증강 폴더들 순회
for foldername in sorted(os.listdir(augmented_root)):
    folder_path = os.path.join(augmented_root, foldername)
    if not os.path.isdir(folder_path):
        continue

    # 🔸 원본 폴더 이름 찾기 (001, IMG_0132 등)
    matched_base = None
    for base_name in folder_to_label.keys():
        if base_name in foldername:
            matched_base = base_name
            break

    if matched_base is None:
        print(f"⚠️ 라벨 매칭 실패: {foldername}")
        continue

    label_info = folder_to_label[matched_base]

    # ✅ 폴더 이름 하나에 대해 라벨 1개만 추가
    augmented_data.append({
        'File_name': foldername,  # 폴더 이름 기준
        'Korean': label_info['Korean'],
        'English': label_info['English'],
        'Label_Index': label_info['Label_Index']
    })

# 🔹 데이터프레임으로 저장
new_df = pd.DataFrame(augmented_data)
save_path = r'G:\Team5\DATA\Final_Data\Word\New_Final_Video_Label_Index.xlsx'
new_df.to_excel(save_path, index=False)
print(f"✅ 폴더 기준 라벨 엑셀 저장 완료: {save_path} (총 {len(new_df)}개)")