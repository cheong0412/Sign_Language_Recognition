<img width="1023" height="535" alt="데이터1" src="https://github.com/user-attachments/assets/d0d38640-c577-4686-8d1d-dfb9b23331ae" /># 수어 인식 기반 농인 지원 시스템

## 목차
  - [프로젝트 기본 정보](#프로젝트-기본-정보)
  - [프로젝트 개요](#프로젝트-개요)
  - [프로젝트 설명](#프로젝트-설명)
  - [분석 결과](#분석-결과)
  - [기대 효과](#기대-효과)

## 프로젝트 기본 정보
- 프로젝트 이름: 수어 인식 기반 농인 지원 시스템
- 프로젝트 기간: 2025.05.07 ~ 2025.06.23
- 프로젝트 참여인원: 4명
- 사용 언어: Python

## 프로젝트 개요
- 농인은 병원 이용 시 사전 접수부터 진료까지 의사소통에 큰 불편을 겪고 있음.
- 키오스크·진료환경에서 수어 사용자를 위한 병원 접수 시스템을 개발하고자 함분석을 진행함.
<img width="686" height="372" alt="1 주제선정배경" src="https://github.com/user-attachments/assets/74099c99-4256-4724-afbf-4f9335f6c54f" />
<img width="700" height="360" alt="2 주제선정배경" src="https://github.com/user-attachments/assets/3188b166-cb3c-49bf-8611-e96697b8b87f" />
<img width="742" height="363" alt="3 주제선정배경" src="https://github.com/user-attachments/assets/0a525e10-0400-4359-b4ce-29319c27e964" />
<img width="716" height="358" alt="4 주제선정배경" src="https://github.com/user-attachments/assets/216a5471-ad9f-495b-8490-b2035b6f505b" />
<img width="762" height="367" alt="5 주제선정배경" src="https://github.com/user-attachments/assets/705b8774-aa87-4580-a3a8-29da482af2e4" />


## 프로젝트 설명
농인들의 진료환경에서의 소외를 개선하기 위해 수어 인식 병원 진료 접수 시스템을 개발하고자 함.
<img width="717" height="271" alt="6  프로젝트 목표" src="https://github.com/user-attachments/assets/24edb61c-4471-4a77-8cdc-5af30a5b49af" />
<img width="680" height="238" alt="7  프로젝트 목표" src="https://github.com/user-attachments/assets/7dfe56b3-ba49-4133-98c5-031473f39819" />
<img width="665" height="340" alt="8  개발환경" src="https://github.com/user-attachments/assets/75a10a12-cc61-42bd-be9e-874bb46e9808" />
<img width="753" height="422" alt="9 시스템 흐름" src="https://github.com/user-attachments/assets/27a1050c-af4f-49bc-868e-7860b6875053" />

## 분석 결과
### **지문자 수어 모델**
- 수어 모음과 숫자 일부가 동일한 모양이기 때문에 별도의 모델 개발
- YOLOv8s 모델의 성능이 자모음과 숫자 모두 고려하였을 때, 가장 성능이 높다고 판단
- 웹에 구현을 고려하였을 때 경량화된 YOLOv8s 모델이 지문자 수어 모델에 가장 적합하다고 판단하여 최종 모델로 채택
<img width="1023" height="535" alt="데이터1" src="https://github.com/user-attachments/assets/31a6b9c2-6d44-4fa1-9ea4-ed96e04cf8af" />
<img width="680" height="332" alt="10 모델" src="https://github.com/user-attachments/assets/b46a59c4-d958-4dbd-a835-b09c8599db09" />
<img width="757" height="352" alt="11 모델" src="https://github.com/user-attachments/assets/b18d50b6-c84b-4d9b-8924-8d4624a8420b" />



### 증상 수어 모델 개발
- MediaPipe + RandomForest 방식은 MediaPipe + CNN + LSTM 방식보다 더 빠르고 간단하며, 단순한 본 데이터의 특성에 잘 부합함.
- 또한, MediaPipe + RandomForest 방식은 MediaPipe + CNN + LSTM 방식보다 속도가 빠르고, 모델 크기도 작은 장점이 있음.
- 따라서 최종 모델로 MediaPipe + RandomForest 방식을 채택함.
<img width="592" height="335" alt="15 증상모델" src="https://github.com/user-attachments/assets/17b69191-db5c-4f08-aca5-b0aa93d097a6" />
<img width="597" height="326" alt="12 증상모델" src="https://github.com/user-attachments/assets/6a5262a8-b6ed-4341-9c47-64312901208b" />
<img width="597" height="325" alt="13 증상수어모델" src="https://github.com/user-attachments/assets/eddf041a-e7c3-4ef1-9ce3-95e933cf3f29" />
<img width="588" height="320" alt="14 증상모델" src="https://github.com/user-attachments/assets/7d115587-e23c-415b-be39-7a0cb5ba1253" />
<img width="600" height="335" alt="16 증상모델" src="https://github.com/user-attachments/assets/2b37ccfa-e84d-4e69-a7df-3da6698de69a" />
<img width="601" height="287" alt="17 증상모델" src="https://github.com/user-attachments/assets/75480020-f83d-469c-8cf1-1995abea984d" />
<img width="568" height="282" alt="18 증상모델" src="https://github.com/user-attachments/assets/b5307b5e-bf1f-4f48-aa6a-60222f90841d" />


## 기대 효과
- 의료기간 내 농인의 접근성과 자율성 증대
- 통역사 부족 문제의 대안 마련
- 사회적 약자 중심의 기술 개발 사례로 디지털 포용 사회의 구현 기여

## Lesson & Learned
- 영상 데이터를 어떻게 처리하는지에 대해 이해하고, 활용 능력을 키움.
- 처음 접하는 MediaPipe라는 프레임워크를 학습하고, 직접 적용해보면서 이해도가 높아졌음.
- 딥러닝 모델이 어떻게 만들어지는지, 그리고 튜닝 종류에 학습하고, 실습하며 알게 됨
- 청각장애인과 농인의 차이, 수어에는 조사가 없다는 점, 지문자와 단어 표현이 다르다는 점 등 수어 관련 도메인 지식을 새롭게 알게 됨
- 수어 통역사 부족 문제가 생각보다 심각하다는 걸 알게 됐고, 이를 기술로 해결해볼 수 있다는 점이 흥미로웠음.
- 수어 번역 기술이 요즘 주목받는 분야이며, 앞으로 더 공부해서 이 프로젝트를 확장시켜보고 싶다는 욕심이 생김.
