---
{"dg-publish":true,"permalink":"/naver-boost-camp/week-11/","created":"2025-01-07T18:10:48.501+09:00","updated":"2025-01-07T18:14:52.604+09:00"}
---

<aside> 📜

1. Data-Centric AI의 개요
2. Data-Centric AI의 중요성
3. OCR Task
4. 거대 모델을 활용한 OCR 및 문서 이해
5. OCR Services & Applications
6. OCR 및 문서 데이터셋 소개
7. OCR 성능 평가
8. Annotation 도구 소개
9. 데이터 구축 작업 설계
10. Data-Centric AI를 위한 데이터 후처리 </aside>

## 1. Data-Centric AI의 개요

### 1.1 AI System

Code + Data로 구성되어 있다.

이에 Data-Centric AI는 Code (model or algorithm) 보다 Data를 이용해 성능을 올리려 하는 접근 방법

### 1.2 Data-Centric AI

- 성능 향상을 위해 Data 관점에서 고민
- Model Modification 없이 어떻게 모델의 성능을 향상 시킬 수 있을까
- Data-Centric Evaluation

## 2. Data-Centric AI의 중요성

- 수업이나 학교, 연구에서는 Public 한 데이터셋을 이용해 그 데이터 셋에 맞는 더 좋은 모델을 찾는 것을 목표로 한다.
- 하지만 실제 서비스 개발 시에는 데이터 셋이 준비되어 있지 않고 요구사항만 존재하기 때문에 서비스에 적용되는 AI 개발업무의 상당 부분은 데이터셋을 준비하는 작업이다.

### 2.1 Production Process of AI Model

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/cb2bd6b7-0191-49d6-ac79-3f34771cc593/image.png)

- AI 서비스 개발 시 ⇒ 데이터와 모델의 정확도가 반반 정도
- AI 서비스 개선 시 ⇒ 모델보다도 데이터가 훨씬 중요

## 3. OCR Task

- OCR - 전통적인 Task 중 하나로 흰 화면의 검은 글씨 형태
- STR - 일상에 존재하는 Text를 인식하는 Task

### 3.1 OCR 정의

1. 먼저 글자를 찾는다.
2. 찾은 글자가 무엇인지 판단한다.

### 3.2 OCR 데이터의 특징

- 매우 높은 밀도
- 극단적인 종횡비
- 특이모양
    - 구겨짐, 휘어짐, 세로 쓰기
- 모호한 객체 영역
- 크기 편차

### 3.3 OCR 데이터 영역 표현 방법

- 직사각형
- 직사각형 + 각도
- 사각형
- 다각형

### 3.4 OCR Module

- Text Detector: 이미지 입력에 글자 영역 위치가 출력인 모델
- Text Recognizer: 하나의 글자 영역 이미지 입력에 해당 영역 글자열이 출력인 모델
    - 이미지 전체의 입력이 아니라 Text Detector로 정해진 글자 영역만 입력!
    - Computer Vision과 Natural Language Processing의 교집합 영역
        - input: image / output: Text
- Serializer: OCR 결과값을 자연어 처리하기 편하게 일렬로 정렬하는 모듈
    - ex) 사람이 읽는 순서대로 정렬한다.
- Text Parser: 정의된 Key들에 대한 Value 추출
    - BIO 태깅을 활용한 개체명 인식: 문장에서 기 정의된 개체에 대한 값 추출

## 4. 거대 모델을 활용한 OCR 및 문서 이해

### 4.1 TrOCR

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/b2081009-2dd0-4fe5-a866-c923b21cc202/image.png)

- Efficient and Accurate
    - 거대한 데이터에 대해 사전 학습된 image/text 트랜스포머를 그대로 사용(plugandplay)
    - 이미지/OCRtask에 대한 사전지식을 활용하는 특수한 전처리를 하지 않는다(easytoimplement)
- 인코더와 디코더를 동시에 사전 학습
- 데이터 증강은 실제 있을 법한 예외 상황을 모델링할 때 가장 효과적!
    - TextRecognitionTask
        - Randomrotation
        - Gaussianblurring
        - Imagedilation
        - Imageerosion
        - Downscaling
        - Underlining
    - SceneTextRecognition
        - RandAugment(후보증강화전략들중가장효과적인증강화sequence를 추정하는알고리즘)

### 4.2 DTrOCR

- 이미지 인코더 없이, 이미지 임베딩을 더 큰 텍스트 디코더 (GPT-2)에 넣은 모델
- 파라미터 효율성이 좋음
- 더 큰 사전학습 데이터 셋을 사용

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/7f6c313c-1c8f-4530-b72c-9d23a4412efc/image.png)

- Arichtecture
    1. Feature extractor
    2. Multi-model feature enchancement
    3. Output fusion

### 4.3 MATRN

- Motivation
    1. 가려지거나 해상도가 낮은 이미지에 대해서 이미지로만 판단이 힘듦
    2. 어떻게 두 modality를 정렬시키고 각각의 attention을 뮤지하지?

## 5. OCR 및 문서 데이터셋 소개

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/cabaf381-1b62-47b6-a327-038ce06fad00/image.png)

- Public Dataset
    - Kaggle, RRC 등의 대회의 Dataset
    - OCR 데이터 셋 논문이나 여러 OCR 학회의 Dataset
    - Google Datasearch, [Zenodo.org](http://Zenodo.org), Datatang 등의 전문 데이터 셋 사이트 이용

### 5.1 OCR EDA

- imagewidth,height분포
- 이미지당단어개수분포
- 전체태그별분포
    - Image tag
    - Word tag
    - Orientation
    - Language tag
- 전체BBOX크기분포
    - 넓이 기준
- Horizontal한단어의aspectratio(가로/세로)

### 5.2 Tips

- detector학습을 위한 영어 및 한국어 public dataset이 많이 있으니 이를 잘 활용 하는 것이 좋다.
- 학습 목적에 맞는 데이터를 잘 선택하여 학습시키면 좋다.
- 다양한 형태의 데이터 셋을 통합하기 위해 UFO처럼 하나의 통일된 포맷을 만드는 것이 실험에 편리하다

## 6. OCR 성능 평가

⇒ 새로운 데이터가 들어왔을 때 얼마나 잘 동작하는가?

### 6.1 성능 평가 시 데이터 분리 방법

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/5001f059-8cc6-4727-8d7b-1c801930e3e3/image.png)

### 6.2 글자 검출 모델 평가 방법

###### 여기는 다시 공부해야할 듯

## 7. Annotation 도구 소개

### 7.1 좋은 데이터

⇒ 골고루 모여있고 일정하게 라벨링된 데이터

- 가이드 숙지 능력, 일관된 작업을 보장하기 위한 프로세스 정립

### 7.2 Annotation 도구

- LabelMe
    - 장점: 설치 용이, Python으로 되어 있어 추가적인 기능 추가 가능
    - 단점: 공동 작업 불가능, object, image에 대한 속성 부여 X
- CVAT
    - 장점: 다양한 Annotation을 지원, Automatic annoataion 기능으로, 빠른 annotaion이 가능, 온라인에서 가능, multi-user 기반 annotation 가능
    - 단점: model inference가 굉장히 느림, object, image에 대한 속성 부여 X
- Hasty Labeling Tool
- 장점
    - 다양한 annotation을 지원한다.
    - semi-automated annotation 기능을 지원한다.
    - cloud storage를 활용할 수 있다. (유료)
    - multi-user 기반 annotation이 가능하며, Assignee, Reviewer 기능이 제공된다.
- 단점
    - 서비스 자체가 free credit을 다 소진한 이후에는 과금을 해야한다.
    - annotator가 수동으로 이미지마다 review state로 변경해 주어야 한다.
    - Hasty 플랫폼에 강하게 연결되어 있어, annotation 도구에 대한 커스터마이징이 불가능하다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/2d1e5278-cab4-47bf-9410-d3eb8d1500b5/image.png)

## 8. 데이터 구축 작업 설계

### 8.1 가이드 라인

⇒ 좋은 데이터를 확보하기 위한 과정을 정리해 놓은 문서

가이드라인은 목적에 맞게 **일관되어야** 한다.

- 필요한 3가지 요소
    - 특이 케이스: 가능한한 특이 케이스가 다 고려되어야한다.
    - 단순함: 작업자가 숙지하기 쉬워야함
    - 명확함: 동일한 가이드 라인에 대해서 같은 해석이 가능해야함

> _**Annotationguide는 절대 한 번에 완성되지 않고, 완벽한 가이드는 존재하지 않는다!**_

- 가이드 라인에는 일관성이 매우매우매우매우 중요!

### 8.2 Summary

1. 충분한 pilot tagging을 바탕으로 가이드 제작
2. 가이드 라인 수정 시 versioning 필요,기존 내용과 충돌 없도록 최소한의 변경만
3. 최대한 명확하고 객관적인 표현을 사용
4. 일관성 있는 데이터가 가장 잘 만들어진 데이터
5. 우선순위를 알고, 필요하다면 포기하는 것도 중요

## 9. Data-Centric AI를 위한 데이터 후처리

### 9.1 Image Data Augmentation

⇒ Image Data Augmentation = Geometric Transformation + StyleTransformation + …

- Geometric Transformation이 강의에서는 특히 중요하다함

잘못된 Geometric Transformation은 도움이 안되는 이미지를 생성할 수 있다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/07b53959-f80d-4f80-9f06-4aa43c98d635/image.png)

### 9.2 올바른 Geometric Transformation을 위한 규칙

1. Positive ratio 보장: 최소 1개의 개체를 포함해야 한다
2. 개체 잘림 방지: 잘리는 개체가 없어야 한다.

### 9.3 합성 데이터 제작

⇒ Synthetic Data: Real Data에 대한 부담을 덜어준다

- 비용이 훨씬 적게 든다.
    
- 개인정보나 라이센스에 관한 제약으로부터 자유롭다.
    
- 더 세밀한 수준의 annotation도 쉽게 얻을 수 있다. (character-level, pixel-level)
    
- TextRecognitionDataGenerator
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/ae601438-ae4b-4e17-8c48-450127a00a5f/image.png)
    
- Synth Text: Depth Estimation을 통해 적절한 위치에 표면 모양에 맞춰서 글자를 합성
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/f8be74a0-f91e-4d4a-b0b0-7aa688b065e8/image.png)
    
- Synth Text 3D: 3D 가상세계를 이용한 텍스트 이미지 합성
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/5cbadc16-53fc-45b6-ba51-3819170e5d60/image.png)
    
- Unreal Text: 3D Virtual Engine을 이용 (개선된 View finding)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/56bdec49-4593-48bb-8f34-cf2ebfdc140e/image.png)
    

### 9.4 합성 데이터 사용 방법

Target dataset만으로 학습할 때:

1. Image Netpretrained model로 부터 backbone을 불러온다
2. target dataset에 대해 fine-tuning을 진행한다 합성 데이터가 주어졌을 때:
3. 합성 데이터로 한번 더 pretraining을 해준다
4. 이후 target dataset에 대해 fine-tuning을 진행.

### 9.5 Data Cleansing

⇒ 이상한 데이터는 모델에 악영향을 준다

- 이미지 자체가 문제가 있는 경우
- 라벨이 잘 못 부여된 경우