---
{"dg-publish":true,"permalink":"/naver-boost-camp/week-7/","created":"2025-01-07T18:10:24.092+09:00","updated":"2025-01-07T18:13:48.505+09:00"}
---

> **1.** **CV Basic Competition
> 2. Image
> 3. Image Processing
> 4. Image Clasisification
> 5. Model
> 6. Representation
> 7. Trainging Process
> 8. Efficient Training
> 9. Evaluation
> 10. Experiments

# **1.** **CV Basic Competition**

- Competition은 문제의 해결이 왜 필요한지 그 배경을 설명 ⇒ 이는 문제해결에 힌트가 될 수 있습니다.
    
- Competition을 하면서 실용적인 경험치가 쌓이고 팀원과 함께하며 여러 방향의 문제 접근 방식을 알 수 있게 됩니다.
    
- **Competition에 들어가면서**
    
    - 문제 정의가 가장 중요 ⇒ 질문을 던지는 것으로 시작
        
        Ex 1) 어떻게 만들어진 이미지일까? 문서일까? 그림일까? 포맷은 일정한가?
        
        Ex 2) 혹시 특별한 목적으로 사용하는 것인가? 또는 특별한 환경에서 사용하는 것인가?
        
        Ex 3) 이미지의 스타일을 어떠한지 인쇄체느낌인지 손글씨 느낌인지
        
        Ex 4) 평가 Metric은 어떤 것을 선택할지
        
        ⇒ 이러한 정보를 알게 되면 앞으로 무엇을 해야할 지 구체적으로 보이기 시작합니다
        

> Competition 점수는 성적표가 아니다. 단지 표시일뿐이다. 하지만 이 대회에서 무엇을 얻었는지 무엇이 부족했는지, 무엇을 추가로 공부해야하는지를 생각할 수 있어야한다,

# **2. Image**

## EDA

⇒ 데이터를 분석 및 탐구, 가장 중요한 것은 본인이 알고 싶은 것을 탐구하는 것

- 데이터를 보면서 궁금했던 내용
- 데이터의 생김새, 특징
- 데이터의 생성 배경에 관련된 자신의 추측을 검증
- 코드를글과함께작성하는능력.
- 말하고자하는바를코드로옮기고설명

⇒ 이미지 기본 정보를 알 수 있고 중복된 이미지 제거, 이미지 샘플 확인 등이 가능하다.

# **3. Image Processing**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/5b0ecd41-a55e-417f-8496-6bac5acf41b1/image.png)

## Geometric Transform

- Translation: 물체의 위치 이동
- Rotation: 각도 $θ$ 만큼 이미지 회전
- Scaling: 이미지 크기 조정
- Perspective Transformation: 이미지의 원근 변환 적용

## Data Augmentation

⇒ 모델의 과적합 감소, 학습 데이터의 다양성 증가, 모델의 견고성 증가

- Common Augmentations: Filps, Rotation, Crops, Clor Jittering
- Advanced Augmentation Techniques: AutoAugment, RandAugment

## Normalization

- 데이터셋 평균을 빼고 데이터셋 표준편차로 나눔
    
    - PyTorch Transform Compose: 이미지를 Tensor로 변환 → 스케일링
    - Albumentations Compose: Normalize( [0,1] Scaling은 파라미터로 지정) → Tensor로 변환
- Batch Normalization
    
    ⇒ mini-batch단위로입력데이터를정규화
    

# **4. Image Clasisification**

- Coarse-grained Classification
    
    ⇒ 상대적으로 큰 범주로 객체를 분류
    
    - 클래스 간의 연관성이 비교적 적은 편인 문제
- Fine-grained Classification
    
    ⇒ 동일한 상위 범주 내에서 세부적인 하위 범주로 분류
    
    - 같은종(Species) 내에서 분류 하기 때문에 다른 클래스라도 비슷한 점이 존재
- N-Shot Classification
    
    - Few-shot Classification
        
        클래스당 단 몇개의 데이터 만으로 학습하고 Unseen Data 예측
        
    - One-shot Classification
        
        클래스당 단 하나의 예시만으로 학습하고 Unseen Data 예측
        
    - Zero-shot Classification
        
        학습 없이 Unseen Data를 예측 ⇒ Foundation model에 의존
        

# **5. Model**

- Inductive Bias (귀납적편향)
    
    모델이 학습 과정에서 특정 유형의 패턴을 잘 학습하도록 하는 사전가정(지식)
    
    - CNN
        1. Locality & Translation Invariance: 이미지의 어디서든 나타나는 지역 패턴에 집중
        2. Hierarchical Feature Learning: 단순한 특징에서 복잡한 특징으로 학습
        3. Weight Sharing: 동일한 필터를 이미지 전체에 사용
    - Transformers
        1. Long-Range Dependencies: 먼 거리의 관계를 포착
        2. Flexible Input Handling: 다양한 입력 크기를 처리
        3. Self-Attention: 입력 부분의 중요도를 동적으로 가중

⇒ Inductive Bias는 모델이 어떤 관점에서 데이터를 보려고 하는가 를설명하는것. 이를 이해하면 모델 선택의 자유도가 높아진다.

- 처음부터 이렇다할 모델을 고르기는 어렵다
    
    ⇒ 논문에서 말하는 BackGround, 논문에서 사용한 벤치마크 데이터 셋으로 특징 파악, 경험치 등의 방법으로 모델 선택이 가능하다.
    
- CNN과 Transformer의 장단점
    

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/7b411a9f-45a2-4292-9e06-7dca3101917a/image.png)

- Hybr id Model

⇒ CNN과 Transformer를 합친 모델 → 개별 아키텍쳐의 한계를 극복, 더 나은 Genrelization, Scalability을 달성

⇒ CoAtNet, ConViT, CvT, LocalViT

# **6. Representation**

## Representation Learning

⇒ 데이터로부터 유의미한 Representation을 자동으로 학습할 수 있도록

- Feature Extraction: Raw 데이터에서 중요한 특징을 식별하고 인코딩하는 것
    
- Generalization: 보편적인 패턴을 포착하여 보지 못한 데이터에서도 잘 수행할 수 있는 능력
    
- Data Efficiency: 효과적인 특징 학습을 통해 대규모 주석 데이터셋의 필요성을 줄이는 것
    
- Transfer Learning: 한 task도메인에서 사전 학습된 모델을 다른 task도메인에 적용
    
- Self-Supervised Learning: Label이 없는 데이터에서 Representation 학습
    
    - Contrastive Learning: Contrastive loss는 positives를 모으고 negatives를 밀어냄
- Multimodal Learning: 여러가지 modality에서 정보 결합
    
- Foundation Models: 광범위하고 방대한 데이터셋에서 사전학습, 개체에 대한 다양한Representation의 의미를 녹여낼 수 있다
    

# **7. Trainging Process**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/91980b7c-a118-4cf1-aefa-2208150643cb/image.png)

## 학습 Task 선언

⇒ 무엇을 학습할지?

- 내가 학습하고자 하는 Task에 맞게 데이터를 구성하고 Dataset class와 Dataloader를 선언
- 적합한 Model 선별

## 학습 방법 선정

⇒ 어떻게 학습할지?

- 손실함수 선언
    
    여러 Loss 함수의 파라미터 값을 설정하여 더 성능을 높일 수도 있읃
    
    Ex ) CrossEntropyLoss에서 Label_smoothing에 0.1값을 부여하니 성능이 올라가는 경우를 확인함
    
- Optimizer 선언
    
- Learning Rate Scheduler 선언
    

## 학습 설계

⇒ 어떻게, 얼마나 학습할지?

- Training loop 설계
- 몇번의 Loop를 학습할지에 대한 Step과 Iteration을 설정

# **8. Efficient Training**

## Data Process 최적화

- Gradient Accumulation: 여러 배치에서 계산된 gradient를 누적하여 업데이트하는 기술
    
    ⇒ 이는 작은 배치 사이즈라도 큰 배치 사이즈를 사용하는 것과 유사한 효과를 냄
    
- Mixed Precision Training: 모델 성능과 관련된 연산에서 FP32를 사용하여 정확도를 높임, 연산이 필요한 시점에서는 FP16을 사용해 속도를 높임
    

## Semi-Supervised Learning

- Pseudo Labeling: 학습한 모델을 이용해 라벨이 없는 데이터를 학습하고 이를 다시 학습에 이용
- Image Generation: 이미지를 생성하여 학습에 활용할 추가 데이터를 확보

# **9. Evaluation**

## Metric

⇒ 모델이 올바르게 학습이 되고 있는지 그 수치를 정량적으로 나타낸 것

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/49118dcb-a787-4793-b764-60000ca57869/image.png)

- Metric 선택
    
    ⇒ 무엇을 더 중요하게 생각하느냐에 따라 비용이 다를 수 있기 때문에 잘 선택해야함
    
    Ex 1) 쓰레기장에 도둑이 들어도 손실이 적다 → 예측이 틀려 실제로 도둑이 아닌데 잡으면 민원 때문에 더문 제다.(False Positive가 낮아야한다.) → Precision이 높은 모델이어야한다
    
    Ex 2) 금은방이라 한 번이라도 도둑을 못 잡으면 손실이 크다. (False Negative가 낮아야한다.) → 예측이 틀려 실제로 도둑이 아니었다고 해도 그건 아무렇지도 않다. → Recall이 높은 모델이어야한다.
    

## Ensemble

⇒ 동일한 태스크의 여러 가지 모델의 결과를 혼합하여 성능을 높인다

- Weighted Ensemble: 성능이 더 좋은 모델에 더 많은 가중치를 부여
    
- Voting
    
    - Soft Voting: 예측확률을 비교해 선택
    - Hard Voting: 예측한 Class 중 가장 많이 나온 것을 정답으로 선정
- Cross-Validation: 훈련데이터가 부족할 때 Validation Set을 바꿔가면 훈련
    
    - K-Fold: Split에 따라 Fold 갯수와 Validation 비율이 결정됨
    
    ⇒클래스의 분포를 고려, Task의 속성에 따라 다양한 전략이 존재
    
- Test-Time Augmentation (TTA): 테스트 데이터의 이미지를 N번 Augmentation 후 그 N개의 결과를 앙상블
    

# **10. Experiments**

- 실험관리 중요성: 팀 내 일관된 실험기록의 중요성
- 실험의 재현성 보장

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/fdd6c116-a7f1-4d0c-a05b-caeef41e3f6c/image.png)

⇒ 언제 무엇을 어떻게 실험했는 지 알 수 있어야 한다.

## 실험개발도구

- Jupyter
    - 장) 인터랙티브한 코드 실행 → EDA할 때 편함
    - 단) 대규모 코드 관리에 불편
- Python Script
    - 장) 모듈화 및 재사용성
    - 단) 인터랙티브한 개발 및 테스트에 비해 불편

## 실험트래킹

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/2ed1634e-5795-4a2f-a3b9-e91c3a5c3527/image.png)

# 11. Conclusion

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/afd0c423-fbef-45e4-b1e2-00ce88f26b8d/image.png)