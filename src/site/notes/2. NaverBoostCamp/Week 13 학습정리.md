---
{"dg-publish":true,"permalink":"/2-naver-boost-camp/week-13/","created":"2025-02-26T15:44:18.937+09:00","updated":"2025-02-24T15:46:20.685+09:00"}
---

# Semantic Segmentation 강의 상세 설명

이 문서는 NAVER Connect Foundation에서 진행한 Semantic Segmentation 관련 강의 PDF 파일들을 바탕으로 각 강의별 주요 내용과 세부 개념, 기법들을 자세하게 설명합니다.

---

## Lecture 1: Introduction (1강)

**주요 내용 및 개요:**

- **커리큘럼 개요 및 강의 진행 방식:**
    - 전체 강의는 총 10개의 세션으로 구성되며, Introduction, Competition Overview, 세그멘테이션의 기초, FCN의 한계 극복, U-Net 계열, 대회 기법, 최신 연구 동향 등 단계별로 진행됩니다.
    - 강의 방식은 이론과 실습을 병행하며, 심화 프로젝트를 통해 실무 적용 방법을 모색합니다.
- **세그멘테이션 개요 및 목적:**
    - Semantic Segmentation의 정의와 역할, 즉 각 픽셀 단위로 객체를 분할하는 작업의 중요성을 설명합니다.
    - 다른 비전 태스크(예: object detection, instance segmentation)와의 차이를 비교하고, 응용 분야(의료영상, 자율주행 등)를 소개합니다.

---

## Lecture 2: Competition Overview (2강)

**주요 내용 및 개요:**

- **데이터셋 개요:**
    - Hand Bone Image Dataset을 사용하며, X-ray 이미지를 기반으로 손 뼈 영역을 분할하는 문제를 다룹니다.
    - 데이터 구성: Train 800개, Test 288개로 구성되며, 이미지 파일은 PNG 형식이고, 어노테이션은 JSON 파일에 Polygon 좌표로 제공됩니다.
- **EDA (Exploratory Data Analysis):**
    - 데이터의 클래스 분포, 각 클래스가 차지하는 픽셀 비율, 그리고 메타 데이터(키, 몸무게, 성별, 나이)를 통해 데이터의 특성을 분석합니다.
    - 특히, 한 픽셀에 여러 클래스가 존재할 수 있는 Multi-label 문제 및 클래스 간 중첩 현상을 확인합니다.
- **평가지표 및 Baseline Code:**
    - 모델 평가를 위한 주요 지표로 Dice Score를 사용하며, 데이터 로딩, 모델 구성, Loss/Optimizer 설정, 학습 및 검증 코드를 설명합니다.

---

## Lecture 3: Semantic Segmentation의 기초와 이해 (3강)

#### 1. FCN을 이용한 세그멘테이션 개요

- **목적 및 배경:**  
    딥러닝을 이용한 이미지 세그멘테이션의 대표적 기법인 FCN에 대해 설명합니다. 원래 이미지 분류에 사용되던 네트워크(예: VGG)를 기반으로, 공간 정보를 유지하며 각 픽셀 단위의 예측을 수행할 수 있도록 구조를 변경한 것이 핵심입니다.
    
- **핵심 아이디어:**
    - **Fully Connected Layer → 1x1 Convolution:**  
        기존의 고정된 크기를 요구하는 FC 계층 대신 1x1 컨볼루션을 사용하여 입력 이미지의 크기에 상관없이 각 픽셀별로 예측할 수 있게 합니다.
    - **Transposed Convolution (Deconvolution):**  
        Pooling 등으로 축소된 feature map을 원래의 해상도로 복원하기 위해 학습 가능한 파라미터를 가진 업샘플링 기법인 Transposed Convolution을 사용합니다.

---

#### 2. FCN의 구성 및 동작 원리

- **VGG 네트워크 백본 활용:**  
    VGG와 같은 기존 분류 네트워크의 컨볼루션 블록을 사용해 특징을 추출하고, FC 계층 대신 1x1 컨볼루션을 적용함으로써 공간 정보를 보존합니다.
    
- **컨볼루션 vs. FC 계층:**
    - **컨볼루션 계층:** 이미지 크기와 무관하게 동작하며, 각 픽셀의 위치정보를 유지합니다.
    - **FC 계층:** 입력 이미지의 공간 구조를 무시하기 때문에, 세그멘테이션에 부적합합니다.
    
- **Transposed Convolution의 상세 설명:**  
    여러 슬라이드를 통해 행렬 형태의 예시와 함께 Transposed Convolution의 계산 과정을 단계별로 설명합니다.
    - 업샘플링 과정에서 학습 가능한 파라미터로 인해 원본 컨볼루션과는 다른 결과를 내며, 이를 통해 손실된 공간 정보를 복원합니다.
    - Stride, kernel size, padding 등의 하이퍼파라미터가 출력 크기에 어떻게 영향을 미치는지 구체적으로 다룹니다.

---

#### 3. FCN 성능 향상을 위한 전략

- **업샘플링 단계 개선:**  
    단순히 FCN-32s와 같이 한 번에 업샘플링하는 대신, 중간 계층의 정보를 활용해 FCN-16s, FCN-8s 등 여러 단계의 업샘플링을 적용함으로써 더 정밀한 예측을 도모합니다.
    
- **Skip Connection 활용:**  
    초기 단계의 더 높은 해상도를 가진 feature map과 후반의 낮은 해상도의 예측 결과를 결합하여, 업샘플링 과정에서 잃어버린 세부 정보를 보완합니다.

---

## Lecture 4: FCN의 한계를 극복한 모델들 1 (4강)

#### 1. FCN의 한계점

- **문제점:**
    - 큰 객체의 경우 지역적 정보만으로 예측하여 세부 경계나 전체 객체 모양을 제대로 파악하지 못함
    - 작은 객체는 무시되거나 잘못 라벨링되는 문제
    - 디코더(업샘플링) 과정에서 경계 정보 손실 및 불일치가 발생
    - pooling 과정에서 중요한 세부 정보가 소실됨  

---

#### 2. Decoder 개선 모델

- **DeconvNet:**
    - **구조:** VGG16 기반 인코더에 대응하는 대칭 구조의 디코더
    - **기법:** Unpooling(최대값 인덱스를 이용해 원래 위치 복원)과 Transposed Convolution을 반복적으로 적용하여 잃은 경계 및 세부 정보를 복원
    - **특징:** Unpooling은 빠르지만 sparse한 activation map을 보완하기 위해 Transposed Convolution이 필수적임  

- **SegNet:**
    - **구조:** 인코더-디코더 네트워크로, VGG16 인코더의 특징을 활용하면서 디코더에서는 1×1 Convolution을 제거하여 파라미터 수를 줄이고 연산 속도를 개선
    - **특징:** 경계 복원과 효율성 측면에서 DeconvNet과 차별화됨  

- **비교:** 두 모델은 모두 디코더 성능을 향상시키지만, 학습 및 추론 시간, 파라미터 효율성 측면에서 차이가 있음

---

#### 3. Skip Connection을 적용한 모델

- **FC-DenseNet:**
    - DenseNet의 특성을 도입해, 이전 레이어의 출력을 건너뛰어 연결함으로써 더 풍부한 공간 정보를 반영하여 세밀한 예측이 가능
- **U-Net:**
    - 주로 의료 영상 등에서 사용되며, 인코더와 디코더 사이의 skip connection을 통해 저해상도 단계에서 잃은 세부 정보를 복원  

---

#### 4. Receptive Field 확장을 통한 모델 개선

- **DeepLab v1:**
    - **문제 해결:** 단순한 downsampling으로 인한 low-resolution 문제를 해결하고, 더 넓은 receptive field를 확보하기 위해 atrous(또는 dilated) convolution을 도입
    - **후처리:** Bilinear interpolation 기반 업샘플링 후 Dense CRF를 적용하여 세밀한 경계와 클래스별 일관성을 개선

- **DilatedNet:**
    - DeepLab과 유사하게, dilation rate를 조절하여 적은 파라미터로 넓은 receptive field를 확보함으로써 고해상도 출력을 유지  


---

#### 부가 내용

- **업샘플링 방법:**  
    Bilinear interpolation을 사용하지만, 픽셀 단위의 정교한 세분화에는 한계가 있어 후처리로 Dense CRF를 적용

---

## Lecture 5: FCN의 한계를 극복한 모델들 2 (5강)

#### 1. Receptive Field 확장을 통한 모델들

- **DeepLab v2:**
    
    - **구조:** Dilated(atrus) convolution을 활용해 다운샘플링을 최소화하면서도 넓은 receptive field를 확보하는 방식
    - **특징:** ResNet-101을 백본으로 사용하여 더 깊은 네트워크 구조와 dilated convolution을 적용, 고해상도 feature map을 유지하면서 전역 문맥을 반영

- **PSPNet (Pyramid Scene Parsing Network):**
    
    - **구조:** Pyramid Pooling Module을 도입하여 다양한 크기의 sub-region에서 평균 풀링을 수행, 이를 통해 전역 문맥 정보를 캡처
    - **특징:** 주변 정보(context)를 활용해 혼동되기 쉬운 카테고리나 작은 객체의 예측 성능을 향상

- **DeepLab v3:**
    
    - **구조:** DeepLab v2의 기본 아이디어를 발전시켜 atrous spatial pyramid pooling(ASPP) 모듈을 도입, 다양한 dilation rate로 여러 스케일의 문맥 정보를 통합
    - **특징:** 다중 스케일의 정보를 효과적으로 융합하여 세밀한 객체 경계와 전역 정보를 동시에 반영

- **DeepLab v3+:**
    
    - **구조:** DeepLab v3의 ASPP 모듈을 기반으로 encoder-decoder 구조를 추가하여, 인코더에서 축소된 공간 정보를 디코더에서 점진적으로 복원
    - **특징:** 수정된 Xception 백본과 depthwise separable convolution을 사용하여 효율성을 높이고, low-level feature와 ASPP 출력을 결합해 세밀한 경계 복원을 도모


---

#### 2. 모델 성능 및 비교

- 각 모델의 성능 비교 결과는,
    - FCN-8s (약 62.2%),
    - DeepLab v1 (약 71.6%),
    - DeepLab v2 (약 79.7%),
    - PSPNet (약 85.4%),
    - DeepLab v3 (약 85.7%),
    - DeepLab v3+ (약 89.0%)  
        등으로 성능이 점진적으로 향상됨을 보여줍니다.
- 이와 같은 성능 향상은 각 모델이 receptive field 확장, 멀티 스케일 문맥 정보 통합, 그리고 인코더-디코더 구조를 통해 세밀한 경계 복원에 기여한 결과로 판단됩니다.


---

## Lecture 6: High Performance를 자랑하는 U-Net 계열의 모델들 (6강)

#### 1. U-Net

- **소개 및 필요성:**
    - 의료 영상과 같이 데이터가 부족한 분야에서 효과적인 segmentation을 위해 제안됨
    - 동일 클래스 내 인접 객체(예, 세포) 간 경계 구분이 중요한 문제 상황을 해결하기 위한 설계
- **구조:**
    - **Contracting Path:** 3×3 컨볼루션 + BN + ReLU를 두 번 반복, 2×2 맥스 풀링으로 차원 축소하며 채널 수를 2배로 증가
    - **Expanding Path:** 2×2 Up-Convolution(Transposed Convolution)을 사용해 해상도를 복원하고, 동일 레벨의 인코더 출력을 concat하여 로컬 정보를 보존
- **적용 기술 및 한계:**
    - 데이터 증강(Elastic deformation)과 경계에 가중치를 주어 학습을 보강하는 방법 적용
    - 그러나 기본 구조는 깊이가 4로 고정되어 있고 단순한 skip connection만 사용하여 최적의 성능을 내기 어려움  

---

#### 2. U-Net++

- **개념 및 동기:**
    - U-Net의 단순한 skip connection이 정보 전달에 한계가 있음을 보완하기 위해 제안됨
- **구조적 특징:**
    - 인코더에서 추출한 여러 깊이의 feature map들을 서로 중첩(nested)하여 연결하는 Dense Skip Connection 구조를 도입
    - 서로 다른 레벨의 정보를 효과적으로 결합하고, deep supervision(중간 출력을 통한 학습 보조) 및 앙상블 기법을 활용
- **장점 및 한계:**
    - 보다 풍부하고 세밀한 feature 재구성이 가능하지만, 구조 복잡성과 연산 비용이 증가할 수 있음  

---

#### 3. U-Net 3+

- **목표:**
    - U-Net과 U-Net++의 한계를 극복하고, 더 다양한 레벨의 정보를 통합해 보다 정밀한 segmentation 성능을 달성
- **구조적 개선:**
    - 인코더와 디코더 사이의 연결을 단순히 같은 레벨에 국한하지 않고, 다중 스케일의 feature를 통합하는 방법 채택
    - 보다 유연한 skip connection 설계를 통해 경계 복원과 세밀한 localization을 강화  

---

#### 4. 기타 U-Net 변형

- **다른 변형 모델:**
    - **Residual U-Net:** 잔차 연결(residual connection)을 추가하여 깊은 네트워크 학습의 안정성을 개선
    - **Mobile-UNet, Eff-UNet:** 경량화와 효율성을 목표로 한 모델로, 모바일 환경이나 실시간 응용에 적합하도록 설계됨

---

## Lecture 7: Semantic Segmentation 대회에서 사용하는 방법들 1 (7강)

#### 1. EfficientUnet Baseline

- **모델 불러오기 및 학습:**
    - qubvel의 segmentation_models.pytorch 라이브러리를 활용하여, EfficientNet-B0와 같은 다양한 사전학습된 encoder를 이용해 Unet 기반의 모델을 쉽게 불러올 수 있음
    - Unet, Unet++, Manet, Linknet, FPN, PSPNet, PAN, DeepLabV3/V3+ 등 여러 아키텍처를 지원하며, 특히 EfficientUnet baseline은 입력 (1, 3, 2048, 2048) → 출력 (1, 29, 2048, 2048) 형태로 추론됨  

---

#### 2. Baseline 이후 실험 시 고려사항

- **주의사항:**
    - 디버깅 모드, 시드 고정, 실험 기록(예: Notion, Google Sheets) 등으로 재현 가능하고 체계적인 실험 환경 구축
    - 한 번에 하나의 조건만 변경하여 어떤 요소가 성능에 영향을 주는지 명확하게 파악할 수 있도록 진행

- **Validation 방법:**
    - Hold Out 방식과 함께 K-Fold, Stratified K-Fold, Group K-Fold 등 다양한 분할 전략을 통해 전체 데이터셋에 대해 신뢰성 있는 검증 진행

- **Augmentation:**
    - 데이터 양을 증가시키고 일반화를 강화하기 위한 다양한 기법 사용 (예: RandomCrop, Horizontal/Vertical Flip, Rotation, Transpose 등)
    - 최근 기법인 AutoAugment, Fast AutoAugment, Albumentations 기반 기법과 함께 Cutout, Gridmask, Mixup, Cutmix, SnapMix 등의 방법도 소개됨

- **SOTA Model 및 Scheduler:**
    - 최신 논문이나 Paper With Code의 SOTA 모델들을 참고하여 baseline 이후 추가 실험 진행
    - Constant Learning Rate의 한계를 극복하기 위해 CosineAnnealingLR, ReduceLROnPlateau, Gradual Warmup 등 다양한 스케줄러 적용

- **Hyperparameter Tuning 및 Optimizer/Loss:**
    - 배치 크기를 키우는 효과를 얻기 위해 Gradient Accumulation 기법 사용
    - Adam, AdamW, AdamP, RAdam, Lookahead 등 다양한 최적화 알고리즘과, imbalanced segmentation에 강한 Compound Loss (예: DiceFocal, DiceTopK 등)를 고려함

---

## Lecture 8: Semantic Segmentation 대회에서 사용하는 방법들 2 (8강)

#### 1. Baseline 이후 실험 시 고려사항 II

- **Ensemble 기법:**
    
    - **5-Fold Ensemble:** K-폴드를 활용해 여러 모델을 학습하고, 이를 앙상블하여 예측 성능을 향상
    - **Epoch Ensemble:** 학습 마지막 몇 개의 Epoch의 weight를 평균 내거나, 여러 Epoch의 결과를 앙상블하는 방식
    - **SWA (Stochastic Weight Averaging):** 일정 주기마다 weight 평균을 수행해 넓은 최적점을 찾아 일반화 성능을 개선
    - **Seed Ensemble & Resize Ensemble:** 서로 다른 시드를 고정해 여러 모델을 학습하거나, 입력 이미지 크기를 다르게 하여 모델을 학습한 후 앙상블하는 방법
- **Pseudo Labeling:**
    
    - 우수한 성능의 모델을 이용해 Test 데이터셋에 대해 예측한 결과 중 높은 확신도(예: threshold 0.9 이상의 값)를 선택해, 이를 추가 학습 데이터로 활용하는 방법
- **외부 데이터 활용 및 그 외 팁:**
    
    - 외부 데이터를 추가해 데이터 부족 문제를 보완하거나, Encoder의 마지막에 Classification Head를 달아 모델의 수렴을 도와주는 방법 등 다양한 추가 실험 아이디어를 소개  


---

#### 2. 대회에서 사용하는 기법 및 트렌드

- **최근 대회 트렌드:**
    
    - 대회에서는 학습 이미지의 크기와 양이 매우 큰 경우가 많아, 제한된 시간(2주 미만) 내에 효율적으로 실험해야 함
    - Mixed Precision Training(FP16)과 같이 모델 학습 시간을 단축시키고, 경량화된 모델이나 일부 데이터로 빠른 실험을 진행하는 전략이 중요
- **이미지 전처리 기법:**
    
    - **Resize 및 Sliding Window:**
        - 원본 이미지가 너무 큰 경우 전체를 resize하거나, 일정 크기의 패치로 나눠서 모델에 입력하고, 예측된 결과를 원본 크기로 복원하는 방법
        - Sliding Window 기법에서는 window 크기와 stride 설정이 중요하며, 겹치는 영역을 통해 중복 예측으로 앙상블 효과를 얻는 방법도 소개됨
    - **추가 팁:**
        - 불필요한 배경 영역은 샘플링을 줄여 학습 속도를 개선하는 등의 전략도 고려

---

#### 3. Label Noise에 대한 대응 전략

- **Label Noise 문제:**
    - 세그멘테이션에서는 픽셀 단위의 annotation 특성상 경계 부근의 라벨 노이즈가 자주 발생하며, 클래스 간 불일치나 annotation 오류가 문제됨
- **해결 방법:**
    - **Label Smoothing:** Hard target 대신 Soft target으로 Loss를 계산해 모델이 과도하게 확신하지 않도록 유도
    - **Pseudo Labeling을 활용한 전처리:** 기존 noisy 라벨 대신, 모델 예측값을 바탕으로 pseudo label을 생성하여 학습 데이터로 재활용하는 방법
    - 관련 논문과 코드들을 참고해, 다양한 접근법을 적용하는 것이 중요

---

#### 4. 대회 평가 트렌드 및 기타 팁

- **평가 기준:**
    - 기존 Accuracy, mIoU 외에도 학습 시간, 추론 시간, 모델 경량성 등 실시간 및 효율성 측면이 중요한 평가 요소로 등장
- **모델 개선 팁:**
    - 출력 마스크를 시각적으로 확인해 큰 객체와 작은 객체, 그리고 특정 클래스의 성능을 별도로 분석하고 개선 아이디어를 도출할 필요가 있음

---

#### 5. 모니터링 도구 활용

- **Weights & Biases (W&B):**
    - 실험 결과를 실시간으로 모니터링하고, loss, learning rate, 각종 metric의 변화를 추적할 수 있음
    - 여러 실험 결과를 비교 분석하여 최적의 하이퍼파라미터를 찾고, 모델의 학습 상태를 시각화함

---
