---
{"dg-publish":true,"permalink":"/naver-boost-camp/week-4/","created":"2025-01-07T18:04:05.212+09:00","updated":"2025-01-08T20:18:47.661+09:00"}
---

[[NaverBoostCamp/Week 5 학습 정리\|Week 5 학습 정리]]
# 강의 복습 목차

> [!NOTE]
> > 1. CNN & ViT
> > 2. Self-supervisedtraining
> > 3. CNN Visualizing & Data Augmentation
> > 4. Segmentation & Detection
> > 5. Computational Imaging

# 1. CNN & ViT

## CNN

- CNN은 이미지 처리와 컴퓨터 비전 분야에서 널리 사용되는 인공 신경망의 한 유형입니다. CNN은 image classification, detection, segmentation 등 다양한 작업에서 탁월한 성능을 보이며, 특히 이미지 데이터에서 중요한 특징(feature)을 자동으로 추출하는 데 강점을 가지고 있다. VGGNet과 ResNet이 일반적으로 백본 모델로 많이 사용된다
- 기존 Fully Connected Layer 구조에서는 각 클래스마다 하나의 프로토타입을 가지게된다. 또한 각 Pixel이 input으로 들어가기 때문에 Parameter 수도 늘어나고 loacl 정보도 알 수 없었다. ⇒ 이를 CNN으로 해결하였다.

### Briefhistory

- LeNet-5는 1998년에 Yann LeCun에 의해 소개된 간단한 CNN 구조입니다.
- AlexNet은 ImageNet 데이터셋을 이용해 훈련된 더 큰 네트워크로, ReLU 활성화 함수와 드롭아웃 규제 기법을 사용하여 성능을 향상시켰습니다. 또한 GPU를 병렬로 사용 가능했습니다.
- VGGNet은 단순하지만 깊은 구조로, 작은 3x3 합성곱 필터를 사용하여 더 깊은 네트워크를 만듭니다.
- ResNet은 '잔차 연결(residual connections)'을 도입하여 더 깊은 네트워크를 효과적으로 훈련할 수 있게 합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/3e03917d-e280-4839-b75d-52858a31badf/image.png)

## ViT

- ViT는 No recurrence! No convolution! But attention! ⇒ 오직 Attention만을 사용하여 만든 모델이다. 기존 CNN에서의 Long-term dependency를 해결하며 Transformer를 Computer Viston에서 사용한다.
- ViT구조
    1. 이미지를 고정된 사이즈의 patch로 자른다. ⇒ 2D 이미지를 1D Sequence로 변환한다.
    2. positional embedding 과 CLS Token을 추가한다. ⇒ patch로 잘린 이미지에는 원본의 어디서부터 추출되었는지 위치정보가 없기 때문에 이를 Positional Embedding을 통해 알려준다. [class] token을 embedding patch 맨 앞에 추가한다.
    3. Transformer Encoredr ⇒ 사진의 오른쪽 부분인 Encorder를 진행한다.
    4. MLP Head ⇒ 학습 가능한 cls token을 이용해 classification을 할 수 있게 된다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/09fb8d94-40d9-4df3-89e6-2c757ace1ddc/image.png)

- 초기 ViT에서 발전된 Swin Transformer가 있다.
    - 먼저 이미지를 작은 패치로 나누고, 각 패치 내에서 로컬 윈도우 기반의 self-attention 메커니즘을 적용합니다.
        
    - 윈도우를 계층 간에 이동시켜(window shifting) 더 넓은 수용 영역과 더 나은 정보 통합을 제공합니다.
        
    - 이를 통해 높은 해상도 이미지 처리에서 효율적이며 계산 비용도 적게됩니다.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/b0d8d81f-fd52-4ded-bbdf-29cc908fe297/image.png)
        

# 2. Self-supervisedtraining

: ViT와 같이 사용하기 좋다.

## Masked Auto encoders (MAE)

기존에 가지고 있는 Ground truth 이미지에서 대부분(약 75%) 부분을 Mask처리하여 input을 만든다.

이를 Encorder와 Decorder를 통해 Ground truth를 예측하는 방식이다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/b2d0b7ab-ea32-4c91-9890-bf27e3b47c0b/image.png)

## DINO

Student - Teacher 네트워크 구조를 통해 Teacher 네트워크의 출력을 Student 네트워크가 학습할 수 있도록 하는 방식이다. 이를 통해 이미지의 특성을 잘 찾을 수 있게 된다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6e9ef343-4076-4ac0-a00f-e9a48602e330/image.png)

# 3. CNN Visualizing & Data Augmentation

## CNN visualization

CNN의 내부는 통칭 black box라고 불린다. 따라서 우리는 어떤 방식으로 output이 나오는지 알기 어렵다. 하지만 이러한 black box 부분을 알 수 있다면 어떻게 성능을 올릴지, 왜 이렇게 작동하는지, 어디에서 문제가 생겼는지 등을 알 수 있게됩니다.

이러한 이유들로 CNN 내부를 시각화 하려는 방법들이 많이 나오게 됩니다.

1. **Nearest Neighbors in feature space**: 이미지의 특징 공간에서 가까운 이웃을 찾는 방법으로, 유사한 특징을 가진 이미지를 검색하거나 군집화하는 데 사용됩니다. 특징 벡터 간의 거리를 계산하여 가장 유사한 이미지를 찾습니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6b1758ad-f7ce-47ac-96f5-1d491dae4c5c/image.png)

1. **Maximally activating patches**: 신경망의 특정 뉴런을 최대한 활성화하는 입력 패치(이미지의 부분)를 찾는 방법으로, 모델이 어떤 패턴을 학습했는지 이해하는 데 사용됩니다. 이는 특정 필터가 활성화되는 원인을 시각적으로 탐색합니다.
    
    ![어떤 부분을 학습에 잘 이용되었는지 보여줌](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/adfad284-fd53-4d76-8e3b-86e46ddd49a6/image.png)
    
    어떤 부분을 학습에 잘 이용되었는지 보여줌
    
2. **Class visualization**: 특정 클래스에 대한 신경망의 내부 표현을 시각화하는 기법으로, 모델이 각 클래스를 어떻게 **"보는지"** 이해할 수 있습니다. 이는 특정 클래스를 가장 잘 나타내는 입력을 생성하는 방식으로 수행됩니다.
    

![타조를 모델은 오른쪽과 같이 본다](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a356f2d8-d8d8-4530-b4f5-4daf959eca55/image.png)

타조를 모델은 오른쪽과 같이 본다

1. **Class activation mapping (CAM)** : 이미지의 어떤 부분이 모델의 특정 클래스 예측에 기여했는지 시각화하는 방법입니다. 이는 마지막 합성곱 계층의 출력과 완전 연결 계층 대신에 GAP(Global Average Pooling)을 적용하여 구현합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/65606c05-beb1-4ddb-a8da-e77839889514/image.png)
    
2. **ViT Visualization**: Vision Transformer(ViT)의 시각화 방법으로, 자기 주의 메커니즘을 사용하여 이미지의 패치 간 상호작용을 시각화합니다. 이를 통해 모델이 이미지의 어떤 부분에 주목하는지 확인할 수 있습니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/15c38fcd-ae18-41e5-ad70-84ec171b02cd/image.png)
    

## Data Augmentation

이미지 데이터셋은 현실 세계의 데이터 분포를 제대로 표현할 수 없다.

따라서 여러 기하학적 연산을 통해 같은 이미지에서도 다양한 이미지로 양을 늘릴 수 있게 됩니다.

예를 들어, 상하반전, 좌우반전, 회전, 밝기 조절, Crop, CutMix등의 방법이 있다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/8b7c53b7-a589-4bfd-9db7-76da2f1edbf7/image.png)

- 특히 CutMix를 진행할 시 성능이 더 잘 좋아지는 것을 알 수 있었다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/7912dff3-9a8a-4f18-ae17-2f7bdc80fee6/image.png)

## Synthetic data

현실 세계의 데이터를 모방하거나 확장하기 위해 **인위적으로 생성된 데이터**입니다.

머신러닝 모델의 훈련을 위해 사용할 수 있으며, 데이터의 다양성을 높이고 실제 데이터를 수집하기 어려운 경우에 특히 유용합니다.

합성 데이터는 개인정보 보호를 강화하고 데이터 편향을 줄이는 데 도움이 됩니다.

예를 들어, 이미지 데이터 증강, 텍스트 데이터 생성, 시뮬레이션 데이터를 통한 다양한 시나리오 테스트 등이 포함됩니다. 이를 통해 모델의 성능과 일반화 능력을 향상시킬 수 있습니다.

# 4. Segmentation & Detection

## Semantic segmentation

이미지의 각 픽셀을 특정 클래스에 할당하는 작업으로, 도로, 건물, 하늘과 같은 다양한 객체를 구분합니다.

이를 통해 컴퓨터가 이미지의 모든 부분을 이해하고, 이미지의 세부적인 정보까지 분석할 수 있게 합니다. 자율 주행, 의료 영상 처리, 로봇 비전 등 다양한 분야에서 활용됩니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/1a56c6df-45cc-468b-8ffe-8485e515993e/image.png)

### FCN

입력 이미지의 크기를 유지하면서 각 픽셀을 분류하는 완전 합성곱 신경망 구조입니다.

이미지 분류를 위해 전통적인 신경망의 완전 연결 계층을 제거하고, 모든 계층을 합성곱 계층으로 대체하여 효율적인 픽셀 단위 분류를 수행합니다.

다양한 해상도의 특징 맵을 결합하여 정확한 세그멘테이션 결과를 제공합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/dde44bde-b21b-4eaa-a0a4-9e0eac5126c3/image.png)

### UNet

U자형 인코더-디코더 구조를 갖춘 합성곱 신경망으로, 인코더는 이미지의 특징을 추출하고 디코더는 원본 해상도로 복원하며 세부적인 정보를 보존합니다. 디코더 진행 시 인코더의 skip-connection을 받아 더 정확한 복원이 가능해진다.

각 인코딩 단계에서의 특징을 대응되는 디코딩 단계로 전달하여 세밀한 세그멘테이션을 수행합니다. 주로 의료 영상에서 장기, 종양 등의 분할 작업에 널리 사용됩니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/2a3812b1-a8a3-4ae5-b7fb-35efb110fcaf/image.png)

## Object Detection

이미지 내에서 여러 객체의 위치와 클래스를 동시에 예측하는 작업으로, 각 객체를 둘러싸는 바운딩 박스를 생성합니다. Faster R-CNN, YOLO, SSD와 같은 다양한 알고리즘이 있으며,

각각 속도와 정확도 면에서 특징이 다릅니다. 이 기술은 자율 주행 차량, 보안 시스템, 로봇 비전 등에서 실시간 객체 추적과 탐지를 위해 사용됩니다.

### R-CNN

- R-CNN은 객체 탐지 알고리즘으로, 이미지에서 여러 개의 제안된 영역(Region Proposals)을 생성하고, 각 영역을 CNN을 통해 특징 맵으로 변환하여 개별적으로 분류합니다.
    
    ⇒ 2 stage ⇒ 계산 속도가 느리고 메모리 사용이 많다는 단점이 있습니다.
    
- 제안된 영역에서 CNN을 사용하여 객체가 존재할 가능성을 평가하고, 객체의 경계 상자(바운딩 박스)를 조정하여 최종 위치를 예측합니다.
    
- R-CNN에서 Fast R-CNN, Faster R-CNN, Mask R-CNN으로 점점 발전해 나간다.
    

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/edcaf7f7-7c27-4ca2-a0cd-17bad6f96393/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/44fed4e3-4974-466c-95d8-501472b9d764/image.png)

### Yolo

- YOLO는 실시간 객체 탐지 알고리즘으로, 입력 이미지를 하나의 패스로 처리하여 모든 객체의 위치와 클래스를 동시에 예측합니다.
    
    ⇒ 1 stage ⇒ 낮은 계산 비용으로 좋은 성능을 유지하지만, 작은 객체에 대한 탐지 정확도는 상대적으로 낮을 수 있습니다.
    
- 이미지 전체를 그리드로 나누고, 각 그리드 셀에서 바운딩 박스와 클래스 확률을 예측하는 방식입니다.
    

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/0eb6b565-2a5a-4b8b-8e91-a5ba526f379c/image.png)

- 1 stage detecter와 2 stage detecter의 차이는 ROI pooling이 명시적으로 있는지 없는지 차이이다.

## Instance segmentation

이미지 내의 각 객체를 개별적으로 구분하고, 각 객체의 경계를 정확하게 추출하는 기술입니다.

객체 탐지와 세그멘테이션을 결합하여 이미지 내에서 객체의 위치와 모양을 동시에 파악합니다.

이는 자율 주행, 의료 영상 분석, 로봇 비전 등에서 객체의 세밀한 분석이 필요한 경우에 사용됩니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/ac8a4e84-6425-4099-92b4-6f7155659c00/image.png)

### Mask R-CNN

Faster R-CNN을 기반으로 한 인스턴스 분할 모델로, 객체 탐지와 픽셀 수준의 분할을 결합하여 각 객체의 정확한 경계를 예측합니다.

ROI Align이라는 기술을 사용하여 분할 마스크를 보다 정확하게 생성하며, 각 객체에 대해 개별적인 마스크를 생성할 수 있습니다.

또한 Mask R-CNN의 확장을 통해 DensePose R-CNN, Mesh R-CNN등을 만들 수 있다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/27d01bf9-1a75-4e14-8655-406b85795a83/image.png)

## Transformer-based methods (Detection)

트랜스포머 아키텍처를 사용하여 이미지에서 객체 탐지와 분할 작업을 수행하는 방법들로, 기존 CNN 기반 방법보다 유연하고 성능이 우수한 특징이 있습니다.

이들 모델은 전역적 self attention 메커니즘을 활용하여 이미지의 모든 부분 간의 종속성을 학습합니다.

주로 장거리 관계를 학습하는 데 강점을 가지며, 객체 탐지와 분할의 효율성을 높입니다.

### DETR

DETR은 Transformer의 Encoder-Decoder방식을 차용해왔고 CNN을 backbone구조로 사용한 모델이다. 특별하게 Decoder 구조 다음에 Prediction heads라는 구조가 있다. Prediction heads는 FFN구조로 Decoder의 embedding된 output을 받아 N개의 prediction에 대한 prob을 출력하게 된다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/410d505e-515d-4fb9-9eb5-3503a9655420/image.png)

### MaskFormer

트랜스포머 아키텍처를 기반으로 한 이미지 분할 모델로, 객체 탐지와 세그멘테이션 작업을 통합하여 보다 효율적인 분할 결과를 제공합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/db057df0-2c3c-492c-9cae-d434df6db4bf/image.png)

## Unified Model

다양한 컴퓨터 비전 작업을 하나의 모델에서 통합적으로 수행할 수 있는 접근 방식으로, 여러 작업을 동시에 해결할 수 있는 효율성을 제공합니다.

이러한 모델은 다양한 데이터셋과 작업 간의 공통된 특징을 학습하여 일반화 능력을 극대화합니다.

이를 통해 모델의 복잡성을 줄이고, 다양한 응용에서 일관된 성능을 발휘할 수 있습니다.

### Uni-DVPS

동적 시각 인식을 위한 통합 모델로, 다양한 시각 작업을 하나의 통합된 프레임워크 내에서 수행하도록 설계되었습니다.

객체 탐지, 세그멘테이션, 추적, 깊이 탐지 등 **여러 비전 작업을 동시에 처리**하며, 실시간 응용에서의 효율성을 극대화합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a5c11928-b2c1-4f4c-8008-006cfa21a6d3/image.png)

## Segmentation foundation model

세그멘테이션 작업을 위한 범용 기초 모델로, 다양한 비전 응용 프로그램에서 높은 성능을 발휘할 수 있도록 설계되었습니다.

범용 모델로서, 한 번의 학습으로 여러 세그멘테이션 작업에 적응할 수 있으며, 데이터 효율성을 극대화합니다. 대규모 데이터셋과 결합하여 다양한 시나리오에서 강력한 성능을 발휘합니다.

### SAM

모든 유형의 객체에 대해 범용 세그멘테이션을 수행할 수 있는 모델로, 다양한 세그멘테이션 작업을 하나의 통합된 프레임워크 내에서 처리합니다.

사용자가 정의한 모든 객체에 대해 동적으로 마스크를 생성할 수 있어, 높은 유연성과 범용성을 자랑합니다. 이는 특히 다양한 환경에서의 세밀한 객체 분할이 필요한 응용에서 유용합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/1505b155-7150-4e8f-85a7-610413057b60/image.png)

### Grounded-SAM

SAM의 개선된 버전으로, 객체의 공간적 맥락을 고려하여 더 정밀한 세그멘테이션 결과를 제공합니다. 단순히 객체를 분할하는 것뿐 아니라,

객체 간의 관계와 배경을 이해하여 보다 정교한 이미지 해석을 가능하게 합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/38009cf6-80e0-48f5-9067-efe1b486410b/image.png)

# 5. Computational Imaging

## Computational Imaging

### Computational Photography와 Computational Imaging

- **Computational Photography**: 이미지 촬영 파이프라인에 계산 과정을 추가하여 이미지 품질을 향상시킵니다.
- **응용 분야**: 고동적 범위(HDR) 이미지 촬영, 저조도 이미지 개선, 노이즈 제거, 초해상도(super-resolution), 블러 제거 등.

### 딥러닝 기반의 Computational Imaging

- **모델 구성 요소**: 이미지 복원 및 향상을 위해 모델, 데이터, 손실 함수를 필요로 합니다.
- **모델 아키텍처**: U-Net, 스킵 연결(skip-connection), 다중 스케일 구조가 많이 사용됩니다.
- **손실 함수**: 일반적으로 L2(MSE), L1 손실 함수를 사용하나, 지각적 손실(adversarial loss) 및 퍼셉션 손실(perceptual loss)이 더 효과적일 수 있습니다.
- Supervised data를 얻기가 힘듭니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/9b81b3d3-63cf-4f48-a843-3b1aa45b3de3/image.png)

## Training Data in Computational Imaging

### Case study 1 - Image 노이즈 제거

- **목적**: 이미지에서 노이즈를 제거하여 원래의 깨끗한 이미지를 복원하는 것.
- **노이즈 모델**: 가우시안 노이즈를 추가하여 Training Data를 만들어 노이즈 이미지 시뮬레이션을 수행합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/ace54d17-5bfc-457d-b503-5bb062963c44/image.png)

### Case study 2 - Image super resolution

- **목적**: 저해상도 이미지를 고해상도로 복원하는 것.
- **데이터 생성**: 고해상도 이미지를 수집하고, 다운샘플링하여 저해상도 이미지를 생성합니다.
- **현실적인 데이터셋**: **RealSR**과 같은 데이터셋은 실제 카메라 시스템을 모방하여 더 현실적인 학습 데이터를 제공합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e59dec87-4f4b-4ce5-8ebb-58b3eb1bbd8c/image.png)

### Case study 3 - Image deblurring

- **목적**: 카메라 흔들림이나 객체 움직임으로 인해 발생하는 블러를 제거하여 선명한 이미지를 복원하는 것.
- **데이터 수집**: 고프레임율 카메라를 사용하여 현실적인 블러 데이터를 수집하고, RealBlur 데이터셋과 같은 현실적인 블러 학습 쌍을 생성합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e5ca245e-f969-4341-a873-d94a2827ec8c/image.png)

### Case study 4 - Video motion magnification(확대)

- **목적**: 비디오에서 작은 움직임을 증폭시켜 육안으로 관찰하기 어렵던 미세한 움직임을 강조하는 것.
- **제약사항**: 실제 확대된 비디오 데이터가 없어 합성 데이터 생성 방법이 필요합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/4aabd619-2550-4b89-bd95-2fdb3f509748/image.png)

## Advanced Loss Functions

### 적대적 Loss (Adversarial Loss)

- **개념**: GAN(Generative Adversarial Network)을 활용하여 Real Data와 비슷한 Fake Data를 생성하며, 생성기와 판별기가 서로 경쟁하며 성능을 향상시킵니다.
- **적용**: 고해상도 이미지 생성(Super-Resolution GAN)과 같은 작업에 사용됩니다.
- **장점**: L1, L2 loss와 달리 디테일이 sharp해지고 명확해지며 디테일이 다 표현이 되는 것을 확인할 수 있다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/fc37b593-8ece-4cb9-aaa3-7979afec073e/image.png)

### 지각적 Loss (Perceptual Loss)

- **개념**: 사전 훈련된 네트워크(예: VGG)를 사용하여 생성된 이미지와 타겟 이미지 간의 스타일과 특징 손실을 계산합니다.
- **장점**: 더 현실적이고 높은 품질의 출력을 생성하며, 특정 시각적 인식을 모델링할 수 있습니다.

### 두 Loss 비교

- **Adversarial Loss (적대적 손실)**
    - **장점**: 사전 훈련된 네트워크가 필요하지 않아 다양한 응용 분야에 쉽게 적용할 수 있습니다.
    - **단점**: 생성기와 판별기가 서로 경쟁하면서 학습해야 하기 때문에 훈련과 코드 구현이 상대적으로 어렵습니다.
- **Perceptual Loss (지각적 손실)**
    - **장점**: 순방향과 역방향 계산만으로 훈련할 수 있어 구현이 간단하며, 더 적은 양의 훈련 데이터로도 효과적으로 학습할 수 있습니다.
    - **단점**: 학습된 손실을 측정하기 위해 사전 훈련된 네트워크가 필요합니다.

## 4. Extension to Video

### 4.1 깜박임 문제 (Flickering Problem)

- **문제점**: 이미지 기반 방법을 비디오에 직접 적용할 경우 프레임 간의 시간적 불일치가 발생해 깜박임 문제가 생깁니다.
- **해결책**: 시간적 일관성을 유지하기 위해 짧은 시간과 긴 시간의 시간적 손실을 고려한 학습 방법을 사용합니다.

### 4.2 비디오 처리 (Video Processing)

- **전체 파이프라인**: 주어진 비디오 프레임과 처리된 프레임을 바탕으로 다음 프레임을 예측하는 순환 신경망을 사용합니다.
- **손실 함수**: 단기 및 장기 시간적 손실과 지각적 손실을 포함하여 모델을 훈련시킵니다.