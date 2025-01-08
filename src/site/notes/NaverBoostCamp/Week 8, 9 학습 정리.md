---
{"dg-publish":true,"permalink":"/naver-boost-camp/week-8-9/","created":"2025-01-07T18:10:39.725+09:00","updated":"2025-01-08T20:21:55.194+09:00"}
---

[[NaverBoostCamp/Week 11 학습 정리\|Week 11 학습 정리]]
> [!NOTE]
> > **1. OD Oveview
> > 2 Stage Detectors
> > 3. OD Library
> > 4. Neck
> > 5. 1 Stage Detectors
> > 6. EfficientDet
> > 7. Advanced OD1
> > 8. Advanced OD2
> > 9. Ready for Competition
> > 10. OD in Kaggle

# **1. OD Oveview**

## Evaluation

- mAP
    
    ⇒ 각 클래스당 AP의 평균
    
    계산 과정: 굿노트([Object Det] (1강) OD Overview (1) p.18)에서 확인
    
- IOU
    
    ⇒ Bbox와 GT가 겹치는 정도
    
- FPS
    
    ⇒ 초당 몇 프레임을 처리 가능한지 평가하는 속도평가
    
- FLOPs
    
    ⇒ 모델이 얼마나 빨리 동작하는지 측정하는 Metric, 연산량을 계산
    

## Library

- MMdetection
    
    ⇒ Pytorch 기반인 OD 오픈소스 라이브러리
    
- Detectron2
    
    ⇒ OD와 segmentation 알고리즘을 제공(보통 OD만 사용하긴함)
    

## ETC

- OD 특성
    1. 통합된 library의 부재
    2. 엔지니어링 적인 측면이 강함
    3. 복잡한 파이프라인
    4. 높은 성능을 위해서는 무거운 모델을 활용
    5. Resolution이 성능에 많은 영향을 끼침

# **2. 2 Stage Detectors**

⇒ 1단계: 위치 파악(localization), 2단계: 해당 위치에 있는 객체가 무엇인지 파악(classification)

## R-CNN

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a0d8e968-a7c9-4189-8cae-2e39ccab13a4/image.png)

- Extract Region proposals (약 2000개 후보(ROI(Region of interest)) 생성)
    
    ⇒ 이미지 내에 객체가 있을 것 같은 후보군을 뽑아내는 과정
    
    - Sliding window ⇒ 너무 많은 후보가 생기고 대부분이 배경(Negative sample)임
        
    - Selective search ⇒ 여러번 반복하면 후보군을 줄여나감 (효율적)
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/485e7c37-3205-4258-8fc8-3df98a037a4c/image.png)
        
- Compute CNN Features
    
    ⇒ ROI에 대한 특징을 뽑아냄
    
    - warped: CNN의 마지막 부분인 FC layer의 input size가 고정 되어 있기 때문에 ROI의 크기를 고정해야함
- Classify regions
    
    ⇒ CNN을 통해 나온 feature를 SVM을 통해 분류, Bbox regression을 이용해 bbox를 예측
    

> 단점

1. 2000개의 ROI가 각각 CNN을 통과
2. 강제 Wrap은 성능 하락 가능성을 가짐
3. CNN, SVM, Bbox regressr 모두 따로 학습
4. End-to-End X

## SPPNet

⇒ R-CNN의 단점인 Wrap과정과, 2000개의 ROI가 CNN을 통과한다는 단점을 Spatial pyramid pooling을 통해 보완

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/3fa43c1b-5a07-42c0-9ace-d8668639e6da/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/0c09aed0-33ae-4ad6-94a7-2bd4b21173a3/image.png)

## Fast R-CNN

⇒ CNN, SVM, Bbox regressr 모두 따로 학습한다는 단점을 해결

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e70a781a-7555-4e5f-b12d-403d338f6816/image.png)

- 이미지를 CNN에 넣어 feature를 추출 (CNN을 한 번만 사용)
    
- RoI Projection을 통해 feature map에서 RoI를 계산
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/617d1534-67f9-4c50-8c0f-b5afeafecdf7/image.png)
    
- **RoI Pooling**을 통해 일정한 크기의 feature가 추출
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/d85f3737-2401-430a-bdcd-9c101f21aec0/image.png)
    
- Fully connected layer 이후, Softmax Classifier과 bounding Box Regressor
    

## Faster R-CNN

⇒ Fast R-CNN + RPN (Region Proposal Network), End-to-End 형태로 만들어짐

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/3619adf1-410f-4db2-a368-00f81b4fe85c/image.png)

- 기존에 사용하던 Selective Search방법 대신에 RPN을 통해 RoI 계산 (Anchor Box 개념 등장)
    
    - Anchor Box: 여러 비율의 Bbox를 미리 설정
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/287ac6b8-ffad-4454-9314-0b6ba3eda627/image.png)
    
- NMS
    
    ⇒ 유사한 RPN Proposal을 제거하기 위해 사용, Class score를 기준으로 proposal 분류
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/37200e92-53dd-4c0f-a9c7-c172ae2a680d/image.png)
    
- Summary
    

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/d4afd879-ff06-469d-862f-9d04b2cfd4f4/image.png)

# **3. OD Library**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/0888e92a-2902-428b-8382-ba50faf08478/image.png)

## MMDetection

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/7ea83cb2-3f6d-4da9-84ef-be697aebb6fa/image.png)

## Detectron2

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e5a8cc52-0b64-4a4f-a314-cc0244e78fba/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/57c7f181-e7a7-41f4-88bb-ec52a85e9e2a/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e808676c-3fee-4ab2-8201-b6985e534773/image.png)

- 자세한 사용법은 **[Object Det] (3강) Object Detection Library 확인할 것**

# **4. Neck**

⇒ Backbone의 마지막 feature map만 사용하는 것이 아니라 중간의 feature map에 대해서도 RoI를 추출할 수 있도록 하는 방법

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/3bfc2862-edc9-406f-b0f4-7cefec9cfe4d/image.png)

- Neck의 필요성
    
    ⇒ 작은 객체는 low level에서 잘 탐지할 수 있고 큰 객체는 high level의 feature map에서 잘 팢을 수 있다 → 다양한 feature map을 사용한다면 크고 작은 객체를 잘 찾을 수 있다.
    

## Feature Pyramid Network (FPN)

- high level에서 low level로 semantic 정보 전달 필요
- top-down path way 추가

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/b4a6b481-4d33-4683-8e6e-10476bf0496c/image.png)

- Problem
    
    ⇒ 위 그림에서처럼 네트워크가 짧아 보이지만 실제 Backbone 모델의 깊이는 상당히 깊다 → low level의 feature map이 high level feature map에 잘 전달되지 않음
    

## Path Aggregation Network (PANet)

⇒ FPN의 문제점을 해결하기 위해 Bottom Up way를 하나 더 추가, 이후 Adaptive Feature Pooling을 통해 각각의 feature map에 RPN이 적용되여 RoI를 생성해서 마지막으로 하나의 Vector로 만듦

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6dbff34b-ebaf-4398-9c5f-5af8508364d2/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a107d07e-e67b-489a-9260-8ef1c5869f49/image.png)

## DetectoRS

- 구성
    - Recursive Feature Pyramid (RFP)
    - Switchable Atrous Convolution (SAC)
- Recursive Feature Pyramid (RFP)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/767f906c-eefe-4654-bc71-e0d9fcbe596d/image.png)

⇒ Neck을 이용해서 다시 Backbone을 학습하는 방식 ⇒ FLOPs 가 많이 증가하게 됨

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/d788ad45-c373-4972-b413-bd65ab172363/image.png)

- ASPP
    
    ⇒ Receptive field를 크게 사용하고 싶어 사용
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/628a42cb-2f3d-42a1-b145-2596707771db/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/599b82e3-0d5b-420f-8d56-a0728d362703/image.png)
    
- Bi-directional Feature Pyramid (BiFPN)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/064f8769-4f30-458e-aff2-f6f2568614f0/image.png)
    
    - Weighted Feature Fusion
        
        ⇒ FPN과 같이 단순 summation을 하는 것이 아니라 각 feature별로 가중치를 부여한 뒤 summation, eature별 가중치를 통해 중요한 feature를 강조하여 성능 상승 (모델 사이즈의 증가는 거의 없음)
        
- NASFPN
    
    ⇒ FPN 아키텍처를 NAS를 이용해서 찾는다는 아이디어
    
    - 단점
        - Parameter가 많이 소요, 범용적인 아키텍처가 아님
        - 아키텍처를 찾기 위해서는 굉장히 많은 cost가 소비됨
- AugFPN
    
    ⇒ Feature map을 단순히 Maxpooling 하는 것과 단순 Summation은 비 효율적, 가중합을 사용해서 해결,
    
    - 주요 구성
        1. Consistent Supervision
        2. Residual Feature Augmentation
        3. Soft RoI Selection

# **5. 1 Stage Detectors**

⇒ 2 Stage Detectors는 시간이 너무 오래 걸려 Real world에서 사용하기 어려웠다. 이에 따라 Localization과 classification을 동시에 진행하는 1 Staget Detectors가 등장하게 된다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/29ab10af-6bf5-45df-946d-b62b5fd239cf/image.png)

- 속도가 상당히 빠름, 이미지의 부분을 보는 것이 아니라 이미지를 전체적으로 보기 때문에 객체에 대한 맥락적 이해도가 높다

## Yolo

- YOLO v1 : 하나의 이미지의 Bbox와 classification 동시에 예측하는 1 stage detector 등장
    
- YOLO v2 : 빠르고 강력하고 더 좋게
    
    - 3가지 측면에서 model 향상
- YOLO v3 : multi-scale feature maps 사용
    
- YOLO v4 : 최신 딥러닝 기술 사용
    
    - BOF : Bag of Freebies, BOS: Bag of Specials
- YOLO v5: 크기별로 모델 구성
    
    - Small, Medium, Large, Xlarge
- Yolo 특징
    
    - Region proposal 부분이 X, Bbox예측과 classification을 동시에 예측 → 이미지의 맥락적 이해도 높음
- Pipeline
    
    - 입력이미지를 SxS Grid로 나누기
    - 각 Grid마다 B개의 Bbox와 Confidence score를 계산
    - 각 Grid영역마다 C개의 class에 해당하는 확률 계산
- 단점
    
    - Grid 영역으로 나누기 때문에 Grid 보다 더 작은 객체는 탐지 할 수 없음
    - 신경망의 마지막 출력 Feature만 사용하기 때문에 정확도가 낮음

## SSD

⇒ Yolo의 단점을 해결하기 위해 나온 모델

- Extra convolution layers에 나온 feature map들 모두 detection 수행
    - 6개의 서로 다른 scale의 feature map 사용
    - 큰 feature map (early stage feature map)에서는 작은 물체 탐지
    - 작은 feature map (late stage feature map)에서는 큰 물체 탐지
- Fully connected layer 대신 convolution layer 사용하여 속도 향상
- Default box 사용 (anchor box)
    - 서로 다른 scale과 비율을 가진 미리 계산된 box 사용

## Yolo v2

- Better (정확도 향상)
    - Batch Normalization
        
    - High Resolution classifier
        
    - anchor box 도입
        
    - Fine-grained features
        
        ⇒ low level 정보를 압축한 Early feature map에 합쳐주는 passthrough layer 도입
        
    - 다양한 크기의 이미지를 이용해 학습
        
- Faster (속도 향상)
    - Backbone model을 Darknet으로 변경
- Stronger (더 많은 class 예측)
    - Imagenet 데이터 셋과 Coco 데이터 셋을 Work Tree를 구성해 약 9000개의 class를 가지는 데이터 셋을 구성

## RetinaNet

⇒ 객체가 있는 부분(positive sample) 보다 배경 부분(negative sample)이 더 많은 class embalance문제를 가진다. 이 문제를 해결하는 방향으로 연구한 모델

- Focal Loss
    
    ⇒ 쉬운 예제에 작은 가중치, 어려운 예제에 큰 가중치 → 결과적으로 어려운 예제에 집중
    

# **6. EfficientDet**

⇒ OD에서 속도가 중요하다. 따라서 Efficiency가 중요하게 됨

⇒ Backbone, FPN, and box/class prediction networks을 동시에 Scale Up하여 적절한 구조를 찾는다. (EfficientNet과 비슷하게)

- Efficient multi-scale feature fusion
    
    - EffcientDet 이전의 모델들은 multi-scale feature fusion을 위해 여러 Neck구조를 사용함 → 이 구조는 resolution 구분 없이 feature map을 단순 합을 하는 문제 존재
        
        ⇒ 각각 input feature map에 weight를 주는 BiFPN을 제시
        
        1. 하나의 간선을 가진 노드는 제거
        2. residual 간선 추가
        3. BiFPN을 여러번 반복하여 사용
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/331f6130-b8b6-4bb2-9fcb-b865fa1780b4/image.png)
        
- Model Scaling
    
    ⇒ EfficientNet에서 사용한 Compound Scaling 방법 사용
    
    - Backbone으로 EfficientNet B0 ~ B6을 사용

# **7. Advanced OD1**

## Cascade RCNN

⇒ high quality detection을 수행하기 위해선 IoU threshold를 높여 학습할 필요가 있음 → 성능 하락의 문제 존재 이를 해결하기 위해 연구된 모델

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/020d9317-d42b-4c78-a8a6-ecf71c0f7847/image.png)

## Deformable Convolutional Networks (DCN)

⇒ CNN의 문제점인 일정한 패턴을 지닌 convolution neural networks는 geometric transformations에 한계가 있다는 것을 해결

- Deformable convolution (다양한 모양의 커널 사용)
    
    ⇒ grid R을 이용해 어디의 pixel과 conv연산을 할 지 정함
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/8e19ab17-0b20-4094-b04d-43a31fdd8ce2/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/29155986-6126-419d-8587-8300f55be600/image.png)
    

## Transformer (DETR)

기존의 Object Detection의 hand-crafted post process 단계를 transformer를 이용해 없앰

- Transformer의 문제점
    
    - 굉장히 많은량의 Data를 학습하여야 성능이 나옴
        
    - Transformer 특성상 computational cost 큼
        
    - 일반적인 backbone으로 사용하기 어려움
        
        ⇒ Swin Transformer
        
- Swin Transformer
    
    - 적은 Data에도 학습이 잘 이루어짐
    - Window 단위를 이용하여 computation cost를 대폭 줄임
    - CNN과 비슷한 구조로 Object Detection, Segmentation 등의 backbone으로 general하게 활용

# **8. Advanced OD2**

## Yolo v4

- BOF (Bag of Freebies) : inference 비용을 늘리지 않고 정확도 향상시키는 방법
    - Data Augmentation: CutMix, Mosaic: 4장의 이미지를 합쳐서 진행
    - Semantic Distribution Bias: 데이터셋에 특정 라벨(배경)이 많은 경우 불균형을 해결하기 위한 방법
        - Label Smoothing
    - Bounding Box Regression: Bounding box 좌표값들을 예측하는 방법(MSE)은 거리가 일정하더라도 IoU가 다를 수 있음 → IoU 기반 loss 제안
        - GIoU: IoU 기반의 loss 함수
- BOS (Bag of Specials) : inference 비용을 조금 높이지만 정확도가 크게 향상하는 방법
    - Enhancement of Receptive field: Feature map의 receptive field를 키워서 검출 성능을 높이는 방법
    - Attention Module: SE, CBAM
    - Feature Integration: Feature map을 통합하기 위한 방법 ( = Neck)
    - Activation Function: ReLU, Mish, Swish
    - Post-processing method: 불필요한 Bbox를 제거하는 방법
- Selection of Architecture
    - Cross Stage Partial Network (CSPNet)
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6151ed6e-b71e-4185-970f-29cd0c4f9fa0/image.png)
        

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/f274ce61-63e0-4114-a354-919757762956/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/2ecc24a8-f0ae-4a0c-8866-91a32f910244/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/60731d6c-48f0-45d5-a3ba-a18d4f568f54/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6c0a95f1-7b20-45e0-b975-bc5b84f8f5f5/image.png)

## M2Det

⇒ Multi-level, multi-scale feature pyramid 제안, SSD에 합쳐서 M2Det라는 one stage detector 제안

- Feature pyramid 한계점: 객체의 shape, 복잡도에 대해서 제대로 대응하지 못함
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/ee03b562-17b3-4047-9c8d-2b3b0356321b/image.png)
    
- Architecture
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/95ae29fc-383e-45c7-9a6c-3e1f58e73006/image.png)
    
    - FFM : Feature Fusion Module
        - FFMv1: 서로 다른 2개의 scale의 feature map을 합쳐 sementic 정보가 풍부한 base feature 생성
        - FFMv2 : base feature와 이전 TUM 출력 중에서 가장 큰 feature concat, 다음 TUM의 입력으로 들어감
    - TUM : Thinned U-shape Module
        - Encoder-Decoder 구조: Decoder에서 여러 scale의 feature map을 만들어냄 이후 가장 큰 Resolution을 가지는 Feature map을 다시한번 FFM
    - SFAM : Scale-wise Feature Aggregation Module
        - TUMs에서 생성된 multi-level multi-scale을 합치는 과정
        - 동일한 크기를 가진 feature들끼리 연결 (scale-wise concatenation)
        - 각각의 scale의 feature들은 multi-level 정보를 포함
        - Channel-wise attention 도입 (SE block)

## CornerNet

⇒ Anchor Box가 없는 1 stage detector, 좌상단 우하단 점을 찾아 객체를 검출

- Corner를 이용하는 이유: 중심점을 잡게 되면 4개의 면을 모두 고려해야하는 반면, corner을 사용하면 2개만 고려

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a9e9f643-76b8-4552-bd19-f11d77a65728/image.png)

- Corner pooling
    - 대부분의 corner는 특징이 없는 배경, corner를 결정하기 위한 Corner Pooling 과정 사용

# **9. Ready for Competition**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/d1cd27ca-2c36-4696-b607-3369390c9b5b/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/1bb8e246-8d76-4999-98c4-bcf83c59c330/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/f04b46c9-a09a-4c7b-8d4b-34b3d9183607/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/589f53d3-bfe7-42c8-9003-e4bbffd30781/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/7f5973b7-a7cf-4ed0-ab4f-aabff9ba2786/image.png)

- mAP에 대한 오해
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/017b74df-e294-4fcb-bf65-6766e0447bfc/image.png)
    
    - 더 많은 bbox에 대해 AP를 측정한다고 해서 패널티가 주어지지 않음 ⇒ Bbox가 많을수록 점수적으로 이득만 있을뿐 손해가 없다
    - 이러한 문제는 추후 모델을 앙상블 하는 경우에도 위와 같은 문제가 발생할 수 있음 ⇒ 따라서 예측 후 시각화 하여 확인해가며 진행해야할 듯

## Validation set 찾기

1. Random split: 전체 데이터를 랜덤하게 Train / Valid 로 분리
2. K Fold validation: 전체 데이터를 일정 비율로 Train / Valid로 분리 ⇒ Split 수만큼의 독립적인 모델을 학습하고 검증
3. Stratified K fold: 데이터 분포를 고려하지 않는 K fold 방식과 달리, fold 마다 유사한 데이터 분포를 갖도록 하는 방법, 데이터 분포가 imbalance한 상황에서 좋음
4. Group K fold

## Data Augmentation

- Cut and mix images and boxes: 이미지에 있는 Bbox가 일부분 잘리는 Small box 문제가 발생함
- Mosaic: 이미지 전체부분을 4개 합쳐서 사용, small box 문제 없음
- Mosaic2: 4개의 이미지의 일부분을 합치는 방법, small box 문제 존재
- Mosaic3: 4개의 이미지의 일부분을 합치는 방법, small box 문제 없음

## Ensemble & TTA

- Soft NMS
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/65c403d0-b0f3-4342-aafc-3a75cd4e9b66/image.png)
    
- WBF(Weighted Box Fusion)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/3932e95e-fd5d-48fd-993f-f75644ae0d14/image.png)
    
- Seed Ensemble
    
    ⇒ Random한 요소를 결정 짓는 seed를 바꿔가며 여러 모델을 학습시킨 후 앙상블하는 방법
    
- Framework Ensemble
    
    ⇒ (Mmdetection + detectron) 또는 (pytorch + tensorflow + torchvision) 등 여러 라이브러리의 결과를 앙상블 하는 방법
    
- Snapshot Ensemble
    
    ⇒ 동일한 아키텍처이지만 서로 다른 local minima에 빠진 신경망을 앙상블 하는 방법
    
- Stochastic Weight Averaging (SWA)
    
    ⇒ 일정한 step마다 weight를 업데이트시키는 SGD와 달리, 일정 주기마다 weight를 평균 내는 방법
    

# **10. OD in Kaggle**

- 모델 다양성은 정말로 중요하다!
    - Resolution, Model(Yolo, Effdet, CornerNet, FasterRCNN), Library…
    - 특히 FastRCNN, Yolov5가 자주 나옴
- Heavy augmentations은 거의 필수적이다!
    - 탑 솔루션들의 공통된 augmentations에는 무엇이 있을까?
- CV Strategy를 잘 세우는 것은 shake up 방지에 있어서 정말 중요하다!
- 체계적인 실험 역시 정말 중요하다!
- Team up은 성능향상의 엄청난 키가 될 수 있다!
    - 단, 서로 다른 베이스라인을 갖는 경우!