---
{"dg-publish":true,"permalink":"/2-naver-boost-camp/week-9-12/","created":"2025-02-26T15:44:19.054+09:00","updated":"2025-03-12T14:29:33.350+09:00"}
---

# Object Detection 평가: 성능과 속도 지표 이해하기

**Object Detection은 이미지 내 객체의 위치와 종류를 동시에 판별하는 복합적인 태스크**입니다. 단순한 분류(classification) 문제보다 더 복잡한 이 문제는 자율 주행, OCR, 질병 진단, CCTV 모니터링 등 다양한 분야에서 핵심 역할을 수행합니다. 이 포스트에서는 Object Detection의 성능 평가 지표와 속도 평가 지표에 대해 알아보겠습니다.

---

## 1. Object Detection 성능 평가: mAP (mean Average Precision)
![Pasted image 20250312140513.png](/img/user/Pasted%20image%2020250312140513.png)

### 1-1. Precision과 Recall 복습

- **Precision:** 예측한 객체 중 실제 객체가 차지하는 비율
- **Recall:** 실제 객체 중 예측된 객체의 비율

이 두 지표를 바탕으로 **Precision-Recall (PR) Curve** 를 그립니다.  
PR Curve 아래 면적을 계산한 값이 **Average Precision (AP)**이며, 여러 클래스에 대해 AP를 평균한 값이 **mAP**입니다. mAP 값이 높을수록 모델의 객체 검출 성능이 우수하다고 평가할 수 있습니다.

### 1-2. Bounding Box 평가: IoU (Intersection over Union)

Object Detection에서는 객체의 위치를 bounding box (bbox)로 나타내는데, 예측 bbox가 Ground Truth bbox와 얼마나 일치하는지를 평가하는 지표로 **IoU**를 사용합니다.

$\text{IoU}=$ 두 bbox의 교집합 영역두 bbox의 합집합 영역
$\text{IoU} = \frac{\text{두 bbox의 교집합 영역}}{\text{두 bbox의 합집합 영역}}$

- IoU 값이 1에 가까울수록 예측한 bbox가 GT bbox와 잘 일치합니다.
- 보통 IoU threshold를 설정하여, 예를 들어 $\text{IoU60}$이면  $\text{IoU} \geq 0.6$ 이면 두 bbox가 일치한다고 판단합니다. 이 threshold에 따라 mAP50, mAP60 등 다양한 기준으로 성능을 평가할 수 있습니다.

### 1-3. mAP 계산 과정

1. 각 클래스별로, 모델의 예측 bbox에 대해 IoU 기준을 만족하는 True Positive (TP)와 False Positive (FP)를 결정합니다.
2. confidence score에 따라 예측 결과를 내림차순으로 정렬한 후, Precision-Recall Curve를 생성합니다.
3. 각 클래스의 AP (Average Precision)를 계산하고, 이를 평균한 값이 mAP입니다.

---

## 2. Object Detection 속도 평가

성능 외에도, 실시간 처리가 요구되는 경우 속도 지표도 매우 중요합니다.

### 2-1. FPS (Frames Per Second)

- **정의:**  
    1초에 처리할 수 있는 frame의 수를 나타내며, FPS 값이 높을수록 모델의 속도가 빠릅니다.

### 2-2. FLOPs (Floating Point Operations)

- **정의:**  
    모델이 1초 동안 수행할 수 있는 부동소수점 연산 횟수를 의미합니다.  
    FLOPs 값이 높을수록 더 많은 연산을 빠르게 수행할 수 있음을 나타내지만, 모델의 효율성과도 밀접한 관련이 있습니다.

---

## 결론

Object Detection의 평가에서는 **성능**과 **속도** 두 가지 측면이 모두 고려됩니다.

- **성능 평가:**
    - mAP를 통해 모델이 얼마나 정확하게 객체를 검출하는지 확인하며, IoU 기준에 따라 mAP50, mAP60 등의 평가가 가능합니다.
- **속도 평가:**
    - FPS와 FLOPs를 통해 모델의 실시간 처리 능력을 측정합니다.

이러한 평가 지표들을 종합하여, 실제 서비스에 적합한 Object Detection 모델을 선택하고 최적화할 수 있습니다.

---
# Object Detection 모델의 발전과 2 Stage Detector의 구조

Object Detection은 이미지 내에서 객체의 위치와 종류를 동시에 판별하는 복합적인 태스크입니다. 단순한 분류 문제보다 복잡한 이 문제는 자율 주행, OCR, 질병 진단, CCTV 등 다양한 분야에서 핵심 역할을 수행합니다. 오늘은 객체 인지를 인간의 인식 과정과 유사하게 두 단계로 나누어 처리하는 **2 Stage Detector** 모델의 발전 과정을 살펴보겠습니다.

---

## 1. 2 Stage Detector 발전 과정

**2 Stage Detector는 인간이 물체를 인지할 때 먼저 위치를 파악한 후, 해당 물체의 종류를 판별하는 방식을 모방**합니다. 이 과정은 아래와 같이 R-CNN 계열 모델의 발전을 통해 이루어졌습니다.

### 1-1. R-CNN (Region-based Convolutional Neural Network)

- **프로세스:**
    1. **Selective Search:**
        - 이미지에서 잘게 segmentation 후, 연관 픽셀들을 병합하여 약 2000개의 Region of Interest (RoI)를 추출합니다.
        - 이전의 Sliding Window 방식보다 훨씬 효율적입니다.
    2. **RoI 조정:**
        - 모든 RoI를 CNN의 FC layer 입력에 맞게 동일한 크기로 조절합니다.
    3. **Feature 추출:**
        - 각 RoI를 Pretrained CNN(AlexNet 등)에 통과시켜 feature를 추출합니다.
    4. **분류 및 회귀:**
        - 추출한 feature를 SVM에 넣어 객체(또는 배경)를 분류하고, bbox regressor로 위치를 보정합니다.
        - Positive sample (예: IoU > 0.5)과 negative sample (예: IoU < 0.5, 혹은 < 0.3)을 사용하며, hard negative mining 기법으로 False Positive를 줄입니다.
- **단점:**
    - 약 2000개의 RoI가 각각 CNN을 통과해야 하므로 연산 비용이 매우 높습니다.
    - CNN, SVM, bbox regressor가 별도로 학습되기 때문에 end-to-end 학습이 이루어지지 않아 비효율적입니다.

### 1-2. SPPNet (Spatial Pyramid Pooling Network)

- **혁신:**
    - 입력 이미지의 크기를 강제로 고정하지 않고, CNN의 마지막 단계에서 고정 길이의 feature vector를 생성할 수 있도록 Spatial Pyramid Pooling을 도입합니다.
- **장점:**
    - 다양한 크기의 이미지를 그대로 처리할 수 있어 성능 하락을 방지합니다.
    - CNN을 한 번만 통과하므로 연산 비용이 크게 줄어듭니다.
- **단점:**
    - 여전히 CNN, SVM, bbox regressor를 개별적으로 학습해야 합니다.

### 1-3. Fast R-CNN

- **혁신:**
    - CNN, SVM, bbox regressor를 통합하여 end-to-end로 학습할 수 있도록 개선했습니다.
- **프로세스:**
    1. **Feature Map 추출:**
        - VGG16 같은 CNN을 사용해 전체 이미지를 한 번만 처리하여 feature map을 얻습니다.
    2. **RoI Projection & RoI Pooling:**
        - Selective Search를 통해 얻은 region proposal을 feature map에 매핑하여 고정 크기의 feature vector로 변환합니다.
    3. **분류 및 회귀:**
        - FC layer를 거쳐 softmax classifier로 객체 분류 및 bbox 회귀(Smooth L1 Loss) 수행.
- **데이터 구성:**
    - Positive sample: IoU > 0.5, Negative sample: 0.1 < IoU < 0.5 (비율 25% : 75%)
- **장점:**
    - 모든 구성 요소가 하나의 네트워크로 통합되어 학습 및 최적화가 용이합니다.

### 1-4. Faster R-CNN

- **혁신:**
    - Region Proposal Network (RPN)을 도입하여, CNN 내부에서 region proposal을 생성하고, Fast R-CNN 방식으로 처리합니다.
- **프로세스 (RPN 부분):**
    1. **Feature Map 추출:**
        - 이미지를 CNN에 입력하여 feature map을 생성합니다.
    2. **Sliding Window & Anchor Boxes:**
        - Sliding window 방식으로 feature map을 스캔하며, 9개의 다양한 크기 및 비율의 anchor box를 생성합니다.
    3. **Objectness Score & Regression:**
        - 각 anchor box에 대해 객체 존재 가능성(objectness score)과 bbox 보정을 위한 regression을 수행합니다.
    4. **Non-Maximum Suppression (NMS):**
        - 겹치는 bounding box들을 제거하여 최종 후보 영역을 결정합니다.
- **장점:**
    - RPN을 통해 region proposal 단계가 통합되어 전체 프로세스가 end-to-end로 이루어집니다.
    - 이전 모델들보다 빠르고, 성능이 향상되었습니다.

---

## 2. Object Detection 모델의 Framework

2 Stage Detector 모델은 **Backbone, Neck, Head**로 구성된 계층적 구조를 가지고 있습니다.
![Pasted image 20250312140932.png](/img/user/Pasted%20image%2020250312140932.png)
- **Backbone:**
    - 이미지에서 기본적인 feature를 추출하는 부분으로, 보통 사전 학습된 CNN(예: ResNet-50 등)이 사용됩니다.
- **Neck:**
    - Backbone에서 추출한 feature map을 재구성하여, 다양한 scale과 aspect ratio의 객체를 효과적으로 탐지할 수 있도록 가공합니다.
    - 예를 들어, Feature Pyramid Networks (FPN)이 Neck 역할을 합니다.
- **Head:**
    - Neck에서 가공된 feature map을 입력받아 각 객체의 위치와 클래스를 최종적으로 예측합니다.
    - **Dense Head:**
        - feature map의 dense location을 수행합니다.
    - **RoI Head:**
        - RoI pooling을 통해 얻은 feature를 기반으로 객체 분류와 bbox 회귀를 수행합니다.

---

## 결론

Object Detection 모델은 R-CNN에서 시작해 SPPNet, Fast R-CNN, Faster R-CNN으로 발전해 왔습니다.

- **R-CNN:**
    - Selective Search를 통해 2000개의 RoI를 추출하고, CNN, SVM, bbox regressor를 별도로 학습하여 객체를 검출하였습니다.
- **SPPNet:**
    - Spatial Pyramid Pooling을 도입하여 다양한 크기의 이미지를 처리하며, 연산 비용을 줄였습니다.
- **Fast R-CNN:**
    - 모든 구성 요소를 end-to-end로 학습하여 최적화와 성능을 향상시켰습니다.
- **Faster R-CNN:**
    - RPN을 통해 region proposal 과정을 내부화하고, NMS를 적용하여 빠르고 정확한 객체 검출을 실현하였습니다.

또한, 이러한 2 Stage Detector 모델의 구조는 Backbone, Neck, Head로 나뉘며, 이를 통해 다양한 객체의 크기와 형태를 효과적으로 검출할 수 있습니다.  

---

# Object Detection 모델의 발전: IoU Threshold부터 Transformer까지

초기 모델들은 객체의 위치와 클래스를 개별적으로 처리했지만, 시간이 흐르면서 효율성과 정확도를 높이기 위해 여러 가지 혁신적 기법들이 도입되었습니다. 이번 포스팅에서는 Fast R-CNN에서 시작해 Cascade R-CNN, Deformable Convolutional Networks, DETR, 그리고 Swin Transformer에 이르기까지의 발전 과정을 살펴보고, IoU Threshold 설정이 detection 성능에 미치는 영향을 함께 알아보겠습니다.

---

## 1. Fast R-CNN과 IoU Threshold의 영향

Fast R-CNN은 기존 R-CNN의 단점을 보완하여 end-to-end로 학습할 수 있도록 설계되었습니다.

- **기존 방식:**
    - IoU 0.5를 기준으로 Positive sample과 Negative sample을 구분하여 학습했습니다.
- **IoU Threshold 변화 효과:**
    - **높은 IoU Threshold (예: 0.6, 0.7):**
        - 모델이 학습 시 보다 엄격한 기준으로 bbox를 선택하므로, False Positive가 줄어들고 localization 성능이 향상됩니다.
        - 다만, 입력 데이터의 IoU가 높아야 이러한 기준이 효과적입니다.
    - **낮은 IoU Threshold:**
        - 입력 IoU가 낮은 경우, 낮은 threshold로 학습한 모델이 더 나은 localization 성능을 보입니다.
- **Detection Performance:**
    - 최종적으로, 높은 IoU threshold에서 학습한 모델은 높은 IoU 조건의 평가에서 우수한 성능을 나타냅니다.

이와 같은 실험 결과는 단순히 IoU threshold를 높인다고 해서 항상 성능이 향상되는 것이 아니라, 데이터의 특성과 학습 조건에 따라 적절한 threshold를 선택해야 한다는 점을 시사합니다.

---

## 2. Cascade R-CNN

Cascade R-CNN은 IoU threshold에 따른 모델 성능의 변동성을 체계적으로 개선하기 위해 고안되었습니다.
![Pasted image 20250312141835.png](/img/user/Pasted%20image%2020250312141835.png)
- **핵심 아이디어:**
    - 여러 개의 RoI head를 cascade(연쇄적으로) 쌓아, 각 단계마다 점진적으로 IoU threshold를 높여가며 학습합니다.
    - 이전 단계의 bbox가 다음 단계의 입력으로 들어가면서, 보다 정확한 위치 추정을 수행합니다.
- **효과:**
    - IoU threshold가 다른 여러 classifier를 연속적으로 학습시켜, 높은 IoU 조건에서 뛰어난 detection 성능을 보장합니다.

---

## 3. Deformable Convolutional Networks
![Pasted image 20250312141843.png](/img/user/Pasted%20image%2020250312141843.png)
기존의 고정된 convolution filter는 이미지에 기울기, 시점 변화, 포즈 변화 등의 geometric transform이 가해졌을 때 한계가 있었습니다.

- **Deformable Convolution의 핵심:**
    - Convolution filter의 각 kernel 요소마다 학습 가능한 offset을 도입하여, 입력 이미지의 다양한 기하학적 변형에 유연하게 대응합니다.
- **장점:**
    - 고정된 filter size의 한계를 극복하여, 객체가 있을 법한 위치를 보다 잘 포착할 수 있습니다.

---

## 4. DETR (DEtection TRansformer)
![Pasted image 20250312141850.png](/img/user/Pasted%20image%2020250312141850.png)
DETR은 Transformer를 object detection에 처음 도입한 모델로, 기존 모델들의 복잡한 후보 영역(NMS 등) 후처리 단계를 제거했습니다.

- **구성:**
    - Backbone으로 CNN을 사용하여 feature map을 추출한 후, Transformer에 통과시킵니다.
    - Transformer의 prediction head에서 N (> 한 이미지에 존재하는 객체 수)개의 bbox와 클래스 정보를 동시에 예측합니다.
    - Ground Truth의 부족한 수는 "no object"로 padding 처리하여 N:N 매핑을 수행합니다.
- **특징:**
    - NMS(Post-processing)가 필요 없고, end-to-end 학습이 가능해졌습니다.

---

## 5. Swin Transformer
![Pasted image 20250312141855.png](/img/user/Pasted%20image%2020250312141855.png)
Transformer는 원래 계산 비용이 매우 높지만, Swin Transformer는 CNN과 유사한 계층적 구조와 window 기반 self-attention을 도입해 이를 극복했습니다.

- **주요 구성 요소:**
    - **Patch Partitioning:** 이미지를 작은 패치로 나누어 처리합니다.
    - **Linear Embedding:** 각 패치를 선형 임베딩하여 특징 벡터로 변환합니다.
    - **Swin Transformer Block:**
        - **Window Multi-head Self Attention (W-MSA):** 각 window 내에서 self-attention을 수행하여 계산 비용을 줄입니다.
        - **Shifted Window MSA (SW-MSA):** window 간 경계를 보완하여 전체 receptive field를 확장합니다.
    - **Patch Merging:** feature map 크기를 줄이면서 채널 수를 늘려, 계층적 구조를 형성합니다.
- **장점:**
    - 계산 효율성이 높아지면서도, 객체 detection과 같은 태스크에서 높은 성능을 유지합니다.

---

## 결론

Object Detection 모델은 Fast R-CNN을 시작으로 Cascade R-CNN, Deformable Convolutional Networks, DETR, 그리고 Swin Transformer에 이르기까지 지속적으로 발전해왔습니다.

- **IoU Threshold의 영향:**
    - 적절한 IoU threshold 선택은 localization 성능과 False Positive 감소에 중요한 역할을 합니다.
- **Cascade R-CNN:**
    - 여러 단계의 RoI head를 통해, 다양한 IoU 조건에 효과적으로 대응합니다.
- **Deformable Convolution:**
    - geometric transform에 유연하게 대응하여 객체의 위치를 보다 정확하게 검출합니다.
- **DETR:**
    - Transformer를 도입해 복잡한 후보 영역 후처리 과정을 제거하고, end-to-end detection을 실현합니다.
- **Swin Transformer:**
    - 효율적인 계산과 계층적 구조를 통해 Transformer 기반 모델의 장점을 극대화합니다.
---
# Neck 모듈: Backbone과 Head 사이의 다리 역할

2 stage detector 모델은 크게 **Backbone, Neck, Head**로 구성됩니다.

- **Backbone:** 이미지에서 기본 특징(feature)을 추출
- **Head:** 추출된 feature를 바탕으로 객체의 위치 및 클래스 정보를 예측
- **Neck:** Backbone에서 추출한 다양한 수준의 feature map을 가공해, Head에 전달하는 역할을 합니다.

**기존의 object detection 모델들은 Backbone의 마지막 layer에서 추출된 단일 feature map을 RPN(Region Proposal Network)에 연결해 객체를 탐지**했습니다. **그러나 객체의 크기가 다양하기 때문에, 여러 단계의 feature map을 활용하여 정보의 풍부함과 세밀함을 동시에 확보하는 것이 필요**해졌습니다. 이번 포스팅에서는 다양한 Neck 구조와 그 발전 과정을 살펴보겠습니다.

---

## 1. Feature Pyramid Network (FPN)

FPN은 Neck 모듈의 대표적인 구조로, CNN을 통과하며 여러 수준의 feature map을 생성한 후, 상위 level의 semantic 정보(저해상도, 의미, 패턴)와 하위 level의 세부 정보(고해상도, 구조)를 결합합니다.
![Pasted image 20250312142309.png](/img/user/Pasted%20image%2020250312142309.png)

- **Top-Down Pathway:**  
    상위 level에서 하위 level로 Nearest Neighbor Upsampling을 적용하여 해상도를 점차 높입니다.
- **Lateral Connections:**  
    upsampling된 상위 level feature map과 하위 level feature map을 결합하여, 두 정보가 모두 반영된 풍부한 feature map을 생성합니다.

**장점:**

- 다양한 크기의 객체를 효과적으로 탐지할 수 있으며, 입력 이미지를 강제로 리사이징하지 않아 성능 저하를 방지합니다.
- 연산 비용을 줄이면서도 다중 scale 정보를 활용할 수 있습니다.

**한계:**

- top-down 경로만 사용하기 때문에, 하위 level의 세부 정보가 상위 level로 충분히 전달되지 않는 단점이 있습니다.

---

## 2. Path Aggregation Network (PANet)

PANet은 FPN의 한계를 극복하기 위해 도입되었습니다.

- **양방향 정보 전달:**  
    FPN의 top-down pathway에 더해, bottom-up 경로를 추가해 하위 level의 세부 정보를 상위 level로 전달합니다.
- **Adaptive Feature Pooling:**  
    각 객체의 크기에 맞춰 적절한 피라미드 레벨에서 특징을 pooling할 수 있도록 하여, 보다 유연한 대응이 가능합니다.

**장점:**

- 상위 level은 semantic 정보와 세부 정보 모두를 포함하게 되어, 보다 정밀한 객체 탐지가 가능합니다.

---

## 3. DetectoRS: Recurrent Feature Pyramid

DetectoRS는 PANet의 아이디어를 한 단계 발전시켜, 상위와 하위 level의 feature map 간 정보 교환을 **반복(recurrent)** 하여 더욱 풍부한 표현을 학습합니다.
![Pasted image 20250312142425.png](/img/user/Pasted%20image%2020250312142425.png)
- **Recursive Feature Pyramid:**  
    여러 번의 top-down, bottom-up 경로를 통해 feature map이 정보를 충분히 교환할 수 있도록 합니다.
- **Switchable Atrous Convolution (SAC):**  
    다양한 dilation rate(팽창률)를 동시에 적용하여, 각기 다른 스케일의 정보를 효과적으로 추출하고, receptive field를 확장합니다.

**장점:**

- 반복적인 정보 교환으로, 복잡한 객체와 다양한 크기의 객체에 대한 탐지 성능을 극대화할 수 있습니다.

---

## 4. Bi-directional Feature Pyramid (BiFPN)
![Pasted image 20250312142432.png](/img/user/Pasted%20image%2020250312142432.png)
BiFPN은 EfficientDet 모델에서 도입된 구조로, PANet처럼 top-down 및 bottom-up 경로를 모두 사용하면서도, 불필요한 연결은 제거해 보다 단순하면서도 효과적으로 중요한 정보를 결합합니다.

- **가중치 조절:**  
    중요한 feature map에 높은 가중치를 부여하고, 덜 중요한 feature map은 낮은 가중치를 주어, 최종 결합 시 중요한 정보가 더 크게 반영되도록 합니다.

**장점:**

- 경로가 간결해지면서도, 중요한 정보는 적절히 강조되어 전체 성능이 향상됩니다.

---

## 5. NAS-FPN

NAS-FPN은 Neural Architecture Search (NAS)를 이용하여 최적의 Feature Pyramid Network 구조를 자동으로 탐색하는 방법입니다.

- **자동 최적화:**  
    NAS를 통해 다양한 피라미드 구조와 연결 방식을 실험하고, 최적의 구조를 찾아내어 성능을 극대화합니다.
- **단점:**
    - COCO dataset과 ResNet을 기준으로 최적화되어 범용성이 떨어질 수 있음
    - 최적 구조를 찾기 위한 추가적인 search 비용이 발생함

---

## 6. Augmented Feature Pyramid Network (AugFPN)

AugFPN은 FPN의 한계를 극복하기 위해 여러 기법을 결합한 모델입니다.

- **Bottom-Up Path Augmentation:**  
    하위 level의 정보를 상위 level로 전달합니다.
- **Residual Feature Augmentation:**  
    다양한 scale의 feature map을 생성하고, 동일한 크기로 upsampling 후 가중치를 두어 합산합니다.
- **Soft RoI Selection:**  
    모든 scale의 feature에서 RoI projection 및 pooling을 수행한 후, channel-wise 가중치 계산을 통해 최종 feature를 summation합니다.

**장점:**

- 다양한 augmentation 기법을 결합하여, 보다 정밀하고 유연한 feature aggregation을 수행할 수 있습니다.

---

## 결론

Neck 모듈은 2 stage detector에서 Backbone과 Head 사이의 다리 역할을 하며, 다양한 scale의 feature를 효과적으로 결합해 객체 탐지 성능을 크게 향상시킵니다.

- **FPN:** Top-down pathway와 lateral connection을 통해 semantic과 세부 정보를 결합
- **PANet:** 양방향 경로를 추가하여 정보 전달을 강화
- **DetectoRS:** 반복적 정보 교환과 SAC를 통해 풍부한 표현 학습
- **BiFPN:** 중요한 feature에 가중치를 부여하며 간결한 구조로 정보 결합
- **NAS-FPN:** NAS로 최적 구조를 자동으로 탐색
- **AugFPN:** 다양한 기법을 결합하여 세밀한 feature aggregation 수행

---
# Object Detection 모델 발전과 최신 기술 동향

초기의 2 stage detector는 객체의 위치와 분류를 분리해 처리하여 높은 정확도를 보였지만, **처리 속도가 느렸습니다. 이에 반해, 1 stage detector는 전체 이미지를 한 번에 처리하여 실시간 성능을 크게 향상**시켰습니다. 이번 포스팅에서는 1 stage detector의 대표 모델들과 함께, 최신 기술들이 object detection에 어떻게 적용되고 있는지 살펴보겠습니다.

---

## 1. 1 Stage Detector의 등장

### 1-1. YOLO 시리즈

**YOLO (You Only Look Once)** 는 1 stage detector의 대표적인 예시입니다.

- **YOLO v1:**
    ![Pasted image 20250312142634.png](/img/user/Pasted%20image%2020250312142634.png)
    - GoogLeNet을 변형하여 24개의 convolution layer와 2개의 Fully-Connected (FC) layer로 구성됨
    - 입력 이미지를 7×7 grid로 나누고, 각 grid마다 2개의 bounding box와 confidence score, 그리고 conditional class probability를 예측
    - **NMS** (Non-Maximum Suppression)를 통해 최종 bbox 도출
    - **장점:** 이미지 전체를 보고 객체의 맥락적 정보를 반영하여 정확도가 높음
    - **단점:** grid보다 작은 객체 탐지에 한계가 있음
- **YOLO v2:**
    
    - Batch Normalization 도입, 앵커 박스 도입, K-means clustering을 통한 앵커 박스 크기 결정
    - 좌표 대신 offset 예측, Passthrough Layer를 통해 early feature map을 late feature map에 결합
    - **향상:** mAP가 약 2% 상승하고, 다양한 크기의 입력 이미지 지원 및 속도 개선
- **YOLO v3:**
    
    - Skip connection 적용, 서로 다른 3개의 스케일의 feature map을 활용한 multi-scale detection
    - FPN과 유사한 방식으로 상위 레벨의 semantic 정보와 하위 레벨의 세부 정보를 결합
- **YOLO v4:**

	![Pasted image 20250312142826.png](/img/user/Pasted%20image%2020250312142826.png)
    - 정확도와 속도를 모두 향상시키기 위해 **Bag of Freebies (BOF)** 와 **Bag of Specials (BOS)** 전략 도입
        - **BOF**: 추가적인 inference 비용 없이 데이터 증강, label smoothing, GIoU Loss 등으로 성능 향상
        - **BOS**: receptive field 확장, global attention, 후처리(NMS) 개선 등으로 정확도 강화
    - **Backbone:** CSPDarknet53를 사용하여 gradient 재사용 문제 개선
    - **특징:** 큰 입력 사이즈, 깊은 네트워크, 다양한 augmentation 및 multi-scale feature integration

### 1-2. SSD (Single Shot Multibox Detector)
![Pasted image 20250312142744.png](/img/user/Pasted%20image%2020250312142744.png)
- **구성:**
    - VGG16을 backbone으로 사용하고, extra convolution layer에서 6개의 서로 다른 스케일의 feature map을 추출
    - 각 feature map 셀에서 미리 계산된 default anchor box를 사용하여 객체의 위치와 클래스를 예측
- **학습:**
    - Hard negative mining과 NMS를 적용, localization loss와 confidence loss를 사용
- **장점:**
    - 여러 scale의 feature map을 사용해 작은 객체와 큰 객체 모두를 효과적으로 탐지

### 1-3. RetinaNet

- **문제 해결:**
    - 1 stage detector에서는 anchor box가 대부분 배경(negative sample)인 class imbalance 문제가 발생
- **해결:**
    - **Focal Loss:** cross entropy loss에 scaling factor를 도입, 쉬운 예제는 가중치를 낮추고 어려운 예제는 높은 가중치를 부여하여 학습 집중도 향상
- **효과:**
    - 배경 학습을 줄이고 실제 객체 학습에 집중하여 성능 향상

---

## 2. 최신 기술 및 Anchor-Free 접근법

### 2-1. M2Det
![Pasted image 20250312142839.png](/img/user/Pasted%20image%2020250312142839.png)
- **문제:**  
    기존 FPN은 단일 레벨의 정보만 활용하여 단순한 외형과 복잡한 외형을 동시에 처리하는 데 한계가 있음
- **해결:**
    - **MLFPN (Multi-Level Feature Pyramid Network):**
        - Backbone에서 나온 다양한 scale의 feature map을 fusion하는 여러 모듈로 구성
            - **FFMv1:** 서로 다른 scale의 feature map 2개를 합쳐 base feature map 생성
            - **TUM (Thinned U-shape Module):** encoder-decoder 구조로 다양한 scale의 feature 생성
            - **FFMv2:** 이전 TUM 출력 중 가장 큰 feature map을 합쳐 다음 TUM 입력
            - **SFAM (Scale-wise Feature Aggregation Module):** 채널-wise attention을 적용해 각 scale의 중요도를 조절
    - **결과:**
        - shallow level은 단순한 외형, deep level은 복잡한 외형을 효과적으로 탐지

### 2-2. Anchor-Free Approaches

- **CornerNet:**
	![Pasted image 20250312142903.png](/img/user/Pasted%20image%2020250312142903.png)
    - Anchor box 없이, 객체의 top-left와 bottom-right corner를 직접 예측
    - **Hourglass Network:** encoder-decoder 구조로 글로벌 및 로컬 정보를 모두 추출
    - **Corner Pooling:** 모서리 정보를 강화하여 정확한 corner 위치를 예측
- **CenterNet:**
    - 객체 중심점을 예측하여, 하나의 anchor box를 생성하는 방식
- **FCOS:**
    - 중심점으로부터 bbox 경계까지의 거리를 직접 예측

### 2-3. DETR (Detection Transformer)

- **혁신:**
    - Transformer를 object detection에 도입하여, 기존의 복잡한 후보 영역(NMS) 후처리 단계를 제거
- **구성:**
    - Backbone으로 CNN을 사용해 feature map을 추출한 후 Transformer에 입력
    - Transformer는 N개의 출력(객체 수)을 생성하며, 부족한 경우 "no object"로 padding
- **장점:**
    - End-to-end 학습 가능, NMS 불필요

### 2-4. Swin Transformer

- **문제:**
    - Transformer는 원래 계산 비용이 높아 실시간 object detection에 부적합할 수 있음
- **해결:**
    - **Swin Transformer:**
        - 이미지 패치를 나누어 처리하고, window 기반 self-attention(W-MSA)과 Shifted Window MSA(SW-MSA)를 도입해 계산 효율성을 높임
        - 각 stage마다 feature map 크기를 절반으로 줄여 효율적인 multi-scale feature extraction 제공
- **장점:**
    - 높은 효율성과 우수한 성능

---

## 결론

1 stage detector는 전체 이미지를 한 번에 처리함으로써, 2 stage detector에서 발생하는 region proposal의 연산 비용과 속도 문제를 크게 개선합니다.

- **YOLO 시리즈와 SSD:**
    - 실시간 성능과 맥락적 정보를 활용한 detection에 강점을 보입니다.
- **RetinaNet:**
    - Focal Loss 도입으로 클래스 불균형 문제를 해결합니다.
- **M2Det:**
    - multi-level, multi-scale feature integration을 통해 다양한 객체를 효과적으로 탐지합니다.
- **Anchor-Free 접근법 (CornerNet, CenterNet, FCOS):**
    - Anchor box의 복잡성을 제거하고, 보다 직관적인 방식으로 객체를 검출합니다.
- **Transformer 기반 모델 (DETR, Swin Transformer):**
    - End-to-end 학습과 계산 효율성 개선을 통해 최신 object detection 기술의 새로운 방향을 제시합니다.

---
# MMDetection: PyTorch 기반 Object Detection 딥러닝 라이브러리 활용하기

MMDetection이나 Detectron2와 같은 라이브러리를 사용하면 미리 갖춰진 configuration 파일을 기반으로 쉽게 object detection 모델을 구축하고 학습할 수 있습니다. 이번 포스팅에서는 MMDetection의 기본 구성 및 configuration 수정 방법, 그리고 커스텀 백본 모델 등록 방법에 대해 알아보겠습니다.

---

## 1. MMDetection 기본 사용법

MMDetection은 PyTorch 기반의 오픈소스 라이브러리로, 다양한 최신 object detection 알고리즘(Faster R-CNN, Mask R-CNN, RetinaNet 등)을 지원합니다.

### 1-1. 주요 Import

```python
from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.apis import train_detector  # (주의: 'apls'가 아니라 'apis' 입니다.)
from mmdet.utils import get_device
```

### 1-2. Configuration 파일 다루기

기본적으로 MMDetection은 미리 작성된 configuration 파일을 상속받아 필요한 부분만 수정해서 사용합니다. 예를 들어 Faster R-CNN을 기반으로 Trash detection 모델을 구성하는 방법은 다음과 같습니다.

```python
# configuration 파일 불러오기
cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

# 데이터 경로 설정
route = './dataset/'

# 클래스 수정
classes = ("General Trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
cfg.model.roi_head.bbox_head.num_classes = len(classes)

# training set 설정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = route
cfg.data.train.ann_file = route + 'train.json'
cfg.data.train.pipeline[2]['img_scale'] = (512, 512)   # resize 크기

# validation set 설정
cfg.data.val.classes = classes
cfg.data.val.img_prefix = route
cfg.data.val.ann_file = route + 'val.json'
cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

# test set 설정
cfg.data.test.classes = classes
cfg.data.test.img_prefix = route
cfg.data.test.ann_file = route + 'test.json'
cfg.data.test.pipeline[1]['img_scale'] = (512, 512)

# 기타 학습 설정
cfg.data.samples_per_gpu = 4
cfg.seed = 2020
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()
```

### 1-3. Dataset, Model 및 학습

```python
# Dataset 정의 (학습용)
datasets = [build_dataset(cfg.data.train)]

# Model 정의
model = build_detector(cfg.model)
model.init_weights()  # 가중치 초기화

# 학습 수행
train_detector(model, datasets[0], cfg, distributed=False, validate=True)
```

---

## 2. Custom Backbone 모델 등록하기

MMDetection에서는 기본 제공되는 다양한 백본 외에도, 직접 구현한 커스텀 백본을 등록하여 사용할 수 있습니다.

### 2-1. Custom Backbone 모델 코드 예시

```python
import torch.nn as nn
from ..builder import BACKBONES  # mmdetection의 백본 빌더 모듈

@BACKBONES.register_module()
class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        # 필요한 layer 정의
        # 예) self.conv = nn.Conv2d(...)

    def forward(self, x):
        # forward pass를 구현 (tuple 형태의 feature map을 return해야 함)
        # 예) feat = self.conv(x)
        return (feat,)  # tuple로 반환
```

이 파일은 `mmdetection/mmdet/models/backbones/mymodel.py` 경로에 저장합니다.

### 2-2. Configuration에 Custom Backbone 적용

```python
cfg.model.backbone = dict(
    type='MyModel',
    args='arg1'  # MyModel에서 필요로 하는 인자 값
)
```

이렇게 수정한 후, 기존과 동일하게 model을 build하고 학습하면 커스텀 백본이 적용된 object detection 모델을 사용할 수 있습니다.

---

## 결론

MMDetection은 미리 구성된 configuration 파일을 수정하는 것만으로도 최신 object detection 모델을 쉽게 구축하고 학습할 수 있도록 도와줍니다.

- **기본 사용법:** 설정 파일에서 데이터, 모델, 학습 파라미터 등을 수정하여 사용합니다.
- **Custom Backbone 등록:** 직접 만든 백본 모델을 등록해 활용할 수 있어, 보다 다양한 모델 구조를 실험할 수 있습니다.

---
# Detectron2로 Object Detection 모델 구축하기

Detectron2는 Facebook AI Research에서 개발한 PyTorch 기반 딥러닝 라이브러리로, object detection, segmentation 등 다양한 컴퓨터 비전 태스크를 손쉽게 수행할 수 있도록 지원합니다. 직접 구현하기에는 복잡한 object detection 모델을 미리 준비된 configuration 파일 하나만 수정하여 사용할 수 있으며, 커스터마이징도 용이합니다. 이번 포스팅에서는 Detectron2의 기본 사용법과 함께, 커스텀 데이터 augmentation, dataset 등록, 학습, 그리고 custom backbone 모델 등록 방법을 알아보겠습니다.

---

## 1. Detectron2 기본 설정

### 1-1. Import 및 Logger 설정

```python
import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, register_coco_instances
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader, build_detection_test_loader
```

- **설명:**  
    Detectron2의 로그를 설정하고, 모델, configuration, 데이터셋, 학습, 평가에 필요한 모듈들을 import합니다.

---

## 2. Configuration 파일 다루기

미리 준비된 configuration 파일을 불러와서 필요한 부분만 수정합니다.

```python
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))

# Dataset 설정
cfg.DATASETS.TRAIN = ("coco_trash_train",)
cfg.DATASETS.TEST = ("coco_trash_val",)

# 학습 설정
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.TEST.EVAL_PERIOD = 500
```

- **설명:**  
    모델 구성, 데이터셋, 학습 하이퍼파라미터를 수정합니다.

---

## 3. 데이터셋 등록 및 메타데이터 설정

COCO 형식의 annotation 파일과 이미지 디렉토리를 사용해 데이터셋을 등록합니다.

```python
# train dataset 등록
register_coco_instances('coco_trash_train', {}, '/home/data/train.json', '/home/data')
# validation dataset 등록
register_coco_instances('coco_trash_val', {}, '/home/data/val.json', '/home/data')

# 메타데이터 설정 (선택 사항)
classes = ["General Trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
MetadataCatalog.get('coco_trash_train').set(thing_classes=classes)
MetadataCatalog.get('coco_trash_val').set(thing_classes=classes)
```

- **설명:**  
    `register_coco_instances`를 통해 dataset의 이름, annotation 파일 경로, 이미지 폴더를 지정합니다.

---

## 4. Augmentation Mapper 정의

Detectron2는 MMDetection처럼 내장된 augmentation 기능이 제한적이므로, 데이터 전처리 및 augmentation은 custom mapper로 직접 정의합니다.

```python
import copy
import torch
from detectron2.data import detection_utils as utils

def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict
```

- **설명:**  
    이미지와 annotation에 적용할 augmentation을 정의하고, 이를 적용한 후 결과를 tensor 형식으로 변환하여 반환합니다.

---

## 5. Trainer 클래스 정의 및 학습

Custom trainer를 정의하여, 학습 데이터 로더에 augmentation mapper를 적용하고, evaluator를 설정합니다.

```python
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(cfg, mapper=MyMapper, sampler=sampler)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('./output_eval', exist_ok=True)
            output_folder = './output_eval'
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

- **설명:**  
    `MyTrainer`는 `DefaultTrainer`를 상속받아, custom train loader와 evaluator를 적용합니다.

---

## 6. Custom Backbone 모델 등록하기

원하는 모델이 없는 경우, 커스텀 백본을 등록할 수 있습니다.

```python
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
import torch.nn as nn

@BACKBONE_REGISTRY.register()
class MyBackbone(Backbone):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(MyBackbone, self).__init__()
        # 필요한 layer 정의 (예: CNN layers)
        self.conv = nn.Conv2d(input_shape.channels, 64, kernel_size=3, stride=1, padding=1)
        self._out_features = ["res5"]  # 출력 feature 이름 정의
        
    def forward(self, x):
        # forward pass 구현
        x = self.conv(x)
        return {"res5": x}  # dict 형태로 feature map 반환
        
    def output(self):
        # 출력 특성 정보 반환 (예: 채널 수, stride 등)
        return {"res5": ShapeSpec(channels=64, stride=1)}

# configuration 수정하여 custom backbone 사용
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
cfg.MODEL.BACKBONE.NAME = 'MyBackbone'
model = build_model(cfg)
```

- **설명:**  
    `MyBackbone` 클래스를 정의하고, Detectron2의 BACKBONE_REGISTRY에 등록한 후, configuration에서 custom backbone으로 지정합니다.

---

## 결론

Detectron2는 미리 구성된 configuration 파일을 수정하는 것만으로도 object detection 모델을 손쉽게 구축하고 학습할 수 있는 강력한 라이브러리입니다.

- **Configuration:** Model Zoo의 config 파일을 불러와 dataset, 학습 파라미터 등을 수정할 수 있습니다.
- **Augmentation:** Custom mapper를 정의하여 데이터 전처리 및 augmentation을 적용할 수 있습니다.
- **Custom Trainer:** 커스터마이즈된 trainer를 통해 학습 및 평가 프로세스를 관리합니다.
- **Custom Backbone:** 원하는 모델이 없을 경우, 커스텀 백본을 등록하여 사용할 수 있습니다.

---