---
{"dg-publish":true,"permalink":"/paper-review/r-cnn/","tags":["Paper"],"created":"2025-02-26T15:44:19.124+09:00","updated":"2025-01-08T19:51:26.291+09:00"}
---


논문 원본 링크: [https://arxiv.org/abs/1311.2524](https://arxiv.org/abs/1311.2524)

# 1. Introduction

## **1.1 문제**:

- PASCAL VOC 데이터셋에서 객체 탐지 성능이 지난 몇 년간 정체되었으며, 기존의 최고 성능 방법들은 복잡한 앙상블 시스템을 사용해 작은 개선만을 이루고 있습니다.

## **1.2 기존의 접근법과 한계**:

- SIFT와 HOG와 같은 특징은 단순한 블록 기반 히스토그램으로, 시각 인식을 위한 다단계, 계층적 특징 계산 과정이 부족합니다.
- CNN은 1990년대에 인기를 끌었으나, SVM의 등장으로 인해 관심을 잃었습니다. 그러나 2012년 Krizhevsky et al.의 연구로 ILSVRC에서 높은 정확도를 보여주며 다시 주목받게 되었습니다.

## **1.3 제안된 방법**:

- R-CNN(Regions with CNN features): 입력 이미지에서 약 2000개의 영역 제안을 생성하고, 각 제안에서 고정 길이의 특징 벡터를 CNN을 통해 추출하여, 카테고리별 선형 SVM으로 분류합니다.
- 간단한 아핀 변환 기법을 사용하여, 다양한 형태의 영역을 고정 크기의 CNN 입력으로 변환합니다.

## **1.4 결과**:

- R-CNN은 ILSVRC2013 탐지 데이터셋에서 OverFeat보다 우수한 성능(mAP 31.4% vs. 24.3%)을 달성하였습니다.
- fine-tuning을 통해, PASCAL VOC 2010 데이터셋에서 기존의 HOG 기반 DPM보다 훨씬 높은 mAP(54% vs. 33%)를 달성하였습니다.

# 2. Object detection with R-CNN

![image.png](/img/user/images/ViT images/image.png)

## 2.1 Module design

- **R-CNN의 구성 모듈:**
    - **모듈 1: 영역 제안 생성(Region proposals)**:
        
        범주에 독립적인 영역 제안을 생성하여, 탐지기가 사용할 수 있는 후보 탐지 세트를 정의합니다. **선택적 검색(selective search)**을 사용하여 이전 탐지 작업과 비교할 수 있게 합니다.
        
        > Selective Search?
        ⇒ Object가 있을 법한 부분만 Search하는 것
        
        1. input image에서 segmentation을 실시하여 가장 아래 이미지처럼 굉장히 많은 영역을 생성한다.
        2. 이후 알고리즘을 통해 유사도가 높은 영역끼리 합쳐가며 segmentation의 갯수를 줄여나간다.
        3. 결과적으로 box의 갯수도 줄어들게 만들어진다.
        > 
        > 
        > ![image.png](/img/user/images/ViT images/image 1.png)
        > 
    - **모듈 2: 특징 추출(Feature extraction)**:
        
        Krizhevsky et al. [25]의 CNN 모델을 사용하여 각 영역 제안에서 4096차원의 특징 벡터를 추출합니다. 이미지 데이터는 고정된 227 × 227 픽셀 크기로 CNN에 입력되기 전에 warp됩니다.
        
        ![image.png](/img/user/images/R-CNN images/image 2.png)
        
    - **모듈 3: 클래스별 선형 SVM(Class-specific linear SVMs)**:
        
        각 클래스별로 선형 SVM을 학습시켜, 영역 제안에서 추출한 특징 벡터를 바탕으로 객체를 분류합니다.
        
        ![image.png](/img/user/images/R-CNN images/image 3.png)
        
        > Bounding Box Regression?
        ⇒ 앞서 region proposal을 통해 얻은 bbox가 P, Ground Truth는 GT일 때 P를 GT에 가까워지도록 만드는 것이 Bounding Box Regression이다.
        > 
        > 
        > ![image.png](/img/user/images/R-CNN images/image 4.png)
        > 

> R-CNN과정 
1. Selective Search를 이용해 여러 영역으로 나눈 후 warp을 통해 227 x 227로 만든다.
2. wrap된 이미지를 CNN을 통해 Feature를 뽑아낸다.
3. 앞서 만든 Feature를 이용해 SVM을 통해 분류, Bounding Box Regression은 위치 정보를 예측한다.
> 

# 3. Conclusion

## **3.1 한계**

1. **계산 비용**: R-CNN은 각 영역 제안(region proposals)에 대해 CNN을 개별적으로 실행해야 하므로, 계산 비용이 높고, 특히 대형 데이터셋에 대해 실시간 처리가 어려울 수 있습니다. 
GPU를 사용하더라도 처리 시간이 상당하며, 이는 실시간 응용 프로그램에는 적합하지 않을 수 있습니다.
2. **복잡한 파이프라인**: R-CNN의 탐지 파이프라인은 여러 단계로 구성되어 있으며, 각 단계에서 별도의 학습 및 최적화가 필요합니다. 
이는 구현과 유지보수를 복잡하게 만들고, 학습에 상당한 시간과 자원을 요구할 수 있습니다.

## **3.2 의의**

1. **객체 탐지 분야의 혁신**: R-CNN은 객체 탐지 분야에서 CNN의 활용 가능성을 크게 확장하였으며, 특히 복잡한 장면에서의 **높은 정확도의 탐지를 가능**하게 했습니다. 
이는 이후의 연구들(예: Fast R-CNN, Faster R-CNN, Mask R-CNN 등)에 직접적인 영감을 주었습니다.
2. **후속 연구의 토대 마련**: R-CNN의 아이디어와 결과는 이후의 많은 객체 탐지 연구와 기술 개발의 토대가 되었습니다. R-CNN의 한계를 극복하기 위한 다양한 개선 연구들이 이어지면서, 객체 탐지 기술의 급속한 발전이 가능해졌습니다.