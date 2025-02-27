---
{"dg-publish":true,"permalink":"/naver-boost-camp/week-2/","created":"2025-02-26T15:44:18.957+09:00","updated":"2025-01-08T20:18:20.969+09:00"}
---

[[NaverBoostCamp/Week 3 학습 정리\|Week 3 학습 정리]]
# Day 1 (머신 러닝 라이프사이클)

## 전반적인 내용

1. 머신 러닝의 개념과 적용 사례, 학습의 종류를 설명한 후, 머신 러닝 라이프사이클의 각 단계에 대해 자세히 설명합니다. 각 단계는 계획, 데이터 준비, 모델 엔지니어링, 모델 평가, 모델 배포 및 모니터링과 유지 관리로 나뉩니다. 또한, 머신 러닝 프로젝트를 효과적으로 수행하기 위한 프로세스와 각 단계의 중요성에 대해 다룹니다.

## 1. 주요 내용

- 💡 **머신 러닝 정의**: 머신 러닝은 경험을 통해 성능을 향상시키는 알고리즘을 연구하는 학문으로 정의됩니다.
    
- 💡 **머신 러닝 적용 사례**: 이미지 분류, 스팸메일 필터링 등 다양한 실제 사례를 통해 설명됩니다.
    
- 💡 **학습의 종류**: 지도 학습, 비지도 학습, 강화 학습으로 나뉘며 각각의 특징과 예시가 제시됩니다.
    
- 💡 **머신 러닝 라이프사이클**: 모델의 개발부터 배포, 유지보수까지 포함하는 일련의 단계로 구성됩니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a067c19e-9990-4500-acfe-e60d777f1f09/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/20ef8f61-a1e3-4d09-aeaa-69cf26105cad/image.png)
    
- 💡 **계획하기**: ML 애플리케이션의 범위와 성공 지표를 정의하는 단계로, 타당성 보고서 작성이 포함됩니다.
    
- 💡 **데이터 준비**: 데이터 수집, 정리, 처리, 관리의 네 가지 파트로 구성된 데이터 준비 과정입니다.
    
- 💡 **모델 엔지니어링**: 모델 아키텍처를 구축하고 학습 및 검증 데이터를 통해 모델을 학습하는 과정입니다.
    
- 💡 **모델 평가**: 모델이 제품에 사용될 준비가 되었는지 확인하는 과정으로, 견고성 테스트와 성능 평가를 포함합니다.
    
- 💡 **모델 배포 및 모니터링**: 모델을 시스템에 배포하고 지속적으로 모니터링하며 개선하는 과정입니다.
    

# Day 2 (선형 대수)

## 전반적인 내용

1. 회귀 분석의 개념과 선형 회귀 방정식, 그리고 최소 제곱법(OLS) 등의 통계적 방법을 설명합니다. 또한, 다중 선형 회귀와 모델 평가 지표에 대해서도 다루며, 이어서 최근접 이웃 분류기의 기본 개념과 구현 방법, 그리고 이론적 한계점에 대해 논의합니다.
2. Linear Classifier의 정의와 매개변수적 접근 방식, 그리고 Softmax Classifier의 필요성과 이를 구현하기 위한 방법론을 다룹니다. 또한, 손실 함수와 최적화 방법에 대해 설명하며, 모델의 학습 과정에서 발생할 수 있는 다양한 문제점과 그 해결 방안에 대해 논의합니다.

## 1. 주요 내용

- 💡 **회귀 분석의 정의**: 과거 상태로 돌아가는 경향을 분석하는 방법으로 정의되며, 변수 간의 선형적 관계를 설명합니다.
    
- 💡 **선형 회귀**: 종속 변수와 독립 변수 간의 관계를 모델링하는 통계적 방법으로, 집값 예측 등 다양한 사례를 통해 설명됩니다.
    
- 💡 **선형 회귀 방정식**: $y = mx + b$ 형태의 방정식을 통해 독립 변수의 값으로 종속 변수를 예측합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/4b857936-bf61-439c-a4c4-a895c11e3eeb/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/61247b08-71c6-4130-9753-cb31dce30947/image.png)
    
- 💡 **모델 평가 지표**: 평균 절대 오차(MAE), 평균 제곱 오차(MSE), 제곱근 평균 제곱 오차(RMSE), 결정 계수(R²) 등의 평가 지표가 소개됩니다.
    
- 💡 **최근접 이웃 분류기(NN Classifier)**: 쿼리 데이터 포인트와 가장 가까운 학습 데이터 포인트의 라벨을 사용하여 예측합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/be1138ed-dcaf-4ff0-8462-0c75f4529a81/image.png)
    
- 💡 **k-NN 분류기**: 가장 가까운 k개의 이웃으로부터 과반수 득표를 통해 예측하는 방법입니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/cd72fd64-414e-4f02-b0bd-eeab64ef0e46/image.png)
    
- 💡 **최근접 이웃 분류기의 한계점**: 픽셀 거리의 정보 부족, 차원의 저주 등 다양한 문제점이 논의됩니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6142e4c8-dd2b-4ff4-8bb0-e67bb7d49f48/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/175ff653-2b2c-4c94-bc01-f1e463c3f994/image.png)
    

## 2. 주요 내용

- 💡 **Linear Classifier 정의**: Linear Classifier는 입력 데이터에 대해 가중치 합계를 계산하여 클래스를 예측하는 단순한 선형 모델입니다.
    
- 💡 **매개변수적 접근**: 모델은 가중치(W)와 편향(b)을 통해 입력 데이터를 특정 클래스로 분류하는 함수 f(x)로 정의됩니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/3444284f-57f1-491b-b6a4-17e997e2aeae/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/1cc4b33a-4e6e-4cdd-af25-bd93d81acc14/image.png)
    
- 💡 **Linear Classifier의 한계**: 점수의 크기가 무한대로 커질 수 있고, 해석이 어려운 문제가 있습니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/3f568b7d-4439-49d2-9983-d883285ae621/image.png)
    
- 💡 **Softmax Classifier**: 각 클래스에 속할 확률을 계산하여 결과를 확률 분포로 나타내며, Linear Classifier의 한계를 보완합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/db7ae2b2-a191-4cad-8477-39595e35f26d/image.png)
    
- 💡 **손실 함수**: 모델의 예측이 얼마나 정확한지를 정량화하는 지표로, 다양한 형태의 손실 함수가 소개됩니다.
    
- 💡 **최적화 기법**: Gradient Descent와 Stochastic Gradient Descent와 같은 방법을 통해 모델의 가중치(W)를 최적화하는 과정을 설명합니다.
    
- 💡 **Gradient Descent 문제점**: 비용 함수의 국부 최적점에 빠질 수 있는 문제와 대규모 데이터셋에서 발생하는 계산 속도 문제를 다룹니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/f624b2cc-cb71-4d94-87a5-c726cb3e5bf2/image.png)
    

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6643287b-891f-4b86-a973-4a511b5cb889/image.png)

# Day 3 (기초 신경망 이론)

## 전반적인 내용

1. Linear Model의 개념과 한계를 설명한 후, 신경망의 기초인 퍼셉트론(Perceptron)과 단층 및 다층 신경망의 구조를 설명합니다. 또한, 활성화 함수(Activation Functions)와 신경망의 학습 과정에서 발생하는 그래디언트 계산 방법에 대해 다룹니다.
2. 역전파 알고리즘을 설명하며, 이를 통해 신경망의 가중치를 효율적으로 업데이트하는 방법을 다룹니다. 또한, 연쇄 법칙(Chain Rule)을 통해 그래디언트 계산 과정을 설명하고, 로지스틱 회귀(Logistic Regression) 예제를 통해 실제 계산 과정을 보여줍니다.
3. 신경망 훈련에서 중요한 요소들을 소개합니다. 먼저, 다양한 활성화 함수의 특징과 단점에 대해 설명하고, 이어서 가중치 초기화 방법의 중요성을 다룹니다. 마지막으로, 학습률 조정 방법에 대해 설명하며, 각 방법이 훈련 과정에 어떤 영향을 미치는지에 대해 논의합니다.
4. 신경망 훈련에서 필수적인 데이터 전처리(Data Preprocessing)와 데이터 증강(Data Augmentation) 기법을 소개합니다. Zero-centering, PCA & Whitening, Data Augmentation의 필요성과 다양한 구현 방법들이 논의됩니다. 또한, 이미지 처리에서 흔히 사용되는 데이터 증강 기법들을 예시와 함께 설명합니다.

## 1. 주요 내용

- 💡 **Linear Model의 한계**: Linear Classifier는 복잡한 데이터를 처리하는 데 한계가 있으며, 비선형적인 데이터는 제대로 분류하지 못합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/9a17791a-4fc3-44ee-8eed-78e902b6d7d9/image.png)
    
- 💡 **Perceptron**: 신경망의 기본 단위로, 입력 값을 받아 가중치와 함께 계산한 후 활성화 함수를 통해 출력을 생성합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/072adfec-52f4-4f7f-840f-e3656c75f070/image.png)
    
- 💡 **단층 신경망**: 입력 층과 출력 층으로 구성된 단순한 구조로, 복잡한 패턴을 학습하는 데 한계가 있습니다.
    
- 💡 **다층 퍼셉트론(MLP)**: 여러 개의 은닉층을 가진 신경망 구조로, 비선형적인 문제를 해결하는 데 유리합니다.
    
- 💡 **활성화 함수**: 신경망에서 비선형성을 도입하기 위해 사용하는 함수로, 대표적으로 Sigmoid, tanh, ReLU가 있습니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/ff184590-4fa2-43fe-a6d9-e7cac1d5a868/image.png)
    
- 💡 **그래디언트 계산**: 신경망 학습을 위해 각 파라미터의 변화가 손실 함수에 미치는 영향을 계산하는 과정입니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e2e5b310-b679-46c0-8fba-9796ed6710f1/image.png)
    
- 💡 **학습의 예시**: Python 코드를 통해 2-layer MLP의 학습 과정을 구현한 예시가 제공됩니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/c5882ca6-c85a-444f-bbbd-eba21960cd44/image.png)
    

## 2. 주요 내용

- 💡 **역전파 알고리즘**: 신경망의 출력에서 입력 방향으로 그래디언트를 역으로 전파하여 각 가중치의 기울기를 계산하는 알고리즘입니다.
    
- 💡 **계산 그래프**: 계산 과정에서의 각 단계와 그래디언트를 시각적으로 표현한 그래프로, 역전파의 계산 과정을 쉽게 이해할 수 있습니다.
    
- 💡 **연쇄 법칙(Chain Rule)**: 복잡한 함수의 그래디언트를 계산할 때 사용하는 수학적 규칙으로, 역전파의 핵심 개념입니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6db65c07-a804-420f-a441-b0efd78cb3da/image.png)
    
- 💡 **로지스틱 회귀 예제**: 역전파를 활용한 로지스틱 회귀의 그래디언트 계산 과정을 단계별로 설명합니다.
    
- 💡 **그래디언트 흐름 패턴**: 역전파 과정에서 발생할 수 있는 다양한 패턴과 문제를 설명하며, 이를 효과적으로 처리하는 방법을 논의합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/07091a5b-7b24-4b34-a58a-d6c5df1637a8/image.png)
    
- 💡 **그래디언트 구현 예제**: Python 코드를 통해 역전파 알고리즘을 구현하는 예시를 제공합니다.
    

## 3. 주요 내용

- 💡 **활성화 함수(Activation Functions)**: 신경망의 출력에 비선형성을 도입하는 함수로, Sigmoid, Tanh, ReLU 등의 다양한 함수들이 소개됩니다.
    
- 💡 **Sigmoid 함수의 단점**: Vanishing Gradient 문제와 zero-centered 되지 않은 출력으로 인한 학습의 비효율성을 설명합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/2e9b628c-4e54-4985-bf61-84e12a5df9e6/image.png)
    
- 💡 **Tanh 함수**: 출력 값이 [-1, 1]의 범위를 가지며, Zero-centered된 출력이 특징이지만 여전히 Vanishing Gradient 문제를 가지고 있습니다.
    
- 💡 **ReLU와 그 변형들**: ReLU는 빠른 학습 속도를 제공하지만 Dead ReLU 문제를 발생시킬 수 있으며, Leaky ReLU와 ELU 같은 변형들이 이에 대한 해결책으로 제시됩니다.
    
- 💡 **가중치 초기화(Weight Initialization)**: Small Gaussian Random, Large Gaussian Random, Xavier Initialization 등의 초기화 방법들이 신경망의 학습에 미치는 영향을 설명합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/88bf84aa-6a08-4fef-8596-3f6a9860d784/image.png)
    
- 💡 **학습률 조정(Learning Rate Scheduling)**: 학습률을 적절하게 설정하는 방법과, 학습률을 점차 감소시키는 다양한 기법들이 소개됩니다.
    
- 💡 **Learning Rate Decay**: 학습이 진행됨에 따라 학습률을 점차 감소시키는 방법으로, Step Decay, Cosine, Linear, Inverse Sqrt 등 다양한 기법들이 설명됩니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/5fdbe6c6-83e9-464c-8daf-0a2bb6ee610d/image.png)
    

## 4. 주요 내용

- 💡 **데이터 전처리(Data Preprocessing)**: Zero-centering, 정규화(Normalization), PCA & Whitening을 통해 데이터를 신경망에 적합한 형태로 전처리하는 과정이 설명됩니다.
    
- 💡 **Zero-centering & Normalization**: 입력 데이터를 평균이 0이 되도록 조정하고, 정규화를 통해 데이터의 분포를 균일하게 맞춥니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/b52d9429-f421-465c-b2ad-2782d194c466/image.png)
    
- 💡 **PCA & Whitening**: 데이터의 분산을 최대화하는 축을 찾아 데이터를 정렬하고, 각 축의 중요도를 균등하게 맞추는 과정입니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/32c2409a-4b82-4b6b-83ae-a5a96629a0fb/image.png)
    
- 💡 **데이터 증강(Data Augmentation)**: 기존 데이터에 노이즈를 추가하거나 변형하여 데이터 양을 늘리는 기법입니다.
    
- 💡 **Horizontal Flips**: 이미지의 수평 반전을 통해 데이터의 다양성을 증가시킵니다.
    
- 💡 **Random Crops**: 이미지의 일부만을 잘라내어 학습시키는 기법으로, 객체의 부분적 정보로도 학습할 수 있도록 도와줍니다.
    
- 💡 **Scaling**: 다양한 크기의 이미지를 처리할 수 있도록 데이터를 확장 또는 축소하는 기법입니다.
    
- 💡 **Color Jitter**: 색상, 채도, 명도 등을 임의로 조정하여 다양한 조명 조건에서 모델이 견고하게 작동하도록 합니다.
    
- 💡 **Data Augmentation in Practice**: 문제와 데이터의 영역에 따라 다양한 데이터 증강 기법들을 실습에 적용하는 방법을 논의합니다.
    

# Day 4 (Transformer)

## 전반적인 내용

1. RNN의 기울기 소실/폭발 문제와 이를 해결하기 위한 LSTM과 GRU를 소개합니다. 또한, Seq2seq 모델의 개념과 구조, 그리고 실제 구현 방법을 다루며, 특히 기계 번역(Machine Translation)과 같은 NLP(Natural Language Processing) 작업에 적용되는 방식을 설명합니다.
2. RNN 모델의 한계와 이를 해결하기 위한 Attention 메커니즘을 설명하고, Transformer 모델의 주요 아이디어와 구조를 다룹니다. 특히, Attention의 역할과 중요성, 다양한 변형 기법들에 대해 구체적으로 설명하며, 이를 통한 성능 향상 방법을 제시합니다.
3. Transformer 모델의 학습 과정, 토큰 집계 방법, Encoder-Decoder 구조를 설명하고, 이어서 BERT와 Vision Transformer의 원리와 활용 방법을 다룹니다. 특히, Self-Attention 메커니즘과 Multi-head Attention의 개념을 중심으로 설명하며, 모델 학습의 구체적인 방법과 실험 결과를 포함합니다.

## 1. 주요 내용

- 💡 **RNN의 정의와 문제점**: RNN은 시계열 데이터 처리를 위한 모델이지만, 기울기 소실(Vanishing Gradient)과 기울기 폭발(Exploding Gradient) 문제가 발생합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/85f367a7-faed-4254-93fd-a5e7af566d0f/image.png)
    
- 💡 **LSTM(Long Short-Term Memory)**: LSTM은 RNN의 기울기 소실 문제를 완화하기 위해 설계된 모델로, 셀 상태(cell state)와 게이트(gate)를 도입하여 장기 의존성을 모델링할 수 있습니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a870c591-d68c-4d4d-8d89-996c6aafb7cb/image.png)
    
- 💡 **GRU(Gated Recurrent Units)**: LSTM의 변형 모델로, 파라미터 수가 적고, LSTM과 유사한 성능을 제공합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/f22425f2-5487-44f0-9d14-e89d7942f401/image.png)
    
- 💡 **Seq2seq 모델**: Encoder-Decoder 구조를 가지며, 입력 시퀀스를 인코딩한 후, 이를 기반으로 출력 시퀀스를 생성하는 모델입니다.
    
- 💡 **Machine Translation 문제**: Seq2seq 모델을 기계 번역에 적용하는 방법과, 입력 시퀀스와 출력 시퀀스의 길이가 다를 때 발생하는 문제를 해결하는 방법이 설명됩니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/7aad9686-6823-454a-a63a-e61f93c25a70/image.png)
    
- 💡 **Encoder-Decoder 구조**: 입력 시퀀스를 인코딩하여 단일 벡터로 표현한 후, 이를 기반으로 출력 시퀀스를 생성합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/b087cde1-7bc2-4962-964a-ff40d1d37433/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/d57afd58-f7fd-4d99-b56f-a57d7833958e/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6f78653f-e514-4728-953e-fae49000c210/image.png)
    
- 💡 **Auto-Regressive Generation**: 이전 출력값을 다음 단계의 입력으로 사용하는 방식으로, 시퀀스 생성 과정을 설명합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/d1f62da1-b13a-45bf-a152-03882c4bb993/image.png)
    
- 💡 **Teacher Forcing**: 학습 단계에서 실제 출력값을 사용하여 모델을 학습시키는 기법입니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a4301016-4bef-43cd-9f9e-2d20cba11d59/image.png)
    
- 💡 **Seq2seq 모델 구현**: Python의 PyTorch 라이브러리를 이용하여 Seq2seq 모델을 구현하는 방법이 소개됩니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e531edb5-412c-4ec7-8495-7f9b371267ff/image.png)
    

## 2. 주요 내용

- 💡 **Attention 메커니즘의 필요성**: RNN 기반 모델에서는 긴 시퀀스 처리 시 정보 손실이 발생할 수 있으며, 이를 해결하기 위해 Attention 메커니즘이 도입됩니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a8aa074c-9326-4eea-b93c-ecfd7b9ee29d/image.png)
    
- 💡 **Attention 아이디어**: Decoder는 모든 입력 시점의 hidden state를 고려하며, 관련성이 높은 입력 토큰에 집중합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/c0b83da1-e0fa-4b22-99fa-84de2e6701d1/image.png)
    
- 💡 **Dot-Product Attention**: Query와 Key 간의 내적(dot-product)을 통해 유사성을 계산하고, 이 값으로 Value의 가중치를 결정하여 최종 Attention 값을 계산합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6f6e5b74-3f91-4faa-94cb-2228f8a38daa/image.png)
    
- 💡 **Attention 메커니즘의 요약**: Query, Key, Value의 역할과 Attention 값 계산 방법이 요약됩니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/05227df8-7f4a-4527-af1a-0271d3448e21/image.png)
    
- 💡 **다양한 Attention 방법**: Dot-product, 학습 가능한 가중치 적용, Concatenation 등 다양한 유사성 계산 방법이 설명됩니다.
    
- 💡 **Machine Translation에서의 Attention**: Attention 메커니즘을 활용한 기계 번역의 인코딩 및 디코딩 과정이 설명됩니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/d0860597-e865-43b8-bdb0-16a4dd1552b5/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/55ee49fb-371b-45fc-873b-ac87fab4ac8a/image.png)
    
- 💡 **Transformer 모델의 주요 아이디어**: Self-Attention을 통해 각 요소가 자신을 포함한 시퀀스의 다른 요소들과의 관계를 학습하여, 더 나은 표현을 생성합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/b8c477d4-5e12-478e-8597-46ce2879ab00/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/38b666ab-b058-499c-bef4-961891832a83/image.png)
    
- 💡 **Self-Attention의 계산 과정**: Query, Key, Value를 생성하고, 이를 바탕으로 가중치를 계산하여 최종 Attention 값을 생성하는 과정이 설명됩니다.
    
- 💡 **다음 학습 주제 예고**: Transformer 모델의 구조와 다양한 응용 사례를 탐구할 예정입니다.
    

## 3. 주요 내용

- 💡 **Token Aggregation 방법**: 평균 풀링(Average Pooling)과 Classification Token을 사용한 토큰 집계 방법을 설명합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/7f0202e0-7085-45ed-a063-757a304c13d6/image.png)
    
- 💡 **Transformer 학습 과정**: 입력 임베딩에서 시작해 Multi-head Self-Attention과 Feed-forward Layer를 통해 입력 데이터를 처리하는 과정을 다룹니다.
    
- 💡 **Positional Encoding**: Transformer 모델에서 순서 정보가 없는 토큰에 위치 정보를 부여하는 방법을 설명합니다.
    
- 💡 **Decoder 구조**: Masked Multi-head Self-Attention과 Encoder-Decoder Attention을 통해 출력 시퀀스를 생성하는 과정이 설명됩니다.
    
- 💡 **BERT 모델**: BERT의 구조와 Masked Language Modeling, Next Sentence Prediction 과제에 대해 설명합니다.
    
- 💡 **Vision Transformer(ViT)**: Transformer 모델을 이미지 데이터에 적용하는 방법과 그 한계에 대해 논의합니다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e838b570-d03c-4aef-8a14-0b5e60a6ca7b/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/d28cd2c7-7870-490c-91af-639429b66ec9/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/ed6b1feb-88c3-4b9a-a140-b093b67c7ba9/image.png)
    
- 💡 **ViT의 실험 결과**: ViT가 매우 큰 데이터셋에서 잘 작동하는 이유와 학습 비용이 논의됩니다.
    

# Day 5

월욜에…