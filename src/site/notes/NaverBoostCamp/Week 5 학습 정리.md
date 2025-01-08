---
{"dg-publish":true,"permalink":"/naver-boost-camp/week-5/","created":"2025-01-07T18:04:05.874+09:00","updated":"2025-01-08T20:19:04.874+09:00"}
---

[[NaverBoostCamp/Week 6 학습 정리\|Week 6 학습 정리]]
> [!NOTE]
> > **1. Multimodal 1, 2
> > 2. Generative Models
> > 3. 3D Understanding
> > 4. 3D Human

# **1. Multimodal 1, 2**

Multi-modal에는 크게 3가지 방법이 있다.

![Pasted image 20250107180902.png](/img/user/images/Pasted%20image%2020250107180902.png)

## 1.1 Multimodal Challenge

1. Modality 간에 서로 표현 방법이 다르다!
2. Modality 간에 표현하는 정보량이 다르다!
3. 두 Modality 중에 보통 하나의 Modality에 편향된다!

## 1.2 Multi-modal alignment

⇒ 서로 다른 Modality 간의 정보를 조화롭게 연결하고 해석할 수 있도록 하는 과정

### 1.2.1 Matching

⇒ 서로 다른 모달리티(데이터 유형) 간의 관련성을 찾아내는 작업

- Joint embedding: 서로 다른 Modality를 같은 feature vector space로 옮겨 매칭이 가능하도록 한다.

### 1.2.2 Translating

⇒ 한 Modality의 표현을 다른 언어 또는 모달리티로 변환하는 과정

### 1.2.3 Referencing

⇒ 여러 Modality가 있는 환경에서는 서로 다른 데이터 유형이 어떻게 상호 연결되는지 명확히 하고, 그 관련성을 밝히는 것이 중요

## 1.2 예시 모델

### 1.2.1 Matching : CLIP

⇒ 텍스트와 이미지를 동시에 처리하여 상호 간의 연관성을 학습하는 Multi modal 모델

![CLIPimages.png](/img/user/images/CLIPimages.png)

- **아이디어**: 이미지와 텍스트를 모두 벡터화하여 동일한 임베딩 공간으로 변환한 후, 이미지와 그에 해당하는 텍스트 설명이 얼마나 가까운지를 평가 ⇒ 이를 통해 이미지와 텍스트 간의 관계를 파악하고 같은 의미를 가진 텍스트와 이미지를 서로 가깝게 배치하려고 한다.
- **Contrastive Learning(대조학습)**: 아래와 같은 방식으로 진행
    - **정답 쌍(True Pair)**: 이미지와 그에 대응하는 올바른 텍스트 설명을 가깝게 배치.
    - **잘못된 쌍(False Pair)**: 이미지와 맞지 않는 텍스트 설명을 멀리 배치.
- **구조**: 두 개의 인코더로 구성
    - **이미지 인코더**: 이미지를 처리하여 벡터 표현으로 변환합니다. 주로 **ResNet**이나 **Vision Transformer(ViT)**와 같은 네트워크를 사용합니다.
    - **텍스트 인코더**: 텍스트 데이터를 벡터 표현으로 변환합니다. 주로 **Transformer** 구조를 사용합니다.
- CLIP의 장점
    1. **Zero-shot 학습**: CLIP은 특정 태스크에 대해 별도의 학습 없이도 바로 적용될 수 있습니다. 예를 들어, CLIP은 이미지 분류 태스크에서 사전에 학습된 클래스 없이도 텍스트 설명만으로 새로운 카테고리를 분류할 수 있습니다.
    2. **다양한 태스크에서의 유연성**: CLIP은 이미지 분류, 객체 인식, 이미지 검색, 이미지 캡셔닝 등 다양한 태스크에 적용될 수 있습니다.
    3. **일반화 능력**: 대규모 데이터로 학습한 덕분에 CLIP은 훈련된 데이터셋 이외의 새로운 데이터셋에도 비교적 잘 일반화되는 성능을 보입니다.

### 1.2.2 Translating:

- Text-to-Image generation: DALL-E2
- Sound-to-Image synthesis
- Speech-to-Face synthesis: Speech2Face
- Image-to-Speech synthesis

### 1.2.3-1 Referencing: Show, Attend and Tell

⇒ 이미지에서 중요한 부분에 집중(attention)하여 자연어 설명(캡션)을 생성하는 방법을 제안한 딥러닝 기반의 모델

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6b54a1de-a163-4847-b5d2-793bd14147a1/image.png)

- **이미지 처리 (인코딩)**: 먼저, 모델은 CNN(예: **ResNet** 또는 **VGG**)을 사용하여 이미지의 특징 맵(feature map)을 추출합니다. 이미지의 각 부분에 대한 정보를 벡터 형태로 나타낸 후, 디코더에 전달하여 텍스트를 생성할 준비를 합니다.
    
- **Attention 메커니즘**: 디코더는 캡션을 생성하는 각 단계에서 이미지의 특정 부분에 집중하게 됩니다. 모델은 각 시점(time step)마다 주목할 이미지의 영역을 선택적으로 결정하고, 그 부분의 시각적 정보를 사용하여 다음에 생성할 단어에 필요한 정보를 추출합니다. 이 과정은 **Soft Attention** 방식으로 구현되며, 이를 통해 캡션 생성 과정에서 이미지의 모든 부분을 동시에 처리하지 않고, 가장 관련된 영역에만 집중할 수 있습니다.
    
    > soft attention이란? ⇒ Soft Attention은 입력의 모든 부분에 대해 가중치를 계산한 후, 가중합(weighted sum)을 구해 출력으로 사용합니다. 이때 각 입력에 부여되는 가중치는 0과 1 사이의 값이며, 모든 가중치의 합은 1이 됩니다. 즉, 입력의 일부 요소는 더 강조(큰 가중치)되고, 덜 중요한 요소는 덜 강조(작은 가중치)되지만, 모든 입력 요소가 어느 정도의 영향을 미칩니다.
    > 
    > ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/76246bb5-94c6-4ea8-b4b0-7c7f90ce20b2/image.png)
    
- **텍스트 생성 (디코딩)**: Attention 메커니즘을 통해 선택된 이미지의 특징을 기반으로, LSTM 네트워크는 순차적으로 단어를 생성합니다. 처음에는 시작 토큰(예: `<start>`)을 입력으로 받아 첫 번째 단어를 생성하고, 그 후 이전에 생성된 단어와 Attention을 통해 얻은 시각적 특징을 사용해 다음 단어를 예측하는 방식입니다. 이 과정을 반복하여 문장을 생성하다가 종료 토큰(예: `<end>`)이 나오면 캡션 생성이 완료됩니다.
    

### 1.2.3-2 Referencing: Flamingo

⇒ **비전(시각적 데이터)**과 **언어(텍스트)**를 동시에 처리하여 이미지나 비디오에 대한 자연어 설명을 생성하거나, 주어진 텍스트와 관련된 이미지를 이해하는 등의 작업을 수행할 수 있는 강력한 모델

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/4a736336-8244-416b-919f-f18c1f7d83ee/image.png)

- AGI: **범용 인공지능**을 의미하며, 인간과 같은 수준의 전반적인 지능을 가진 인공지능
    
    ![사람](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/2026a90d-1216-4452-a2fc-23c1b91b80b0/image.png)
    
    사람
    
    ![AGI](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6653e938-cf81-4202-9561-23484cba1e9c/image.png)
    
    AGI
    

## 1.3 LLaVA (Large Language and Vision Assistant)

⇒ 시각적 추론 모델로, 비전과 언어를 결합하여 텍스트와 이미지를 동시에 이해하는 모델입니다. 이는 복잡한 시각적 데이터를 해석하고 이를 언어적으로 설명할 수 있는 기능을 제공

### 1.3.1 Feature alignment (Projection)

⇒ Feature alignment)을 위해 **선형 레이어**를 사용하여, 시각적 입력을 언어 모델과 결합합니다. 즉, 시각 모달리티와 언어 모달리티 간의 연결을 제공합니다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a542b897-7a0a-462d-8302-86eb281bda31/image.png)

### **1.3.2 Visual instruction tuning**

- **MSCOCO**와 같은 데이터셋을 이용해 GPT 모델에서 이미지를 기반으로 다양한 응답을 생성합니다. 여기에는 대화형 응답, 상세 설명, 복잡한 추론이 포함됩니다.
- **Feature alignment**를 위한 사전 학습 단계와 시각 인코더 및 언어 모델을 고정(freeze)하고 **프로젝션 레이어**를 학습하는 단계로 나누어져 있습니다.
    - Step-1 : Pre-training for feature alignment ⇒ vision encoder와 LLM을 고정하고 projection layer만 훈련합니다.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/f6b1ab6e-5707-4468-b705-ada73f510f6b/image.png)
        
    - Step-2 : Fine-tuning end-to-end ⇒ vision encode 고정, projection layer 및 LLM 학습
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/72fbf6b4-ddd4-4a6a-8e41-23af0f254539/image.png)
        

## **1.3. InstructBLIP**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/f2bdc6c3-d205-43b0-bed8-4ac2aebffa66/image.png)

### **1.3.1 InstructBLIP 개요**

- **InstructBLIP**은 다양한 **비주얼 질문 응답(VQA)** 및 **이미지 캡셔닝** 작업에 훈련된 모델입니다. 이 모델은 지시(instructions)에 맞춰 다양한 비전-언어 태스크를 처리할 수 있도록 설계되었습니다.

### **1.3.2 Feature alignment (Q-Former)**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/60508dcb-4a32-4ebb-8cc1-507dea4f6147/image.png)

- **Q-Former**는 InstructBLIP에서 사용되는 모듈로, 시각적 특징을 추출하는 방법입니다. 이 모듈은 **Instruction-aware** 방식으로, 즉, 지시에 따라 시각적 특징을 더 효과적으로 추출하는 과정을 포함합니다.
- 학습 가능한 쿼리(learnable queries)를 통해 시각적 특징과 텍스트 지시를 결합하여 처리합니다.

### **1.3.3 InstructBLIP의 변형**

- **InstructBLIP**는 다양한 모달리티 간의 조정을 목표로 합니다. 이를 통해 이미지와 텍스트뿐 아니라 더 많은 모달리티를 처리할 수 있도록 확장된 버전으로 발전할 수 있습니다.
- **X-InstructBLIP**는 교차 모달리티 간의 상호 작용을 더욱 강화한 변형 모델입니다. 이는 LLM과 함께 다양한 모달리티 간의 추론을 할 수 있는 프레임워크를 제공합니다.

# **2. Generative Models**

⇒ 실제 데이터와 비슷한 가짜 데이터를 만드는 것 → Training 데이터를 이용해 새로운 샘플을 만들고 이를 실제 데이터와의 차이를 계산해 가깝게 만든다.

## 2.1 Autoregressive Model

- **Chain Rule**: 이미지의 가능도를 1차원 분포로 나누어 설명하는 규칙. 이 규칙을 사용하여 복잡한 이미지 데이터를 각 픽셀의 분포로 분해할 수 있음.
    
- **PixelRNN**: 이미지의 픽셀을 한 방향(좌상단에서 시작)으로 생성하며, 이전 픽셀 값에 따라 다음 픽셀 값을 예측하는 방식.
    
    - RNN(LSTM)을 사용해 이전 픽셀 값의 의존성을 모델링함.
    - 훈련 시에는 이미지의 Likelihood를 최대화하는 방식으로 학습함. ⇒ 인공 데이터가 실제 데이터 처럼
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/ba3a9e9e-0388-4538-b377-d9a4b16c4ee9/image.png)
    

## 2.3 VAE, Variational Autoencoder

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/09cc2c1e-24ac-4b9a-bac3-d5f1b3ef5da6/image.png)

- **오토인코더 구조**: 데이터 압축 및 복원을 통해 데이터를 잠재 공간(Latent Space)으로 표현.
- **VAE와 오토인코더의 차이**:
    - 오토인코더는 입력을 고정된 벡터로 매핑하는 반면, VAE는 입력을 확률 분포로 매핑하여 더 유연한 표현을 가능하게 함.
- **잠재 공간**: 학습된 모델은 잠재 변수 z를 통해 데이터를 생성하는 과정을 학습.
- **문제점**: 오토인코더는 입력 데이터를 압축한 후 재생성하는 과정에서 단순 복사처럼 작동할 수 있으며, 일반화된 새로운 샘플을 생성하기 어려움.

## 2.4 DDPM, Denoising Diffusion Probabilistic Models

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e0506ec9-6440-413c-87a0-57679ff503bb/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6b51dd48-34e3-4f81-ac2e-e94f22a5dafc/image.png)

- **잡음 추가 및 제거**: 이미지에 점진적으로 잡음을 추가하고, 이를 반대로 제거하는 과정을 학습하여 이미지를 생성.
- **마르코프 과정**: 연속적인 과정에서 각 단계의 상태는 이전 상태에만 의존하는 특성을 가지고 있음.
- **변분 하한**: VAE와 유사하게 학습할 수 있는 손실 함수를 도출하기 위해 변분 하한을 사용함.

## 2.5 Latent Diffusion Models, Stable Diffusion

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/29f4cc58-0f10-4b78-b130-38c46ad0f023/image.png)

- **기존 모델(DDPM)의 한계**:
    - 픽셀 공간에서 작업하므로 많은 계산 자원이 필요하고, 최적화에 많은 시간이 소요됨.
- **잠재 공간에서의 확산 모델 학습**: 사전 훈련된 오토인코더를 사용하여 이미지의 잠재 표현을 생성하고, 이 잠재 공간에서 확산 모델을 학습하여 효율적으로 고해상도의 이미지를 생성.
- **크로스 어텐션 레이어**: 텍스트 또는 바운딩 박스와 같은 조건 입력을 처리하기 위해 사용됨.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/82c90647-550e-4d6c-8b50-ef1aedf981b5/image.png)

## 2.6 Condition in the Diffusion Models

- **ControlNet**: 확산 모델을 텍스트나 다른 입력 조건에 따라 제어할 수 있도록 설계된 구조. ControlNet은 기존의 조건 입력(예: Canny Edge, 사람의 자세 등)을 사용하여 Stable Diffusion을 더욱 세밀하게 제어할 수 있도록 함.
    - **ControlNet 아키텍처**: 학습된 신경망 블록에 추가적인 조건을 입력으로 받아 제어 기능을 확장함.
    - 모델의 일부 파라미터를 고정하고, 이를 클론하여 외부 조건 입력을 받아들이는 방식으로 새로운 조건을 처리함.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/be00b6c6-e7fb-4cb6-81cf-39bcf2e6cfcc/image.png)

- **LoRA (Low-Rank Adaptation)**:
    
    - **LoRA 개념**: 모델의 파라미터를 직접 업데이트하지 않고, 파라미터의 저차원 근사치(랭크 분해 행렬 A와 B)를 학습하여 메모리 사용량을 줄이고 효율적으로 미세 조정을 가능하게 함.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/9cb0e05f-962a-4fa4-9481-8694e1bae717/image.png)
    

## 2.7 Image Editing

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/95b4f345-7718-4918-8542-4316ec30b3ca/image.png)

- **Prompt-to-Prompt 이미지 편집**:
    - 텍스트 기반 이미지 편집 기술로, 원본 이미지의 구성과 구조를 보존하면서 텍스트를 사용하여 이미지를 제어함.
        
    - **주요 방법**:
        
        - 크로스 어텐션 맵을 이용하여 각 텍스트 토큰의 spatial attention maps를 생성.
            
        - 생성된 이미지의 spatial 레이아웃과 기하학적 구조를 제어하기 위해 원본 이미지의 어텐션 맵을 사용.
            
        - 원본 이미지와 동일한 부분의 어텐션 맵을 유지하고, 변경된 부분의 맵만 수정함으로써 세밀한 이미지 편집을 가능하게 함.
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/605e4e8a-fc58-4c59-b67e-e1f42f619c33/image.png)
            
    - **활용 사례**:
        
        - 특정 텍스트와 이미지의 관계를 재구성하여 새로운 이미지를 생성.
        - Bear라는 단어가 특정 이미지의 곰 부분과 연결되어 있어 이를 조작할 수 있는 예시 제시.
- **InstructPix2Pix**:
    - 이미지 편집을 지시문 기반으로 수행하는 모델로, 텍스트로 주어진 지시를 따라 이미지를 수정함.
    - **주요 특징**:
        - 기존 이미지에 간단한 텍스트 명령어를 입력하면, 이에 맞는 적절한 편집을 수행함.
        - 사용자는 이미지의 전후 설명을 완벽하게 제공할 필요 없이 간단한 지시로 편집 가능.
    - **방법**:
        - GPT-3를 미세 조정하여 사람의 설명에 맞는 텍스트 명령어를 생성하고, 그에 따라 이미지 편집을 학습함.
        - 생성된 텍스트 명령어와 편집 전후 이미지를 쌍으로 묶어 데이터셋을 생성하고, 이를 통해 이미지 편집 모델을 훈련.

### 2.2 깊이 생성 (Depth Generation)

- **Marigold**:
    
    - 깊이 정보를 생성하는 데 사용되는 모델로, 3D 정보나 장면의 깊이를 이해하고 예측하는데 활용됨.
    - 생성 모델이 단순 이미지 생성뿐만 아니라 깊이 정보 생성에도 활용될 수 있다는 점을 보여줌.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/3f244c5b-c746-4a7c-a402-dd9e082f6aab/image.png)
    

# **3. 3D Understanding**

## 3.1 3D가 중요한 이유 (Why is 3D important?)

- 우리는 3D 공간에서 살아가며, 인공지능(AI) 에이전트는 이 3D 공간에서 작동함.
- AI가 현실 세계에서 작업하려면 3D 공간을 이해하는 것이 필수적임.
- **3D의 주요 활용 분야**:
    - **증강 현실(AR) 및 가상 현실(VR)**: 3D 공간에서의 몰입 경험을 제공.
    - **로봇 및 3D 프린팅**: 3D 공간에서 로봇이 물리적으로 상호작용하거나 3D 객체를 실제로 프린팅하는 데 활용.
    - **의료 분야**: 3D를 활용한 단백질 서열 분석 및 합성, 신경 영상(neuroimaging) 등에서 중요한 역할을 함.

## 3.2 3D를 관찰하는 방식 (The Way We Observe 3D)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/a163c224-f240-413c-801a-89d3b70b141d/image.png)

- **이미지와 3D 세계의 관계**: 이미지는 3D 세계가 2D 공간에 투영된 결과임.
    
- **카메라의 역할**: 카메라는 3D 장면을 2D 이미지 평면에 투영하는 장치로서 작동함.
    
- **두 개의 뷰를 사용한 기하학적 구조**: 3D 구조는 두 개 이상의 시점을 사용하여 복원할 수 있음.
    
    - **Structure from Motion (SfM)**: 여러 이미지에서 대응되는 점들을 찾아 카메라의 움직임과 3D 구조를 추정하는 방법. COLMAP과 같은 툴이 대표적임.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/b06538af-d130-48dd-885e-605d6ed7b1f2/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/bb965992-280a-499b-979a-222f4b135171/image.png)
    

## 3.3 3D 데이터 표현 방식 (3D Data Representation)

- **2D 이미지 표현**: 각 픽셀의 RGB 값을 사용하여 2D 배열 구조로 이미지를 표현.
    
- **3D 데이터 표현 방식**:
    
    - **멀티뷰 이미지(Multi-view images)**: 다양한 각도에서 촬영한 2D 이미지로 3D를 유추.
    - **암묵적 형태(Implicit shape)**: 수학적 표현으로 형태를 나타냄.
    - **부피 기반 표현(Volumetric, voxel)**: 3D 공간을 작은 셀(보통 큐브)로 나누어 표현.
    - **점 구름(Point cloud)**: LiDAR 스캔과 같은 방식으로 3D 공간 내의 점들의 집합으로 표현.
    - **메쉬(Mesh)**: 점, 선, 면을 사용해 3D 객체의 표면을 표현. 그래프 CNN과 같은 네트워크에서 활용됨.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/32f989d5-35fa-4210-8da3-a0bd419f84bf/image.png)
    
    ## 3.4 3D 작업 (3D Tasks)
    
    ### 3.4.1 3D 인식 (3D Recognition)
    
    - **3D 객체 인식**: 2D 이미지에서 객체를 인식하는 것처럼, 3D 공간에서도 객체를 인식함.
        - **3D 객체 탐지**: 자율 주행 차량 등의 응용 프로그램에서 3D 객체의 위치를 이미지나 3D 공간에서 감지.
        - **3D Semantic segmentation**: 신경 영상(Neuroimaging)과 같은 데이터에서 3D 공간을 의미론적으로 분할.
    
    ### 3.4.2 3D 재구성 (3D Reconstruction)
    
    - **NeRF (Neural Radiance Fields)**:
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e8b244ff-9ac2-44b4-90de-18eec40146f8/image.png)
        
        - **복잡한 장면의 새로운 뷰 합성**: NeRF는 소수의 입력 뷰로부터 복잡한 장면의 새로운 시점을 합성함.
        - **3D 객체나 장면을 신경망에 메모리화**: NeRF는 신경망을 통해 장면을 저장하고, 학습한 장면으로부터 새로운 이미지를 생성할 수 있음.
    - **볼륨 렌더링(Volume Rendering)**:
        
        - 3D 부피 데이터를 사용하여 2D 이미지를 계산하는 과정.
            
        - NeRF는 3D 데이터를 렌더링하여 새로운 2D 이미지를 생성함.
            
            ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e086b3e4-d17f-4a31-b7a6-d5d03c626ca2/image.png)
            
    - **3D 가우시안 스플래팅(3D Gaussian Splatting)**:
        
        - **실시간 방사선 필드 렌더링**: 3D 장면을 실시간으로 렌더링하는 데 사용되는 방법.
        - **장점**: 장면 최적화 및 새로운 시점 합성을 가속화함.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/4f69a282-f1f3-4171-880e-48ff8591f87e/image.png)
        
    
    ### 3.4.3 3D 생성 (3D Generation)
    
    - **Mesh R-CNN**:
        
        - Mask R-CNN에 3D 메쉬 생성을 위한 "3D 브랜치"가 추가된 모델.
        - 2D 객체 탐지 및 분할 작업뿐만 아니라 객체의 3D 메쉬를 출력할 수 있음.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/ec0d16ce-5b22-437d-8c36-e891a805483b/image.png)
        
    - **DreamFusion**:
        
        - **텍스트에서 3D 생성**: 사전 훈련된 2D 텍스트-이미지 확산 모델을 사용해 텍스트를 기반으로 3D 모델을 생성.
        - **SDS(Score Distillation Sampling) 손실**: 확산 모델의 노이즈 예측을 사용하여 3D 모델을 업데이트하는 방식. U-Net의 야코비안 계산이 비싸기 때문에 이를 생략하여 효율적인 그래디언트를 얻음.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/ce3fe827-55d7-4571-beee-975fd1f09e19/image.png)
        
    - **Paint-it**:
        
        - **텍스트를 사용한 텍스처 합성**: 텍스트 지시를 통해 3D 모델에 텍스처를 합성하는 방법.
        - **SDS 손실**을 활용하여 물리 기반 렌더링(PBR) 텍스처 맵을 최적화함.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/cc1fb833-0975-4231-ae6b-fb2d7201bc4e/image.png)
        

# **4. 3D Human**

## **4**.1 인간 모델의 중요성 (Why are Human Models Important?)

- 인간은 3D 세계에서 중심적인 역할을 하며, 가상 인간 모델은 인간-객체 상호작용, 인간-인간 상호작용, 자율 주행, 로봇, AR/VR 등의 다양한 응용 프로그램에서 필수적임.
- **인간 아바타 생성**:
    - 현실적인 3D 인간을 생성하고 제어 및 애니메이션이 가능하게 만들기 위해서는 비용이 많이 들고 시간 소모적이며, 개별 주제에 특화된 장비가 필요함.
    - 목표는 보다 효율적이고 대중적인 방법으로 인간 아바타를 생성하는 것.

## **4**.2 가상 인간의 목적 (Purpose of Virtual Humans)

- **가상 인간 생성**:
    - 현실적인 3D 인간을 생성하여 실제 사람처럼 움직이고 보이도록 하며, 제어가 쉽고 데이터에 맞추기 쉬운 특성을 가져야 함.
    - 이를 통해 가상 세계에서 현실적인 인간을 창조할 수 있음.
- **가상 인간의 움직임**:
    - 가상 인간은 물체 및 장면과 상호작용할 수 있으며, 실제 사람처럼 자유롭게 움직일 수 있어야 함.

## 4.3 인간 모델 생성의 어려움 (Challenges in Human Model Creation)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e21603f8-0eb2-422a-863a-ca5f07b76cf6/image.png)

- **주요 문제**:
    - **저조한 대비**와 **자기 폐색**: 이미지에서 일부 신체 부위가 다른 부위에 의해 가려지는 문제.
    - **2D 투영에서의 3D 정보 손실**: 3D 객체가 2D 평면으로 투영될 때 중요한 3D 정보가 손실됨.
    - **비정상적인 자세**: 고차원적인 복잡한 자세는 모델링이 어렵다는 문제가 있음.
    - **배경, 조명, 의복, 폐색**과 같은 요소들이 인간 모델 생성의 복잡성을 증가시킴.

## 4.4 신체 모델이란? (What is a Body Model?)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/ad6d4b05-6ead-4f43-8dff-1d155ff9317d/image.png)

- **신체 모델**은 실제 사람처럼 보이고 움직일 수 있는 수학적 모델을 정의하는 것임.
    - **특징**: 저차원, 미분 가능, 관절을 포함, 데이터를 기반으로 쉽게 맞출 수 있음.
    - **목표**: 인간의 형태를 3D 메쉬로 표현하여 실제와 유사하게 움직이도록 설계함.
    - 이러한 모델은 그래픽 도구와 호환이 가능하고, 애니메이션에서 쉽게 사용할 수 있음.

## 4.4 선형 블렌드 스키닝 (Linear Blend Skinning, LBS)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/074a5f8a-867d-4bbf-a398-94867b6aca20/image.png)

- **선형 블렌드 스키닝(LBS)**는 가장 일반적이고 간단한 신체 모델링 방식임.
- 각 정점(verteces)은 변형된 템플릿 정점들의 선형 결합을 통해 계산됨.
    - **문제점**:
        
        - LBS는 일부 관절 움직임에서 "캔디 래퍼 문제"와 같이 비현실적인 변형을 유발할 수 있음.
        - 이를 보완하기 위해 포즈 블렌드 쉐이프(pose blend shapes)를 도입하여 LBS의 한계를 극복하고, 더 자연스러운 변형을 가능하게 함.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/1a580574-cb9e-4383-b013-44c2844053a3/image.png)
        

## 4.5 SMPL (Skinned Multi-Person Linear Model)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/dc57f5e4-24df-4d7d-9625-b33e05da747f/image.png)

- **SMPL 모델**은 선형 블렌드 스키닝(LBS)에 포즈 종속 변형을 결합한 모델로, 3D 신체 메쉬를 생성함.
    - **특징**: 약 7,000개의 3D 정점으로 신체를 표현하며, 전체 신체를 21,000개의 숫자로 설명할 수 있음.
    - **데이터 기반**: SMPL은 다양한 신체 형태와 포즈를 학습하기 위해 수천 개의 3D 스캔 데이터를 사용함.
    - **신체의 다양한 형태 표현**: SMPL은 신체 형태를 저차원 공간에서 표현하며, 주성분 분석(PCA)을 사용하여 신체 형태를 설명함.
    - **포즈 블렌드 쉐이프**는 신체가 다양한 포즈에 따라 자연스럽게 변형되도록 하여 스키닝 문제를 보완함.
- **SMPL의 중요성**:
    - SMPL은 학계와 산업계에서 3D 신체 포즈 및 형태 모델링에 널리 사용되며, 사람의 손이나 얼굴 같은 세부 부위의 표현에서도 확장되어 사용됨.

## 4.6 SMPLify

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/8301b1ea-73e3-437f-a2c8-10ee4824cb71/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/f9ffe8c5-afac-46ea-93da-c3e0564ea7e1/image.png)

- **SMPLify**는 단일 이미지에서 자동으로 3D 신체의 포즈와 형태를 추정하는 방법을 제공함.
    - 2D 이미지에서 특징을 추출하고, 이 특징을 기반으로 3D 신체 메쉬를 예측하며, 2D 조인트와 3D 조인트의 차이를 최소화하는 방식으로 최적화함.
        
    - **문제점**: 깊이 모호성(Depth ambiguity) 문제로 인해 동일한 2D 투영이 여러 3D 포즈에 의해 생성될 수 있음. 예를 들어, 옆모습에서 포즈가 잘못될 수 있음.
        
    - **해결책**: 포즈와 형태에 대한 선험적 지식(Pose and Shape Prior)을 사용하여 포즈 및 형태 추정의 불확실성을 줄임.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/fd2da191-5286-41f2-871c-1c89f312f9db/image.png)
        

## 4.7 SPIN (SMPL oPtimization IN the loop)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/c522bf21-a016-47d0-bfa8-33e0740e8ba9/image.png)

- **SPIN**은 SMPLify의 상단에 최적화를 추가하여 2D 조인트 정보를 학습에 사용함.
    - **방법**: 하단에서 2D 조인트를 예측한 후, 상단에서 이 조인트를 바탕으로 3D 신체 포즈와 형태를 추정함.
    - **특징**: 단순한 회귀 기반 방법에 비해 더 정교한 3D 신체 메쉬를 생성할 수 있음.

## 4.8 MultiPly

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/c40b5650-a08f-4000-b07f-aa15ce8fdf9b/image.png)

- **MultiPly**는 단일 비디오에서 다수의 사람의 3D 포즈와 형태를 추정하는 방법임.
    - 실시간으로 복잡한 장면에서도 여러 사람의 정확한 3D 포즈를 복원할 수 있음.
    - 단일 카메라 설정에서 다중 사람 추적과 포즈 추정을 가능하게 함.