---
{"dg-publish":true,"permalink":"/2-naver-boost-camp/week-18/","created":"2025-03-11T13:34:42.964+09:00","updated":"2025-03-11T14:11:57.176+09:00"}
---

# Generative AI
---
## 대형 언어 모델(LLM)이란?
대형 언어 모델(LLM, Large Language Model)은 수십억 개의 파라미터로 구성되어 방대한 사전 학습 텍스트 데이터를 기반으로 훈련된 범용 언어 모델입니다. 대표적인 예로는 OpenAI의 **ChatGPT**, Meta의 **LLaMA**, MISTRAL AI의 **Mistral**, Upstage의 **Solar** 등이 있습니다. 과거에는 GPT-1, GPT-2, BERT와 같은 pretrained language model이 각 태스크에 맞게 fine-tuning을 거쳐 사용되었으나, LLM은 하나의 모델로 다양한 태스크를 수행할 수 있다는 점에서 큰 변화를 가져왔습니다.

---
## LLM의 동작 원리
LLM은 **Zero-shot learning**과 **Few-shot learning**을 기반으로 동작합니다.
- **Zero-shot learning:** 모델이 사전 학습된 지식과 prompt만으로 태스크를 이해하고 수행하는 방식입니다.
- **Few-shot learning:** 모델이 prompt와 함께 제공된 몇 가지 예시(demonstration)를 통해 태스크를 이해하고 수행합니다.

---
### **Prompt란?**
LLM에게 원하는 작업과 실제 입력값을 제공하는 방법으로, 모델의 태스크 수행 및 출력문을 제어합니다.
Prompt는 다음 세 가지 요소로 구성됩니다.
- **Instruction:** 태스크 수행을 위한 구체적 지시문
- **Demonstration:** 해당 태스크의 입력-출력 쌍 예시
- **Input:** 실제 태스크 수행을 위한 입력문

> **예시**  
> **[Instruction]** 위 예시는 영화 리뷰에 대한 분석 결과야. 예시를 보고 아래 리뷰의 감성을 분석해줘.  
> **[Demonstration]**  
> 리뷰 : 러닝 타임 내내 웃음이 끊이지 않은 영화  
> 감성 : 긍정  
> **[Input]**  
> 리뷰 : 너무 길고 지루했다.  
> 감성 :

---

## LLM의 아키텍처

대부분의 LLM은 **Transformer** 구조를 변형한 두 가지 모델 구조를 채택합니다.
### 1. Encoder-Decoder 구조
![Pasted image 20250311135132.png](/img/user/images/Pasted%20image%2020250311135132.png)

- **동작 원리:**
    - **Encoder:** 입력 문장을 이해
    - **Decoder:** 이해한 내용을 바탕으로 문장 생성
- **예시:** T5(Text-to-Text Transfer Transformer)
- **학습 방법:**
    - **Span Corruption:** 입력 문장의 일부를 masking 처리한 후, masking id와 함께 복원 문장을 학습

## 2. Decoder-Only 구조
![Pasted image 20250311135142.png](/img/user/images/Pasted%20image%2020250311135142.png)
- **동작 원리:**  
    입력된 토큰 단위로 다음 토큰을 예측하며 문장을 생성합니다.
- **특징:**  
    대부분의 LLM이 causal decoder 구조를 사용하여 자연스러운 문장 생성을 수행합니다.

---

## Corpus 구축

LLM의 사전 학습을 위해서는 대규모 텍스트 데이터 집합인 **Corpus**가 필요합니다.  
원시 데이터에는 욕설, 혐오 표현, 중복 데이터, 개인 정보 등 학습에 불필요한 내용이 포함될 수 있으므로, 정제 과정을 통해 깨끗한 데이터를 구축합니다.

---
## Instruction Tuning

LLM의 응답 품질을 높이기 위해 **instruction tuning** 과정을 거칩니다. 이는 사용자의 다양한 입력에 대해 안전하고 도움이 되는 답변을 생성하도록 fine-tuning하는 과정으로, 세 가지 단계로 구성됩니다.

1. **Supervised Fine-Tuning (SFT):**
    - Prompt와 Demonstration을 활용하여 지도 학습을 진행합니다.
2. **Reward Modeling:**
    - LLM이 생성한 답변의 helpfulness(질문 의도에 맞는 유용성)와 safety(안전성)를 평가하여 점수를 산출합니다.
3. **Reinforcement Learning with Human Feedback (RLHF):**
    - PRO 알고리즘 등을 사용해 높은 점수를 받은 답변을 더욱 강화하도록 학습합니다.

이러한 과정을 통해 instruction tuning을 진행하면, 사용자 지시 호응도가 높아지고 거짓 정보 생성(할루시네이션) 빈도가 감소하게 됩니다.

---

## 기존의 언어 모델 학습 방법론과 In-Context Learning

과거에는 target task에 맞춰 모델 전체 또는 일부 파라미터를 업데이트하는 **fine-tuning** 방법이 주로 사용되었습니다.

- **방법론 종류:**
    1. **Feature-based approach:**
        - 사전 학습 모델에서 embedding을 추출 후, 별도의 classifier를 학습
    2. **Fine-tuning I:**
        - output layer만 업데이트
    3. **Fine-tuning II:**
        - 모든 layer를 업데이트성능은 일반적으로 파라미터를 많이 업데이트할수록 향상되지만, fine-tuning II가 가장 높은 성능을 보이는 대신 training efficiency는 떨어집니다.

또한 **In-Context Learning (ICL)** 방식은 몇 가지 예시를 prompt로 제공함으로써 새로운 태스크를 수행하는 방식인데, 이 경우 모델의 가중치는 업데이트되지 않습니다. 단, ICL은 무작위 label에도 모델이 잘 반응하는 연구 결과가 있어 신뢰성 측면에서 한계가 있습니다.

---

## PEFT (Parameter-Efficient Fine-Tuning)

모델 전체를 업데이트할 때 발생하는 문제(기억 상실, 막대한 자원 소모 등)를 해결하기 위해 **PEFT** 방법이 고안되었습니다. PEFT는 전체 파라미터 중 일부만 업데이트하는 접근 방식으로, 대표적인 4가지 방법이 있습니다.

### 1. Adapter Tuning
![Pasted image 20250311135204.png](/img/user/images/Pasted%20image%2020250311135204.png)

- **구조:**
    - 기존 모델의 각 레이어에 학습 가능한 작은 Feed Forward Network(FFN)를 삽입합니다.
    - Transformer의 벡터를 작은 차원으로 압축 후 비선형 변환을 거쳐 원래 차원으로 복원하는 병목 구조로 구성됩니다.
- **단점:**
    - 병목 레이어 추가로 inference latency(추론 지연)가 증가할 수 있습니다.

> **Adapter Tuning 구현 예시:**

```
def transformer_block_with_adapter(x):     
	residual = x     
	x = SelfAttention(x)     
	x = FFN(x)     # Adapter     
	x = LN(x + residual)     
	residual = x     
	x = FFN(x)     # transformer FFN     
	x = FFN(x)     # Adapter     
	x = LN(x + residual)     
	return x
```

### 2. Prefix Tuning
![Pasted image 20250311135329.png](/img/user/images/Pasted%20image%2020250311135329.png)
- **구조:**
    - 각 레이어에 학습 가능한 prefix(가상의 embedding)를 추가합니다.
- **특징:**
    - 각 태스크를 위한 벡터를 최적화하여 기존 모델과 결합합니다.

> **Prefix Tuning 구현 예시:**
```
def transformer_block_for_prefix_tuning(x):     
	soft_prompt = FFN(soft_prompt)     
	x = concat([soft_prompt, x], dim=seq)     
	return transformer_block(x)
```

### 3. Prompt Tuning

- **구조:**
    - 입력 레이어에 학습 가능한 prompt vector를 통합합니다.
- **특징:**
    - 자연어 prompt를 직접 추가하는 방식과 달리, embedding layer를 최적화하여 target task에 맞게 튜닝합니다.

> **Prompt Tuning 구현 예시:**

```
def soft_prompted_model(input_ids):     
	x = Embed(input_ids)     
	x = concat([soft_prompt, x], dim=seq)     
	return model(x)
```

## 4. Low-Rank Adaptation (LoRA)
![Pasted image 20250311135401.png](/img/user/images/Pasted%20image%2020250311135401.png)

- **구조:**
    - 사전 학습된 모델의 파라미터는 고정하고, 학습 가능한 low-rank 행렬을 삽입합니다.
- **특징:**
    - 행렬의 차원을 낮춰 추가 파라미터만 학습한 뒤 기존 모델에 합쳐 추가 연산 없이 활용할 수 있습니다.
- **장점:**
    - 기존 방법보다 높은 성능과 효율성을 보이며, 현재 가장 널리 사용되는 PEFT 방식입니다.

> **LoRA 구현 예시:**

```
def lora_linear(x):     
	h = x @ W     # regular linear     
	h += x @ W_A @ W_B     # low-rank update     
	return scale * h
```

---

# LLM의 평가 방법

LLM이 다양한 태스크를 얼마나 잘 수행하는지를 평가하기 위한 여러 데이터셋과 평가 방법이 있습니다.

## 주요 평가 데이터셋

- **MMLU (Massive Multitask Language Understanding):**
    
    - 57개 이상의 태스크(생물, 정치, 수학, 물리학, 역사, 지리, 해부학 등)로 구성된 평가 데이터셋.
    - 객관식 형태로 정답 보기를 생성하여 평가합니다.
- **HellaSwag:**
    
    - 모델이 일반 상식을 보유하고 있는지 평가하는 데이터셋.
    - 주어진 문장에 자연스러운 이어지는 문장을 선택하는 방식입니다.
- **HumanEval:**
    
    - 코드 생성 능력을 평가하는 데이터셋으로, 함수명과 docstring을 기반으로 생성된 코드의 결과물을 검증합니다.

## 평가 프레임워크

- **LLM-Evaluation-Harness:**
    
    - MMLU, HellaSwag 등 다양한 평가 데이터셋을 불러와 자동으로 평가할 수 있는 프레임워크입니다.
- **G-Eval:**
    
    - LLM의 창의적 글쓰기 능력(자기소개서 수정, 광고 문구 생성, 어투 변경 등)을 평가합니다.
    - AutoCoT(모델 스스로 추론 단계를 구축하는 프롬프트 방식)을 통해 평가 단계를 정의하고, 인간 평가 점수와의 상관관계를 측정합니다.

---

## 결론

대형 언어 모델(LLM)은 방대한 파라미터와 다양한 학습 기법을 통해 하나의 모델로 여러 태스크를 수행할 수 있는 혁신적인 기술입니다.

- **학습 방식:** Zero-shot, Few-shot learning 및 In-Context Learning
- **모델 아키텍처:** Encoder-Decoder와 Decoder-Only 구조
- **파인튜닝 기법:** 전통적인 fine-tuning 방식과 PEFT (Adapter, Prefix, Prompt, LoRA)
- **평가 방법:** MMLU, HellaSwag, HumanEval 등 다양한 데이터셋과 평가 프레임워크 활용
---

## GAN과 다양한 이미지 생성 모델 기술 정리

이 포스트에서는 기본적인 GAN(Generative Adversarial Networks)부터 최신 생성 모델에 이르기까지, 다양한 이미지 생성 기법과 그 학습 방법, 그리고 평가 지표에 대해 살펴봅니다. 각 모델의 특징과 손실 함수, 그리고 paired/unpaired 데이터의 개념까지 자세히 정리해 보았습니다.

---

## 3 1. GAN (Generative Adversarial Networks)
![Pasted image 20250311135856.png](/img/user/images/Pasted%20image%2020250311135856.png)
- **기본 개념:**  
    GAN은 **생성자(generator)**와 **판별자(discriminator)**가 적대적으로 학습하는 구조입니다.
    - **판별자:** 입력 이미지가 실제(real) 이미지인지 생성된(fake) 이미지인지를 구분하도록 학습합니다.
    - **생성자:** 잠재 변수 zzz를 입력받아 학습 데이터의 분포를 모방하는 진짜 같은 이미지를 생성합니다.
- **학습 방식:**
    - 판별자는 진짜와 가짜를 잘 분류하도록 (maximize)
    - 생성자는 판별자가 속지 못하도록 (minimize) 학습합니다.

---

### 2. 조건부 GAN (cGAN)

- **정의:**  
    기본 GAN에 조건(condition)을 추가하여, 주어진 조건에 따라 이미지를 생성할 수 있도록 한 모델입니다.

---

### 3. Pix2Pix

- **개념:**
    - cGAN의 한 종류로, **조건 이미지**를 받아서 다른 이미지를 생성하는 모델입니다.
    - 학습을 위해 **paired image** 쌍, 즉 변환 전후의 1:1 대응 이미지가 필요합니다.
        - 예: 구두 스케치와 해당 구두의 실제 사진, 장화 스케치와 실제 장화 사진

---

### 4. CycleGAN
![Pasted image 20250311135930.png](/img/user/images/Pasted%20image%2020250311135930.png)

- **동기:**
    - Pix2Pix처럼 paired 데이터를 준비하기 어려운 경우가 많습니다.
- **해결책:**
    - **Unpaired image**를 사용하여 학습하고, **cycle consistency loss**를 도입하여 변환의 일관성을 확보합니다.
- **Cycle Consistency Loss:**
	![Pasted image 20250311140152.png](/img/user/images/Pasted%20image%2020250311140152.png)
    - **아이디어:**
        - 도메인 XXX의 이미지를 도메인 YYY로 변환한 후 다시 XXX로 복원했을 때, 원본과 최대한 유사해야 한다.
        - 반대로 $Y→X→Y$ 에서도 동일한 원칙을 적용합니다.
    - **구성:**
        - $X→Y$ 변환 생성자 GGG
        - $Y→X$ 변환 생성자 FFF
        - 도메인 $X$ 판별자 $D_X$​, 도메인 $Y$ 판별자 $D_Y$​
        - 전체 목적 함수는 **adversarial loss $L_{GAN}$**와 **cycle consistency loss ($L_{cyc}$​)**로 구성되어, paired 데이터 없이도 변환 신뢰성을 확보합니다.

---

### 5. Paired Image vs. Unpaired Image

- **Paired Image:**
    - **정의:** 변환 전후에 1:1로 정확하게 대응되는 이미지 쌍
    - **예시:** 구두 스케치 ↔ 해당 구두의 실제 사진
    - **특징:** 직접적인 대응 정보로 모델이 명확하게 학습할 수 있음
- **Unpaired Image:**
    - **정의:** 두 이미지가 1:1로 직접 대응되지 않는 쌍
    - **예시:** 실제 성의 사진 ↔ 숲의 유화 이미지, 낮의 강 사진 ↔ 밤의 저택 유화 이미지
    - **특징:** 직접적인 매칭은 없지만, 도메인(예: 실제 사진 vs. 유화)의 분포 차이를 학습하는 데 사용됨

---

### 6. StarGAN
![Pasted image 20250311140259.png](/img/user/images/Pasted%20image%2020250311140259.png)
![Pasted image 20250311140630.png](/img/user/images/Pasted%20image%2020250311140630.png)
- **목적:**  
    하나의 생성 모델로 **여러 도메인**의 이미지를 생성할 수 있도록 설계되었습니다.
- **주요 손실 함수 구성:**
    - **판별자 $L_D$: **$L_D = -L_{GAN} + \lambda_{cls} L_{cls}^{r}$
        - $L_{GAN}$​: adversarial loss
        - $\lambda_{cls}$: classification loss의 중요도를 조정하는 가중치
        - $L_{cls}^{r}$​: 실제 이미지의 도메인 분류 손실
    - **생성자 $L_G$​:** $L_G = L_{GAN} + \lambda_{cls} L_{cls}^{f} + \lambda_{rec} L_{rec}$​
        - $L_{cls}^{f}$​: 생성된 이미지의 올바른 도메인 분류 손실
        - $\lambda_{rec}$​: reconstruction (cycle consistency) loss의 가중치
        - $L_{rec}$: cycle consistency loss

---

### 7. ProgressiveGAN

- **목적:**  
    고해상도 이미지 생성을 위한 학습 비용을 낮추고, 단계적으로 해상도를 증강하는 모델입니다.
- **특징:**
    - 저해상도 이미지 생성부터 시작하여 점진적으로 해상도를 높이는 방식
    - 작은 해상도와 큰 해상도의 결과를 weighted sum으로 계산하여 사용

---

### 8. StyleGAN
![Pasted image 20250311140642.png](/img/user/images/Pasted%20image%2020250311140642.png)
- **개념:**
    - ProgressiveGAN의 확장으로, 이미지의 **style** (색상, 톤, 대비, 텍스처, 패턴, 디테일 등)을 제어할 수 있도록 설계되었습니다.
- **주요 과정:**
    
    1. 기본 랜덤 노이즈 벡터 zzz를 비선형 변환을 통해 새로운 latent space WWW로 매핑
        - 매핑 함수 fff는 데이터 분포에 맞춰 얽힘을 해소합니다.
    2. WWW에 affine transform을 적용하여 스타일 정보 γ\gammaγ와 β\betaβ를 계산
    3. **Adaptive Instance Normalization (AdaIN)**을 통해 각 레벨에 스타일을 주입합니다.
    
    **AdaIN 공식:** AdaIN(x,γ,β)=γ(x−μ(x)σ(x))+β\text{AdaIN}(x,\gamma,\beta) = \gamma \left(\frac{x - \mu(x)}{\sigma(x)}\right) + \betaAdaIN(x,γ,β)=γ(σ(x)x−μ(x)​)+β
    - 저해상도 레벨에서는 global style, 고해상도 레벨에서는 local style 제어

---

### 9. AutoEncoder 계열 모델

#### 9-1. AutoEncoder
![Pasted image 20250311140654.png](/img/user/images/Pasted%20image%2020250311140654.png)
- **구조:**
    - **Encoder gϕg_\phigϕ​:** 입력 이미지를 저차원 latent space로 매핑하여 잠재 변수 $z$ 생성
    - **Decoder fθf_\thetafθ​:** $z$를 입력 받아 원본 이미지를 복원
- **목적 함수:**
    - 주로 **reconstruction loss** (MSE 혹은 MAE)를 사용

#### 9-2. Variational AutoEncoder (VAE)

- **특징:**
    - AutoEncoder와 유사하지만, latent space의 분포를 가정하여 학습
    - **목적 함수:**
        - Reconstruction loss와 함께 latent 분포를 정규화하기 위한 **KL divergence**를 추가

#### 9-3. VQ-VAE (Vector Quantized-Variational AutoEncoder)

- **개념:**
    - 연속적인 latent space 대신 **고정된 크기의 codebook**에서 이산적인 벡터를 선택하여 데이터를 표현
- **프로세스:**
    1. Encoder가 입력 xxx를 압축하여 latent vector $z(x)$ 생성
    2. **Vector Quantization:** $z(x)$를 사전 정의된 codebook 내 가장 가까운 이산 벡터로 매핑
    3. Decoder는 양자화된 벡터 $z_q(x)$를 사용해 원본 데이터를 복원
- **손실 함수 구성:** $L = L_{rec} + L_{commit} + L_{codebook}$
    - $L_{rec}$​: 재구성 오차
    - $L_{commit}$​: Encoder의 출력 $z(x)$가 codebook 벡터 $z_q(x)$에 가까워지도록 유도
    - $L_{codebook}$: 코드북 벡터가 $z(x)$에 근접하도록 조정 (역전파 차단 포함)

---

### 10. Diffusion Model 계열

#### 10-1. DDPM (Denoising Diffusion Probabilistic Models)
![Pasted image 20250311140856.png](/img/user/images/Pasted%20image%2020250311140856.png)
- **원리:**
    - **Forward process:** 입력 이미지에 점진적으로 Gaussian noise를 추가해 latent 공간으로 매핑
    - **Reverse process:** 추가된 노이즈를 추정, 제거하여 이미지를 복원

#### 10-2. DDIM (Denoising Diffusion Implicit Models)

- **목적:**
    - DDPM의 sampling step이 많은 문제를 해결
- **특징:**
    - 몇 단계 이전의 정보를 고려하는 deterministic(비마르코프) 방식으로 노이즈 제거를 수행, 더 적은 단계로 고품질 샘플링 가능

#### 10-3. Classifier Guidance & Classifier-free Guidance
![Pasted image 20250311140925.png](/img/user/images/Pasted%20image%2020250311140925.png)
- **Classifier Guidance (CFG):**
    - 노이즈 제거 시, 특정 클래스에 속할 확률이 높아지도록 조정하는 방법
    - 단점: diffusion pipeline에 classifier를 추가해야 하므로 복잡도가 상승하고, 모든 step에 classifier가 필요
- **Classifier-free Guidance:**
    - conditional score와 unconditional score로 분해하여, 별도의 classifier 없이 class에 대한 guidance를 가중치로 조정할 수 있게 함

---

### 11. Latent Diffusion Model (LDM)
![Pasted image 20250311141022.png](/img/user/images/Pasted%20image%2020250311141022.png)
- **개념:**
    - 기존의 diffusion model은 pixel space에서 처리하지만, LDM은 **latent space**에서 diffusion을 수행하여 계산 효율성을 극대화합니다.
- **프로세스:**
    - Encoder를 통해 입력 이미지를 latent space로 변환
    - latent space에서 노이즈를 추가·제거한 후, Decoder를 통해 원본 이미지 공간으로 복원
    - **Cross attention**을 활용해 조건(condition) embedding을 반영

---

### 12. Stable Diffusion
![Pasted image 20250311141052.png](/img/user/images/Pasted%20image%2020250311141052.png)
- **개요:**
    - LDM을 기반으로 한 text-to-image 생성 모델로, 텍스트 프롬프트를 입력받아 고품질 이미지를 생성합니다.
- **구조 및 작동 방식:**
    1. **Image Information Creator:**
        - latent space에서 U-Net 구조가 noise 예측을 담당
        - input latent와 noise 주입 정도를 noise scheduler를 통해 noisy latent 생성
        - noisy latent, time embedding, 그리고 text encoder의 token embedding(크로스 어텐션 활용)을 결합하여 노이즈를 예측
    2. **Conditioning:**
        - 텍스트 입력은 CLIP text encoder를 통해 embedding으로 변환되고, latent space에서 noisy latent와 결합하여 조건에 부합하는 이미지 생성 유도

#### Stable Diffusion의 학습 과정
![Pasted image 20250311141100.png](/img/user/images/Pasted%20image%2020250311141100.png)
1. **입력 데이터 인코딩:**
    - 이미지와 텍스트를 각각의 encoder를 통해 latent space로 변환
2. **노이즈 추가:**
    - image latent에 noise scheduler를 사용하여 random한 timestep만큼 노이즈 추가
3. **U-Net 학습:**
    - noisy latent, token embedding, time step을 입력받아 U-Net이 노이즈를 예측
    - 예측된 노이즈와 실제 노이즈 간의 차이를 MSE loss로 계산하여 학습

#### Stable Diffusion의 Inference

- **Text-to-Image Task:**
    - Gaussian noise 상태에서 시작하여, text의 token embedding을 바탕으로 iterative하게 노이즈를 제거
    - 최종 latent를 Decoder를 통해 이미지로 복원
- **Inpainting Task:**
    - 기존 이미지에서 변환된 latent에 noise를 가한 noisy latent에서 시작
    - time step 조절로 input 이미지의 영향 정도를 조정 후, 동일한 noise 제거 과정을 통해 이미지 생성

#### Stable Diffusion 2 및 XL

- **Stable Diffusion 2:**
    - 생성 이미지 해상도가 기존 512×512에서 768×768로 향상
    - text encoder를 CLIP에서 OpenCLIP으로 업데이트
    - v-prediction을 도입하여 노이즈와 원본 데이터 간 혼합 정도(velocity)를 예측
    - super-resolution upscaler diffusion model 추가로 2048×2048 이상의 고해상도 생성 및 depth guided image generation 지원
- **Stable Diffusion XL (SDXL):**
    - 보다 현실적인 이미지 생성
    - 2-stage 모델 (base + refiner)
    - 두 개의 text encoder 사용
    - 정방형을 벗어난 다양한 비율의 이미지 생성 가능
    - SDXL Turbo: Adversarial Diffusion Distillation을 적용해 추론 속도 향상 (one-step generation)

---

### 13. 이미지 생성 모델 평가 지표

#### 13-1. Inception Score

- **목적:**
    - 생성된 이미지의 **fidelity(질)**와 **diversity**를 평가
- **원리:**
    - 생성 이미지가 특정 클래스에 치우친 likelihood 분포를 보이면 fidelity가 높다고 판단
    - 다양한 클래스가 생성되면 marginal distribution이 균일해져 diversity가 높다고 평가
    - 두 분포의 KL divergence를 계산하여 score가 클수록 좋은 결과

#### 13-2. FID (Frechet Inception Distance) Score

- **원리:**
    - Inception Network를 통해 추출한 실제 이미지와 생성 이미지의 embedding 간 Frechet distance를 계산
    - FID score가 낮을수록 실제와 생성 이미지 간의 분포 차이가 작아, 더 나은 품질을 의미

#### 13-3. CLIP Score

- **목적:**
    - 이미지와 캡션(텍스트) 사이의 상관 관계를 평가
- **방법:**
    - CLIP 모델을 사용해 이미지와 캡션의 embedding을 생성하고, 두 embedding의 cosine similarity를 계산
    - 유사도가 높을수록 이미지와 텍스트 간 일치도가 높다고 평가

---

# 결론

본 포스트에서는 GAN의 기본 개념부터 시작하여, 조건부 GAN, Pix2Pix, CycleGAN, StarGAN, ProgressiveGAN, StyleGAN과 같은 다양한 생성 모델과 AutoEncoder 계열, Diffusion Model, 그리고 최신 Stable Diffusion까지 폭넓게 다루었습니다.  
또한, 각 모델의 학습 전략(예: cycle consistency, classifier guidance, latent diffusion)과 평가 지표(Inception, FID, CLIP score)에 대해서도 정리하여, 이미지 생성 모델의 전체적인 흐름과 최신 기술 동향을 한눈에 파악할 수 있도록 하였습니다.