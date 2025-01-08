---
{"dg-publish":true,"permalink":"/naver-boost-camp/week-1/","created":"2025-01-07T18:03:53.724+09:00","updated":"2025-01-08T20:18:10.271+09:00"}
---

[[NaverBoostCamp/Week 2 학습 정리\|Week 2 학습 정리]]
# Day 1

## 1. Pytorch Intro

### Pytorch

간편한 딥러닝 API를 제공, 확장성이 뛰어난 멀티플랫폼 프로그래밍 인터페이스

API : 응용 프로그램이 서로 상호작용하는데 사용하는 명려어, 함수, 프로토콜의 집합, 즉 딥러닝을 개발할 때, Pytorch를 사용할 수 있도록 제공하는 인터페이스

Pytorch 아키텍처
![Pasted image 20250107180532.png](/img/user/images/Pasted%20image%2020250107180532.png)

### Tensor

Pytorch에서 사용하는 핵심 데이터 구조, Numpy의 다차원 배열과 유사함.

- 0-D Tensor : Scalar, 하나의 숫자로 표현
- 1-D Tensor : Vector, 순서가 지정된 여러 개의 숫자들이 일렬로 나열된 구조
- 2-D Tensor : 1-D Tensor가 모여 행과 열로 구성된 2차원 구조
- 3-D Tensor : 2-D Tensor를 모아 행, 열, 채널로 구성된 3차원 구조
- N-D Tensor(N ≥ 4 ) : (N-1)-D Tensor들이 여러 개 모여 형성된 N차원 구조

## 2. Pytorch의 데이터 타입

### Pytorch의 데이터 타입 == Tensor가 저장하는 데이터 유형

- 정수형 : 소수 부분이 없는 숫자를 저장하는 데이터 타입unsigned - 부호가 없는 유형 (ex. uint8 → 8비트 모두 숫자로 표현 (0~255))signed - 부호가 있는 유형 (ex. int8 → 맨 앞의 1bit는 부호 표현(0 = 양수, 1 = 음수), 나머지 7bit로 숫자 표현(-128~127))
- 실수형 : 32, 64 bit 부동 소수점 수의 유형으로 구분함, 신경망 수치 계산에 사용부동 소수점 수 : 숫자를 정규화하여 가수부와 지수부로 나누어 표현32bit = torch.float32, torch.float / 64bit = torch.float64, torch.double

![https://blog.kakaocdn.net/dn/ctfbx2/btsITwpdOB0/XGp7MPIqVbSCvhTXRERbb1/img.png](https://blog.kakaocdn.net/dn/ctfbx2/btsITwpdOB0/XGp7MPIqVbSCvhTXRERbb1/img.png)

![https://blog.kakaocdn.net/dn/v2M6A/btsIURTlAaM/t9uZwH8x7MLSpwo8QNLgtk/img.png](https://blog.kakaocdn.net/dn/v2M6A/btsIURTlAaM/t9uZwH8x7MLSpwo8QNLgtk/img.png)

### 타입 캐스팅

한 데이터 타입을 다른 데이터 타입으로 변환하는 것

```
# Tensor 생성
i = torch.tensor([2,3,4], dtype = torch.int8)

# 32bit 부동 소수점 수로 변환
j = i.float()

# 64bit 부동 소수점 수로 변환
k = i.double()
```

## 3. Tensor의 기초 함수 및 메서드

```python
torch.min("tensor")# tensor의 최솟값 반환
torch.max("tensor")# tensor의 최댓값 반환
torch.sum("tensor")# tensor의 요소의 합 반환
torch.prod("tensor")# tensor의 요소의 곱 반환
torch.mean("tensor")# tensor의 요소의 평균 반환
torch.var("tensor")# tensor의 요소의 표본분산 반환
torch.std("tensor")# tensor의 요소의 표본표준편차 반환# tensor T가 있을 때
T.dim()# T의 차원 수를 확인
T.size()# T의 크기를 확인
T.shape# T의 크기를 확인(메서드가 아닌 속성)
T.numel()# T에 있는 요소의 총 개수 확인
```

---

# Day 2

## 1. Tensor의 생성

```
# 0 or 1로 초기화 후 생성
torch.zeros("shape")# shape 크기의 Tensor를 0으로 초기화하여 생성
torch.ones("shape")# shape 크기의 Tensor를 1로 초기화하여 생성
torch.ones_liskes("tensor")# tensor의 크기와 자료형이 같은 Tensor를 1로 초기화# random으로 초기화
torch.rand("shape")# shape 크기의 Tensor를 0~1사이의 연속균등분포 값으로 초기화 후 생성
torch.randn("shape")# shape 크기의 Tensor를 평균:1, 분산:1의 표준정규분포 값으로 초기화 후 생성# rand -> randn
torch.rand_like(k)# k와 크기와 자료형이 같은 연속균등분포 Tensor로 생성
torch.randn_like(k)# k와 크기와 자료형이 같은 표준정규분포 Tensor로 생성# 일정 간격의 값을 가진 Tenssor 생성
torch.arange(start = 1, end = 11, step = 2)# 1부터 10까지 2씩 증가하는 Tensor 생성
```

- 연속정규분포: 모든 값이 동일한 확률을 가지는 확률분포

![https://blog.kakaocdn.net/dn/cMgI4D/btsIV64d2cg/m3bkaBmu79EAcEpFkobo8K/img.png](https://blog.kakaocdn.net/dn/cMgI4D/btsIV64d2cg/m3bkaBmu79EAcEpFkobo8K/img.png)

- 표준정규분포: 평균이 0이고 표준편차가 1인 종 모양의 곡선을 가지는 확률분포

![https://blog.kakaocdn.net/dn/brboI2/btsIWRrNk6A/P837Ay7rV1roEXUGWBLDkK/img.png](https://blog.kakaocdn.net/dn/brboI2/btsIWRrNk6A/P837Ay7rV1roEXUGWBLDkK/img.png)

- 초기화 되지 않은 Tensor: 다른 값으로 채워질 예정이라면 초기값을 설정하는 것이 불필요하다 → 초기화 하지 않는다.메모리 효율성을 높일 수 있다.실제로 비어있는 것은 아니고 메모리에 존재하는 임의의 값으로 채워짐

```
torch.empty("shape")# shape 크기의 초기화 되어 있지 않은 Tensor 생성
tensor.fill_("value")# value 값으로 tensor를 채움
```

- List, Numpy로 Tensor 생성Numpy : C언어로 구현된 Python 핵심 과학 컴퓨팅 라이브러리, 배열을 효율적으로 조작할 수 있다.

```
s = [1, 2, 3, 4, 5, 6]
torch.tensor(s)# 배열 s로 Tensor 생성

u = np.array([[0, 1], [2, 3]])
v = torch.from_Numpy(u)# Numpy 배열을 Tensor로 변환
v = torch.from_Numpy(u).float()# Numpy는 기본 정수형이라서 실수형으로 타입캐스팅
```

- 여러 자료형으로 Tensor 생성torch.(자료형)Tensor 형태로 사용함

```
torch.ByteTensor()# 8비트 부호 없는 정수형 CPU Tensor 생성
torch.CharTensor()# 8비트 부호 있는 정수형 CPU Tensor 생성
torch.ShortTensor()# 16비트 부호 있는 정수형 CPU Tensor 생성
torch.FloatTensor()
torch.IntTensor()
torch.LongTensor()# 64비트 부호 있는 정수형 CPU Tensor 생성
torch.DoubleTensor()# 64비트 부호 있는 실수형 CPU Tensor 생성
```

## 2. Tensor의 복제

![https://blog.kakaocdn.net/dn/zV6bf/btsIXd2jvDc/wFzGRUktwdk3wGzlEfHKKk/img.png](https://blog.kakaocdn.net/dn/zV6bf/btsIXd2jvDc/wFzGRUktwdk3wGzlEfHKKk/img.png)

## 3. CUDA Tensor 생성 및 변환

```
tensor.device# tensor가 어디에 있는지 확인
torch.cuda.is_available()# cuda를 사용할 수 있는지 확인
torch.cuda.get_device_name(device=0)# cuda 이름을 확인 ex. NVIDIA GeForce GTX 1080
tensor.cuda() or tensor.to('cuda')# tensor를 GPU로 이동
```

## 4. Tensor의 Indexing과 Slicing

Numpy 에서의 인덱싱과 슬라이싱과 유사하다.

indexing : Tensor의 특정위치의 요소에 접근하는것

slicing : 부분집합을 선택하여 새로운 Sub Tensor 생성

## 5. Tensor 모양 변경

- `tensor.view("shape")` Tensor가 메모리에 연속적으로 저장된 경우에만 사용가능

```
tensor.is_contiguous()# tensor가 메모리에 연속적인지 아닌지 알려줌
```

- `torch.flatten("tensor") or tensor.flatten()` Tensor가 다차원인 경우 평탄화할 때 사용가능, 다차원 데이터를 처리할 때 유용

```
torch,flatten(k, 1)# 1차원과 같이 특정 차원부터 평탄화 가능
torch,flatten(k, 0, 1)# 0차원 부터 1차원까지 특정 차원 범위만 평탄화 가능
```

- `tensor.reshape("shape")` view와 다르게 Tensor가 메모리에 연속적이지 않아도 사용가능, 안전하고 유연하지만 성능이 저하됨

> 메모리가 연속적인게 확실하고 성능이 중요하면 view를 사용, 아니라면 reshape를 사용

- `tensor.transpose("dim1", "dim2")` dim1과 dim2의 축을 바꾼다.
- `torch.squeeze("tensor")` tensor내의 차원의 값이 1인 차원을 축소한다. ex) [1, 2, 3] → [2, 3]squeeze(dim = 1) → 특정 차원이 1일 때만 축소되도록 설정 가능
- `torch.unsqueeze("tensor", dim = n)` tensor내의 n차원의 값을 1로 확장한다. ex) dim=0일때 [2, 3] → [1, 2, 3]
- `torch.stack(("tensor1", "tensor2", ... ), dim=n)` dim의 default값은 0이고 dim축을 기준으로 tensor들을 쌓아올린다.tensor의 크기가 모두 같아야 사용할 수 있다.새로운 차원을 생성해서 결합한다.R, G, B 채널을 쌓아 컬러 이미지를 만든다고 생각하면 편하다.

---

# Day 3

## 1. Tensor의 모양 변경 2

- `torch.cat(("tensor1", "tensor2"), dim=0)` stack함수와 다르게 새로운 차원이 아니라 기존의 차원을 유지하면서 Tensor를 연결함
- `tensor.expand("shape")` f = (1,3)일때 g = f.expand(4,3)을 하면 g에는 (4,3)의 텐서가 저장됨Tensor 차원 중 하나라도 크기가 1이어야하는 제약이 없음, 메모리 할당이 없어 메모리 효율성↑
- `tensor.repeat("반복횟수 1", "반복횟수 2")` tensor를 dim=0축으로 반복횟수 1만큼 반복, dim=1축으로 반복횟수 2만큼 반복Tensor의 차원 중 하나라도 크기가 1이어야함, 추가 메모리를 할당하기 때문에 메모리 효율성↓

## 2. Tensor의 기초 연산

- 더하기 연산 두 Tensor를 요소별로 더하는 연산, 두 Tensor의 크기가 다르면 브로드 캐스팅하여 진행 `torch.add(tensor1, tensor2)`
    
    - **in-place 방식 : 메모리를 절약하며 Tensor의 값을 업데이트 할 수 있는 연산 추가적인 메모리 할당이 필요 없기 때문에 메모리 사용량을 줄일 수 있다. 하지만 Autograd와의 호환성에 문제가 생길 수 있다. `tensor1.add_(tensor2)`**
- 빼기 연산 두 Tensor의 요소별로 빼는 연산, 두 Tensor의 크기가 다르면 브로드 캐스팅하여 진행 일반 : `torch.sub(tensor1, tensor2)` in-place : `tensor1.sub_(tensor2)`
    
- 스칼라 곱 하나의 Tensor의 각 요소에 동일한 값을 곱하는 연산 `torch.mul(salar_value, tensor)`
    
- 요소별 곱하기 연산 (element-wise product) 두 Tensor의 요소별로 곱하는 연산, 두 Tensor의 크기가 다르면 브로드 캐스팅하여 진행
    
    일반 : `torch.mul(tensor1, tensor2)` in-place : `tensor1.mul_(tensor2)`
    
- 요소별 나누기 연산 두 Tensor의 요소별로 나누는 연산, 두 Tensor의 크기가 다르면 브로드 캐스팅하여 진행
    
    일반 : `torch.div(tensor1, tensor2)` in-place : `tensor1.div_(tensor2)`
    
- 요소별 거듭제곱 연산 Tensor1의 요소에 각각 Tensor2의 값만큼 거듭제곱하는 연산, 두 Tensor의 크기가 다르면 브로드 캐스팅하여 진행 일반 : `torch.pow(tensor1, tensor2)` in-place : `tensor1.pow_(tensor2) torch.pow(tensor1, 1/n)` ⇒ n제곱근을 계산하는 코드
    
- 비교연산 `torch.eq(v, w)` 를 이용해 두 tensor의 대응요소가 같은지 boolean tensor로 출력
    
    `torch.ne(v, w)` 를 이용해 두 tensor의 대응요소가 다른지 boolean tensor로 출력
    
    대응요소들보다 큰지를 비교하는 코드 표현: `torch.gt(v, w)` 대응요소들보다 크거나 같은지를 비교하는 코드 표현: `torch.ge(v, w)`
    
    Tensor v의 요소들이 Tensor w의 대응요소들보다 작은지를 비교하는 코드 표현: `torch.lt(v, w)` Tensor v의 요소들이 Tensor w의 대응요소들보다 작거나 같은지를 비교하는 코드 표현: `torch.le(v, w)`
    
- 논리 연산 AND 연산 - `torch.logical_and(x, y)` OR 연산 - `torch.logical_or(x, y)` XOR 연산 - `torch.logical_xor(x, y)`
    

## 3. Tensor의 노름

1-D Tensor의 노름은 Vector가 원점에서 얼마나 떨어져 있는지를 의미함 Vector의 길이를 측정하는 방법으로 사용됨

- L1 norm 1-D Tensor에 포함된 요소의 절댓값의 합으로 정의, 맨해튼 norm이라고도 부름 코드 : `torch.norm(a, p=1)` p는 L1인지 L2인지를 입력

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/38e42743-2c72-4d87-ab86-5be012769864/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e7f4e356-d56d-4904-816c-146701eeb567/Untitled.png)

- L2 norm 1-D Tensor에 포함된 요소의 제곱합의 제곱근으로 정의, 유클리디안 norm이라고도 부름
    
    코드 : `torch.norm(a, p=2)`
    

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/32126d83-cd93-4c19-ae7e-09f86e747610/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/f13fae5b-6d62-4ee0-87a9-dd37aae88da2/Untitled.png)

- L∞ norm 1-D Tensor에 포함된 요소의 절대값 중 최대값으로 정의,
    
    코드 : `torch.norm(a, p=float('inf'))`
    

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/face7645-6c82-4b71-bec2-1823e1d8e69b/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/167a5efc-3c68-4b93-a997-c915369cc4a1/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/fed49bc5-0a39-43ad-adcf-a27276821317/Untitled.png)

## 4. 유사도

유사도(Similarity)란 두 1-D Tensor(=Vector)가 얼마나 유사한지에 대한 측정값을 의미

- 맨해튼 유사도 두 1-D Tensor 사이의 맨해튼 거리를 역수로 변환하여 계산한 값 맨해튼 거리의 값이 작아질 수록 맨해튼 유사도의 값은 커짐 1에 가까울수록 두 Tensor가 유사
    
    `manhattan_distance = torch.norm(b – c, p = 1) – 1 / (1 + manhattan_distance)`
    

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/c614f547-f214-4e9c-bb43-d388d363858a/Untitled.png)

- 유클리드 유사도 두 1-D Tensor 사이의 유클리드 거리를 역수로 변환하여 계산한 값 유클리드 거리의 값이 작아질 수록 맨해튼 유사도의 값은 커짐 1에 가까울수록 두 Tensor가 유사
    
    `euclidean_distance = torch.norm(b – c, p = 2) – 1 / (1 + euclidean_distance)`
    

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/72e00a25-d835-4225-9205-1c68fcc1aa1d/Untitled.png)

- 코사인 유사도 두 1-D Tensor 사이의 각도를 측정하여 계산한 값 코사인 유사도의 값이 1에 가까울 수록 두 Tensor가 유사 1-D Tensor(=Vector)의 **내적(dot product 또는 inner product)**을 활용하여 계산 `cosine_similarity = torch.dot(b, c) / (torch.norm(b, p = 2) * (torch.norm(c, p = 2))`
    - 내적 `torch.dot(b, c)`
        
        1. 두 1-D Tensor의 각 요소를 곱해서 더하는 방법
        2. 두 1-D Tensor의 길이를 곱하는 방법
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6f43071d-18d1-40c0-adae-f13a34177936/Untitled.png)
        
    - 코사인 유사도 수식표현 유도
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/fbe9c95f-218f-499e-b7ac-5891ae71bc71/Untitled.png)
        

## 5. 2-D 행렬 곱셈 연산

코드 표현 : `D.matmul(E)`, `D.mm(E)` , `D @ E`

행렬 곱셈을 통해 좌우 대칭이동이 가능하다 ⇒ $I =$ $\begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \end{bmatrix}$행렬을 곱해주면 된다. 흑백이미지 행렬 * $I$

그렇다면 흑백 이미지의 상하로 대칭 이동은 어떤 축을 기준으로 이미지를 뒤집는 변환일까요?

- 정답
    
    x 축을 기준으로 뒤집는 변환
    

또한, 흑백 이미지를 상하로 대칭 이동시키기 위해서는 어떤 행렬을 어떻게 곱셈해야 할까요?

- 정답
    
    $I$ * 흑백 이미지 행렬
    

---

# Day 4

## 1. 선형 회귀 모델

### 1.1 선형 회귀의 의미

- **정의**: 선형 회귀는 주어진 트레이닝 데이터를 사용하여**(1)** 특징 변수와 목표 변수 사이의 선형 관계를 분석하고**(2)**, 이를 바탕으로 모델을 학습시켜**(3)** 새로운 데이터의 결과를 연속적인 숫자 값으로 예측하는 과정입니다.**(4)**
- **활용 예시**:
    - **임금 예측**: 연차에 따른 연봉 예측
    - **부동산 가격 예측**: 주택의 방 개수, 교육 환경, 범죄율 등에 따른 주택 가격 예측

### 1.2 트레이닝 데이터

- **트레이닝 데이터 정의**: 트레이닝 데이터는 모델을 학습시키기 위해 필요한 데이터로, 예시로 `YearsExperience`(연차)와 `Salary`(임금)의 관계를 나타낸 데이터셋이 사용됩니다.
    
- **특징 변수와 목표 변수**:
    
    - **특징 변수**: `YearsExperience`(연차) ⇒ 예측할 하기 위한 변수
        - **목표 변수**: `Salary`(임금) ⇒ 예측할 변수
- **코드 예시**:
    
    - Kaggle에서 데이터셋을 다운로드하고 이를 불러오는 코드를 포함하여 설명합니다.
    
    ```python
    !kaggle datasets download –d abhishek14398/salary-dataset-simple-linear-regression
    !unzip salary-dataset-simple-linear-regression.zip
    
    data = pd.read_csv(“Salary_dataset.csv”, sep = ‘,’, header = 0)
    
    x = data.iloc[:, 1].values
    t = data.iloc[:, 2].values
    ```
    

### 1.3 상관 관계 분석

- **목적**: 특징 변수와 목표 변수 간의 선형 관계를 파악하기 위해 상관 관계를 분석합니다.**(2)**
- **상관 관계 분석**:
    - 두 변수 간의 선형 관계를 파악하고, 그 관계가 양의 관계인지 또는 음의 관계인지를 파악합니다.
    - 높은 상관 관계를 가지는 특징 변수들을 파악합니다. (절댓값이 클수록 높은 상관 관계)
- **수식 표현**: 표본 상관 계수를 통해 두 변수 간의 관계를 수식으로 나타냅니다. $r_{xt} = \frac{\sum_{i=1}^{n} (x_i - \overline{x})(t_i - \overline{t})}{\sqrt{\sum_{i=1}^{n} (x_i - \overline{x})^2 \sum_{i=1}^{n} (t_i - \overline{t})^2}}$ (표본상관계수)
- **코드 예시**:
    - 상관 관계 분석 코드: `np.corrcoef(x, t)`
    - 상관 관계 시각화 코드: `plt.scatter(x, t)`

### 1.4 선형 회귀 모델에서의 학습

- **학습 정의**: 선형 회귀 모델에서의 학습은 주어진 트레이닝 데이터를 가장 잘 표현할 수 있는 직선 $y=wx+$b의 기울기(가중치) $w$와 절편(바이어스) $b$
    
- **신경망 관점에서의 설명**:
    
    - 선형 회귀 모델은 입력층의 특징 변수를 출력층의 예측 변수로 매핑하는 과정으로 설명됩니다.
    - 각 뉴런은 가중치와 바이어스를 통해 연결되며, 모델이 학습하는 파라미터로 사용됩니다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/955e9397-2dc6-41ca-b989-e860d9de7fce/Untitled.png)
    
- **PyTorch 구현**:
    
    - **nn.Module 클래스**: PyTorch에서 신경망의 모든 계층을 정의하기 위해 사용되는 기본 클래스입니다. 이를 상속받아 복잡한 신경망 모델을 구축할 수 있습니다.
        
        - 장점
            
            일관성: 모든 신경망 모델을 같은 방식으로 정의하고 사용할 수 있음 모듈화: 모델을 계층별로 나누어 관리할 수 있어 코드가 깔끔하고 이해하기 쉬움 GPU지원: 대규모 연산을 가속화하여 모델의 학습 속도를 높임 자동 미분, 최적화, 디버깅과 로깅
            
    - **코드 예시**:
        
        ```python
        python코드 복사
        class LinearRegressionModel(nn.Module):
            def __init__(self):
                super(LinearRegressionModel, self).__init__()
                self.linear = nn.Linear(1, 1)
        
            def forward(self, x):
                y = self.linear(x)
                return y
        model = LinearRegressionModel()
        ```
        
- **오차**:
    
    - 오차는 목표 변수 $t$와 예측 변수 $y$의 차이를 의미합니다.
    - 선형 회귀 모델에서는 오차의 총합이 최소가 되도록 가중치 $w$와 바이어스 $b$를 찾는 것이 중요합니다.
- **손실 함수**:
    
    - 손실 함수는 **목표 변수와 예측 변수 간의 차이를 측정하는 함수**이며, 이를 통해 모델을 최적화합니다.
        
    - 평균 제곱 오차(Mean Squared Error, MSE) 방식이 주로 사용됩니다.
        
    - **수식**:
        
        $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} [t_i - (wx_i + b)]^2$
        
    - 목적 위 수식에서 $w$와 $b$를 바꾸어 가며 손실함수가 작아지도록 만들어야합니다. 손실 함수의 값이 크다면 평균오차가 크다. 손실 함수의 값이 작으면 평균오차가 작다.
        
    - **코드 예시**:
        
        ```python
        python코드 복사
        loss_function = nn.MSELoss()
        ```
        

### 1.5 학습 정리

- **요약**:
    - 선형 회귀 모델은 주어진 트레이닝 데이터를 사용하여 특징 변수와 목표 변수 사이의 선형 관계를 분석하고, 이를 바탕으로 새로운 데이터를 예측하는 모델입니다.
    - 신경망 관점에서 선형 회귀는 입력층의 특징 변수가 출력층의 예측 변수로 사상되는 과정이며, 학습 과정에서 오차를 최소화하는 최적의 직선을 찾는 것이 목표입니다.

## 1. 경사하강법

### 1.1 경사하강법

- **경사하강법이란**:
    
    - 경사하강법(Gradient Descent Algorithm)은 머신러닝의 최적화 알고리즘 중 하나로, 주어진 손실 함수에서 모델의 가중치와 바이어스의 최적의 값을 찾기 위해 사용됩니다.
- **가중치 $w$값에 따른 손실 함수**:
    
    - 여러 가중치 $w$값(예: -0.5, 0, 0.5, 1, 1.5)에 따른 손실 함수 $l(w, b)$의 계산을 통해, 가중치가 손실에 미치는 영향을 수식으로 표현합니다.
    - 이 과정에서 경사(기울기)의 개념이 등장하며, 이는 손실 함수의 최소값을 찾는 데 중요한 역할을 합니다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/9aff9bb8-ac32-48ea-8c86-483cc8a7986e/Untitled.png)
    
    - 현재 손실 값에 대한 기울기를 자동 미분을 사용하는 코드를 Pytorch에서는 `loss.backward()`로 구현됩니다.
- **경사의 수식 표현**:
    
    - 경사는 손실 함수 $l(w, b)$의 변화량을 가중치 $w$의 변화량으로 나눈 값으로 정의됩니다.
    
    $\frac{\partial l(w, b)}{\partial w} = \frac{1}{n} \cdot -2 \sum_{i=1}^{n} (t_i - y_i) \cdot x_i$
    
    - 유도
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/7ba32378-91c3-423d-9b9b-8b28bf35c11b/Untitled.png)
        
    - 이 수식을 통해 가중치 업데이트의 필요성을 설명합니다.
        
- **가중치 $w$ 값 업데이트**:
    
    - 학습률 $\alpha$를 적용하여, 경사에 따라 가중치를 업데이트하는 과정이 설명됩니다. $w^* = w - \alpha \cdot \frac{\partial l(w, b)}{\partial w}$
    - 이는 코드 예시와 함께 설명되며, PyTorch에서는 `optimizer.step()`으로 구현됩니다.
- 이전 단계에서 계산된 기울기를 초기화하는 코드 표현
    
    - Pytorch에서는 `optimizer.zero_grad()` 로 구현됩니다. ⇒ 이 단계를 하지 않으면 기울기값이 누적되어 계산되 문제가 생김
- 경사하강법 작동 원리
    
    - 임의의 가중치 𝑤값을 선택하고
    - 선택된 𝑤에서 직선의 기울기를 나타내는 미분 값 $\frac{\partial l(w, b)}{\partial w}$ 를 구한 후에
    - 미분 값 $\frac{\partial l(w, b)}{\partial w}$ 과 반대 방향으로 𝑤를 감소시켜 나가다 보면
    - 최종적으로 기울기가 0이 되는 것을 알 수 있음
    - 바이어스 𝑏 또한 같은 방식으로 최적의 바이어스를 찾을 수 있음 바이어스 𝑏를 구하는 수식 표현: $b^* = b - \alpha \cdot \frac{\partial l(w, b)}{\partial b}$
- 문제
    
    1. 전체 데이터 셋을 이용하여 $w$와 $b$를 구하기 때문에 데이터 셋이 대규모이면 계산비용이 매우 커진다.
    2. 전역 minima가 아닌 local minima에 빠질 수 있다.

### 1.2 확률적 경사하강법

- **필요성**:
    
    - 경사하강법은 모든 데이터를 사용하여 가중치와 바이어스를 업데이트하므로 계산이 정확하고 안정적이지만, 대규모 데이터셋에서는 비효율적입니다.
        
    - 확률적 경사하강법(Stochastic Gradient Descent, SGD)은 **각각의 데이터 포인트마다 오차를 계산하여 가중치와 바이어스를 업데이트하는 방식입니다.**
        
    - 이는 계산 비용을 줄이고, 로컬 미니마에서 탈출하여 더 나은 글로벌 미니마에 도달할 수 있도록 돕습니다.
        
    - 수식 표현
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/d55183b1-0728-4d9b-ba3e-2b83e8fec486/Untitled.png)
        
- **코드 표현**:
    
    - PyTorch에서 확률적 경사하강법을 사용하는 방법:
    
    ```python
    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    ```
    

### 1.3 에폭

- **에폭이란**:
    
    - 에폭(Epoch)이란 모델이 전체 데이터셋을 한 번 완전히 학습하는 과정을 의미합니다.
    - 동일한 데이터셋을 여러 번 학습함으로써 모델의 성능을 향상시킬 수 있지만, 에폭 수가 너무 많으면 과적합(overfitting)이 발생할 수 있습니다.
- **코드 표현**:
    
    - 에폭 수를 설정하고, 반복적으로 학습을 수행하는 코드:
    
    ```python
    num_epochs = 1000
    for epoch in range(num_epochs):
        y = model(x_tensor)
        loss = loss_function(y, t_tensor)
    ```
    
- 손실 값 문제 해결하기
    
    - 손실 값이 큰 이유는 여러 가지가 있을 수 있음
        1. 학습률이 너무 크면 모델이 최적의 가중치로 수렴을 못함 · 학습률을 낮추어 보기
        2. 데이터에 노이즈가 많거나 이상치가 존재할 경우 학습이 어려울 수 있음 · 데이터를 시각화하여 이상치를 확인하고 처리
        3. 에폭 수가 충분하지 않으면 모델이 수렴하지 않음 · 에폭 수 늘려보기

### 1.4 데이터 표준화

- **데이터 표준화의 필요성**:
    
    - 손실값이 큰 이유는 학습률, 데이터 노이즈, 에폭 수 등이 원인일 수 있습니다. 하지만 이를 개선해도 문제가 해결되지 않으면, 데이터 전처리 단계에서 표준화를 적용할 수 있습니다.
    - 표준화는 특징 변수와 목표 변수의 평균을 0, 분산을 1로 맞추어 학습을 안정화하는 방법입니다.
    - 데이터 산포도는 같지만 변수의 값이 변하게 됩니다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/51547389-ef47-4fa2-a64b-76af0e5b7db1/Untitled.png)
    
- **코드 표현**:
    
    - 데이터를 표준화하는 방법:
    
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x.reshape(-1, 1))
    
    ```
    

### 1.5 학습 정리

- **경사하강법**:
    - 머신러닝의 최적화 알고리즘 중 하나로, 주어진 손실 함수에서 모델의 가중치와 바이어스의 최적의 값을 찾기 위해 사용됩니다.
- **확률적 경사하강법**:
    - 각각의 데이터 포인트마다 오차를 계산하여 가중치와 바이어스를 업데이트하는 최적화 알고리즘입니다.
- **에폭**:
    - 모델이 전체 데이터셋을 한 번 완전히 학습하는 과정을 의미하며, 과도한 에폭 수는 과적합을 초래할 수 있습니다.
- **표준화**:
    - 특징 변수와 목표 변수의 차이가 클 때, 두 변수의 평균을 0, 분산을 1로 맞추어 학습을 안정화하는 전처리 방법입니다.

---

# Day 5

## 1. 선형 회귀 모델의 테스트

### 1.1 테스트

- **테스트 데이터**: 선형 회귀 모델은 주어진 트레이닝 데이터를 사용하여 특징 변수와 목표 변수 사이의 선형 관계를 학습합니다. 이를 바탕으로 테스트 데이터에 대해 예측을 수행하여 모델의 성능을 평가합니다.
    
- **테스트 데이터의 대응표**: 테스트 데이터는 트레이닝 데이터에 포함되지 않은 YearsExperience 데이터로 구성됩니다.
    
- **코드 예시**:
    
    ```python
    python코드 복사
    def predict_test_data(test_data):
        test_scaled = scaler_x.transform(test_data.reshape(-1, 1))
        test_tensor = torch.tensor(test_scaled, dtype=torch.float32).view(-1, 1).to(device)
    
        model.eval()
        with torch.no_grad(): # 기울기 계산을 중지하여 메모리 사용을 줄이고 예측 속도를 높힘
            predictions_scaled = model(test_tensor)
        predictions = scaler_t.inverse_transform(predictions_scaled.cpu().numpy())
        return predictions
    ```
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/be5f30f1-5463-4e8b-adf8-574f1be60b8f/Untitled.png)
    

### 1.2 학습 정리

- **정리**: 테스트란 트레이닝 데이터에 포함되지 않은 새로운 데이터의 **결과를 연속적인 숫자 값으로 예측하는 과정입니다.**

---

## 2. 이진 분류 모델

### 2.1 이진 분류의 의미

- **정의**: 이진 분류는 주어진 트레이닝 데이터를 사용하여 특징 변수와 목표 변수(두 가지 범주) 사이의 관계를 학습하고, 이를 바탕으로 새로운 데이터를 **두 가지 범주 중 하나로 분류하는 모델을 구축하는 과정입니다.**
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/5fa13b5a-4bb8-4615-8599-d5ebe3868e35/Untitled.png)
    

### 2.2 트레이닝 데이터

- **트레이닝 데이터 구성**: 이진 분류 모델 학습을 위한 데이터는 특징 변수와 이진 목표 변수(0 또는 1)로 구성됩니다.
    
- **코드 예시**:
    
    ```python
    # 데이터 불러오기
    df = pd.read_csv("Iris.csv", sep=",", header=0)[["PetalLengthCm", "Species"]]
    # 목표변수가 2가지만 가지도록 설정
    filtered_data = df[df['Species'].isin(['Iris_setosa', 'Iris-versicolor'])]
    # 목표변수를 이산형 레이블로 변환
    filtered_data['Species'] = filtered_data['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})
    # 데이터 표준화를 진행할 때 2차원을 요구해 2차원으로 설정
    x = filtered_data[['PetalLengthCm']].values
    # 데이터를 1차원 배열로 변환
    t = filtered_data['Species'].values.astype(int)
    ```
    
- **데이터 분할**: 데이터셋을 트레이닝 데이터와 테스트 데이터로 나눕니다.
    
- **데이터 표준화 및 변환**:
    
    - 자세히
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/4b8a8754-76f4-408b-b585-d8ec0cfce7b5/Untitled.png)
        
    
    ```python
    # 데이터 표준화 
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Tensor로 변환
    x_train = torch.tensor(x_train, dtype=torch.float32)
    # unsqueeze를 통해 2차원으로 변환 << 배치처리를 위해서는 [N, 1]이어야함
    t_train = torch.tensor(t_train, dtype=torch.float32).unsqueeze(1)
    ```
    

### 2.3 Dataset & DataLoader 클래스

- **Dataset & DataLoader 클래스**: PyTorch에서 Dataset과 DataLoader 클래스를 사용하여 데이터의 전처리와 배치(데이터를 처리하는 묶음 단위) 처리를 용이하게 할 수 있습니다.
    
- **미니 배치 경사하강법** 경사하강법과 확률적경사하강법의 장점과 단점을 보완한 기법, 작은 배치 단위로 가중치를 업데이트한다.
    
- **코드 예시**:
    
    ```python
    python코드 복사
    from torch.utils.data import DataLoader, TensorDataset
    
    class IrisDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
    
        def __len__(self):
            return len(self.features)
    
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    train_dataset = IrisDataset(x_train, t_train)
    test_dataset = IrisDataset(x_test, t_test)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    ```
    

### 2.4 이진 분류 모델

- **로지스틱 회귀 알고리즘**: 로지스틱 회귀는 데이터를 잘 구분할 수 있는 최적의 결정 경계를 찾고**(1)**, 시그모이드 함수를 통해 이 경계를 기준**(2)**으로 데이터를 이진 분류하는 알고리즘입니다.**(3)**
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/9bd86b3e-9dd3-4128-9660-0d5c04a3bc26/Untitled.png)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/d1335cfd-30f6-473c-bc74-673f2981f1ac/Untitled.png)
    
- **코드 예시**:
    
    ```python
    class BinaryClassificationModel(nn.Module):
        def __init__(self):
            super(BinaryClassificationModel, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.sigmoid = nn.Sigmoid()
    
        def forward(self, x):
            z = self.linear(x)
            y = self.sigmoid(z)
            return y
    ```
    

### 2.5 학습 정리

- **정리**: 이진 분류는 트레이닝 데이터를 사용하여 두 가지 범주 중 하나로 데이터를 분류하는 모델을 구축하는 과정입니다. PyTorch에서는 데이터의 전처리와 배치 처리를 용이하게 할 수 있도록 Dataset과 DataLoader 클래스를 사용합니다. 로지스틱 회귀는 이진 분류 모델의 중요한 알고리즘으로 널리 사용됩니다.

## 1. 이진 교차 엔트로피

### 1.1 이진 교차 엔트로피

- **이진 분류의 의미 (Review)**:
    
    - 이진 분류는 주어진 트레이닝 데이터를 사용하여 특징 변수와 목표 변수(두 가지 범주) 사이의 관계를 학습하고, 이를 바탕으로 새로운 데이터를 두 가지 범주 중 하나로 분류하는 모델을 구축하는 과정입니다.
- **이진 교차 엔트로피**:
    
    - 이진 분류 모델에서 최종 출력 값인 $y$는 시그모이드 함수에 의해 0과 1 사이의 연속적인 값을 가지므로, 이 값을 0 또는 1로 분류하기 위해 이진 교차 엔트로피(Binary Cross Entropy, BCE)라는 손실 함수를 사용합니다.
    - 이진 교차 엔트로피는 이진 분류 문제에서 모델의 예측 변수와 목표 변수 간의 차이를 측정하기 위해 사용되는 손실 함수입니다.
- **이진 교차 엔트로피의 수식 표현**: $t_i$= 0 or 1을 가지는 label이다.
    
    - 이진 교차 엔트로피는 다음과 같은 수식으로 표현됩니다:
    
    $E(w, b) = -\sum_{i=1}^{n} \left[ t_i \log(y_i) + (1 - t_i) \log(1 - y_i) \right]$
    

### 1.2 이진 교차 엔트로피 유도를 위한 사전 개념

- **조건부 확률**:
    
    - 조건부 확률은 사건 A가 발생한 상황에서 다른 사건 B가 발생할 확률을 의미합니다.
    - 수식으로는 다음과 같이 표현됩니다:
    
    $P(B|A) = \frac{P(A \cap B)}{P(A)}$
    
- **가능도 함수**:
    
    - 가능도는 주어진 데이터가 특정 모수 값 하에서 관찰될 확률을 의미합니다.
    - 여러 개의 데이터가 있을 때 가능도 함수는 각 데이터의 가능도를 곱하여 계산하며, 이를 수식으로 표현하면 다음과 같습니다:
    
    $L(\theta; X) = P(x_1|\theta) \cdot P(x_2|\theta) \cdot \dots \cdot P(x_n|\theta)$
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/8fe97817-f35c-4a23-a8a4-3c4fbad77464/Untitled.png)
    
- **최대 가능도 추정(MLE)**:
    
    - **최대 가능도 추정은 주어진 데이터셋에 대해 모수를 추정하는 방법론으로, 데이터를 가장 잘 설명하는 모수를 찾기 위해 가능도(likelihood) 함수를 최대화하는 과정입니다.**
    - 가능도 함수의 최댓값을 찾아야하는데 이때 함수의 미분계수가 0이 되는 지점을 찾으면 된다. 이때 가능도함수를 로그 가능도 함수로 바꾸어 편미분을 계산한다.
    - MLE를 통해 실제로 계산된 구체적인 모수 값을 최대 가능도 추정치라고 합니다.
- **로그 가능도 함수**:
    
    - 확률 값의 범위를 확장해 데이터 비교가 용이해집니다. ex) [0,1] → (-∞, 0]
        
        - 그래프
            
            ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/44ac3366-fce9-4ad5-a603-8ef25f1f5b33/Untitled.png)
            
    - 로그를 사용하여 가능도 함수를 변환하면 곱셈을 덧셈으로 변환할 수 있고, 연산의 복잡성을 줄일 수 있습니다.
        
    - 데이터 숫자 단위를 줄일 수 있다.
        

### 1.3 이진 교차 엔트로피 유도

- **유도 과정**: 이진 분류 모델의 출력 값 $y$는 시그모이드 함수를 통해 0과 1 사이의 값을 가지며, 이를 기반으로 확률변수 $T$를 사용하여 출력 값을 표현할 수 있습니다.
    
- 수식으로 표현하면 다음과 같습니다: $t$ = 0 or 1 $P(T = t|x) = y^t \cdot (1 - y)^{1-t}$
    
- 이 수식을 기반으로 여러 개의 데이터에 대한 로그 가능도 함수를 계산하고, 이를 최소화하면 이진 교차 엔트로피 손실 함수를 도출할 수 있습니다:
    
    $E(W, b) = -\sum_{i=1}^{n} \left[ t_i \log(y_i) + (1 - t_i) \log(1 - y_i) \right]$
    
- **코드 예시**:
    
    ```python
    loss_function = nn.BCELoss()
    ```
    

### 1.4 학습 정리

- **요약**:
    - 이진 교차 엔트로피는 이진 분류 문제에서 모델의 예측 변수와 목표 변수 간의 차이를 측정하기 위해 사용되는 손실 함수입니다.
    - 조건부 확률과 최대 가능도 추정은 이진 교차 엔트로피 수식을 이해하는 데 필요한 중요한 개념입니다.

---

## 2. 이진 분류 모델의 테스트

### 2.1 테스트

- **테스트 데이터**:
    
    - 이진 분류 모델은 트레이닝 데이터에 포함되지 않은 PetalLengthCm 데이터를 사용하여 모델의 성능을 평가합니다.
- **코드 예시**:
    
    ```python
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
      # 모델의 출력 값이 0.5넘으면 1, 아니면 0
    	predicted_labels = (predictions > 0.5).float()  
    
    actual_labels = t_test.numpy()
    predicted_labels = predicted_labels.numpy()
    ```
    
- **결과 시각화**:
    
    - 예측 결과와 실제 라벨을 시각화하여 모델의 성능을 시각적으로 평가할 수 있습니다.
    
    ```python
    plt.scatter(range(len(actual_labels)), actual_labels, color='blue', label='Actual Labels')
    plt.scatter(range(len(predicted_labels)), predicted_labels, color='red', marker='x', label='Predicted Labels')
    plt.legend()
    plt.show()
    ```
    

### 2.2 학습 정리

- **요약**:
    - 테스트란 트레이닝 데이터에 포함되지 않은 새로운 데이터를 사전에 정의된 두 가지 범주 중 하나로 분류하는 모델의 성능을 평가하는 과정입니다.