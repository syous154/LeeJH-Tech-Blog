---
{"dg-publish":true,"permalink":"/3-paper-review/se-net-2017/","tags":["Paper"],"created":"2025-02-26T15:44:19.138+09:00","updated":"2025-01-08T19:51:32.534+09:00"}
---

# 1. 소개

합성곱 신경망(CNN) 아키텍처의 개선을 위한 "Squeeze-and-Excitation (SE) 블록" 제안한다.

- **SE 블록**: 컨볼루션 특징의 채널 간 상호 의존성을 모델링하여 네트워크의 표현력을 향상시킨다.
- **구성 요소**:
    - **Squeeze 단계**: 채널 설명자(채널별 중요도) 생성한다.
    - **Excitation 단계**: 채널별 가중치 적용한다.

**장점**

- 기존 CNN 아키텍처(SOTA급)에 쉽게 통합 가능하다.
- 계산적으로 가벼워 모델 복잡도를 크게 증가시키지 않는다.

**결과 및 성과**

- ImageNet 데이터셋에서 SE 블록을 사용한 SENet의 우수한 성능 검증
- ILSVRC 2017 분류 대회 1위 달성

# 2. 기존 방법의 문제점(SE 블럭이 없는)

![Pasted image 20250108193547.png](/img/user/images/Pasted%20image%2020250108193547.png)

위 그림처럼 여러 채널을 통해 여러 정보를 얻을 수 있다. 여러 정보를 확인해보면 1, 2번째는 안중요하고 3번째 채널은 중요한 것을 알 수 있다.

이러한 상황이면 1, 2번째 채널의 비중을 줄이고 3번째 채널의 비중을 늘리는 것이 **유리할 것이다**.

하지만 기존 CNN은 이러한 기능이 없었으며 “SE 블럭”이 이 기능을 추가한다.

# 3. 구조

전체적인 구조
![Pasted image 20250108193614.png](/img/user/images/Pasted%20image%2020250108193614.png)


## 3.1 Squeeze: 정보 압축 단계

![Pasted image 20250108193708.png](/img/user/images/Pasted%20image%2020250108193708.png)
z 는 각 채널의 값을 의미함

각 채널별 중요도를 확인하기위해 **글로벌 평균 풀링**을 사용합니다.

이는 각 채널의 값들을 평균을 내서 각 채널의 전반적인 값의 크기를 구합니다.

Input의 크기 (H,W,C)이라면 글로벌 평균 풀링을 진행하면 (1, 1, C)의 크기로 줄어들게 됩니다.

## 3.2 Excitation: 중요도 계산 단계

![Pasted image 20250108193738.png](/img/user/images/Pasted%20image%2020250108193738.png)

s 는 최종 채널별 가중치, W1과 W2는 FC Layer를 의미하고 δ는 ReLU σ는 Sigmoid이다.

Squeeze 단계에서 집계된 정보는 어떤 채널이 중요한지에 대한 정보는 반영되있지 않은 상태입니다.

따라서 이를 반영하기 위해 학습이 진행되어야합니다. 이를 Fully Connected -> ReLU -> Fully Connected -> Sigmoid 순서로 구성하여 학습합니다.

마지막 활성화 함수로 Sigmoid를 사용해 0~1 사이의 값을 가져 채널별 중요도(Attention Score)를 사용할 수 있도록 합니다.

## 3.3 Scale: 중요도 적용
![Pasted image 20250108193803.png](/img/user/images/Pasted%20image%2020250108193803.png)
이전의 연산을 통해 얻은 채널별 중요도(=$S_c$)를 $U_c$와 곱하여 $U_c$를 재보정 해준다.

이 연산을 통해 중요한 채널의 값은 유지하고 안 중요한 채널의 값은 무시하며 학습이 진행되도록 합니다.

![Pasted image 20250108193825.png](/img/user/images/Pasted%20image%2020250108193825.png)
ResNet에 적용한 SE 블럭의 모습이다. 여기서 처음 나타나는 r의 값은 FC 구조가 BottleNeck 구조를 가지며 이떄 사용하는 축소 비율의 값이다.

# 4. 모델 및 계산 복잡도

SE 블록을 적용하고 안하고의 계산 복잡도 차이를 비교하기위해 ResNet-50과 SE-ResNet-50을 예시로 사용합니다.

### **계산 비용 비교**

- **ResNet-50**: 224 × 224 입력 이미지에 대해 단일 순방향 패스 시 약 3.86G FLOPs 필요.
- **SE-ResNet-50**: 동일한 입력에 대해 약 3.87G FLOPs 필요, 이는 ResNet-50 대비 0.26%의 계산 비용 증가.

### **실행 시간 비교**

- **ResNet-50**: 순방향 및 역방향 패스에 약 190ms 소요.
- **SE-ResNet-50**: 약 209ms 소요(256개의 이미지 미니배치, 8개의 NVIDIA Titan X GPU 사용).
- **CPU 추론 시간**: ResNet-50은 164ms, SE-ResNet-50은 167ms.
- **결론적을으로 추가 비용은 미미합니다.**

### 추가적인 매개변수와 모델 용량

- ResNet-50과 SE-ResNet-50의 추가적인 매개변수 차이는 두개의 FC 계층에서만 발생합니다. 이는 전체적인 모델 용량에서 아주 작은 부분을 차지하며 약 10% 증가한 용량을 보여줍니다.

# 5. 성능평가

SE 블럭은 CNN 모델 중간에 사용될 수 있다고 했습니다. 따라서 image Classification 뿐만 아니라 Scene Classification, Object Detection에서 성능 평가를 진행했습니다.

## 5.1 Image Classification

![Pasted image 20250108194006.png](/img/user/images/Pasted%20image%2020250108194006.png)
Imagenet 데이터셋 실험 결과

Imagenet 데이터셋 실험 결과를 보면 SE를 적용한 모델이 약간의 연산량 증가를 보이고 더 낮은 Error율을 보이는 것을 확인 할 수 있습니다.

![Pasted image 20250108194024.png](/img/user/images/Pasted%20image%2020250108194024.png)
CIFAR-10, CIFAR-100 데이터셋 실헙결과 (Error% 비교)

다른 두 가지 데이터셋을 사용하여 실험한 결과 SE 블럭을 추가한 모델의 Error율이 더 낮은 것을 확인할 수 있습니다.

## 5.2 Scene Classification

![Pasted image 20250108194036.png](/img/user/images/Pasted%20image%2020250108194036.png)
Places365 데이터 셋 실험결과

장면 분류 데이터셋을 사용하여 실험한 결과를 보면 SE 블럭을 추가한 모델의 성능이 더 좋은 것을 확인할 수 있습니다.

## 5.3 Object detection

![Pasted image 20250108194051.png](/img/user/images/Pasted%20image%2020250108194051.png)
COCO 데이터셋 실험 결과

객체 탐지 데이터셋을 사용하여 실험한 결과를 보면 SE 블럭을 추가한 모델의 성능이 더 좋은 것을 확인할 수 있습니다.

**여러 문제에서 적용한 결과를 보면 문제에 관계없이 기존 CNN 구조에 SE 블럭을 추가하면 성능이 개선됨을 알 수 있었습니다.**