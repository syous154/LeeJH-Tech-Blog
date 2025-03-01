---
{"dg-publish":true,"permalink":"/2-naver-boost-camp/week-3/","created":"2025-02-26T15:44:18.973+09:00","updated":"2025-01-08T20:18:27.713+09:00"}
---

[[2. NaverBoostCamp/Week 4 학습 정리\|Week 4 학습 정리]]
# Day 1

> 데이터와 획일화된 테크닉은 정답을 알려주지 않는다. ⇒ 관찰과 해석으로 본인만의 관점이 중요하다.

- 데이터 분석 시 장애물
    
    1. 데이터로 할 수 있는게 뭘까?
    2. 한정된 리소스로 해결 할 수 있나?
    3. 테이터가 만능인가?
- 데이터 문해력 : **데이터를 읽고 이해한 것을 바탕으로 분석 결과를 전달하는 능력**
    
    1. 좋은 질문을 할 수 있는 역량 ⇒ 전처리를 어떻게 할 것 인지, 어떤 모델을 쓸건지 등
        
    2. 필요한 데이터를 선별하고 검증할 수 있는 역량 ⇒ 데이터를 보간, 제거, 이상치 등을 어떻게 사용할 것인지 등
        
    3. 데이터 해석 능력을 기반으로 유의미한 결론을 만들어내는 역량
        
    4. 가설 기반 A/B테스트*를 수행하여 결과를 판별할 수 있는 역량
        
        - A/B 테스트*
            
            웹 사이트 방문자를 임의로 두 집단으로 나누고, 한 집단에게는 기존 사이트를 보여주고 다른 집단에게는 새로운 사이트를 보여준 다음, 두 집단 중 어떤 집단이 더 높은 성과를 보이는지 측정하여, 새 사이트가 기존 사이트에 비해 좋은지를 정량적으로 평가하는 방식
            
    5. 의사결정자들도 이해하기 쉽게 분석 결과를 표현할 수 있는 역량
        
    6. 데이터 스토리텔링을 통해 의사결정자들이 전체그림을 이해하고 분석 결과에 따라 실행하게 하는 역량
        

> 데이터 시각화란? ⇒ 무조건적인 데이터의 양보다 인간의 **지각 능력, 인지 능력**을 균형적으로 사용하여 효과적으로 이해 가능하도록 함

- 정보와 데이터는 비슷하지만 다르다.
    
    데이터 : 현실 세계에서 측정하고 수집한 사실이나 값
    
    - 데이터 수집 시에는 가정과 목표를 가지고 진행
    
    정보 : 어떠한 목적이나 의도에 맞게 데이터를 가공 처리한 것
    
    - 정보는 텍스트로 옮기는 과정에서 손실이 발생할 수 있고 문맥에 따라 다르게 해석이 가능함
- 데이터 시각화
    
    1. Expressiveness : 데이터가 가진 정보를 시각 요소로 모두 표현
    2. Effectiveness : 중요한 정보가 부각되어 표현
- Mark : 데이터를 나타내기 위한 빈 공간(베이스) ⇒ 점, 선, 면으로 이루어짐
    
- Channel : 빈 공간(베이스)에 수치값을 넣는 역할 ⇒ 각 Mark를 변경하는 요소
    
- 시각화 5가지 원칙
    
    Accuracy:정확도.데이터의 값이 정확하게 표현되어야 함 ­ Discriminability:구별 가능성.채널 내 값에 대한 구분 ­ Separability:분리성.시각적 채널간 상호작용에 대한 구분 ­ Popout:시각적 대비.채널을 통한 데이터 구분이 명확해야 함 ­ Grouping:그룹화.유사한 것은 그룹을 통해 쉽게 인지 가능
    
- Popout : 시각적으로 다른 label을 가진 데이터를 분리 ⇒ 한눈에 이해 가능
    
- 전주의적 속성 : 따로 생각하지 않아도 한눈에 보자마자 이해가능한 속성
    
- principle of proportion ink : 실제 값과 그래프로 나타나는 값의 비율이 같아야함, 반드시 x축의 시작은 0!
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/c930d07c-c2c9-4577-ab86-bf52bdeb7ffc/image.png)
    
- 쓸데 없이 복잡하게 할 필요는 없음
    
- Overplotting : Scatter를 그릴 때 점이 많아질 수록 분포를 한눈에 이해하기 힘듦 ⇒ 2차원 히스토그램, Contour plot을 추천
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/bff80673-0717-408f-af6a-71be4b6792d3/image.png)
    
- 인과 관계와 상관 관계는 비슷한 느낌이지만 다르다.
    
    - 인과 관계 : 특정 요인 A가 B에 영향을 준다.
    - 상관 관계 : 특정 요인 A와 B가 연관성이 있다.
- Text 추가 ⇒ 생길 수 있는 오해를 방지할 수 있음(제목, label, tick label, legend, Annotation 등등)
    
- Color
    
    - 범주형 Color palette : 독립된 색상으로 구성됨 ⇒ 번주형 변수에서 사용, 색의 차이로 구분
    - 연속형 Color palette : 정렬된 값을 가지는 순서형, 연속형 변수에 적합 ⇒ 균일한 색상 변화가 중요
    - 발산형 Color palette : 연속형과 비슷하지만 중앙을 기준으로 발산, 양 끝으로 갈수록 색이 진해짐 ⇒ 상반된 값, 서로다른 2개(ex 지지율)를 표현하는데 적합

# Day 2

## 데이터

- 범주형 데이터 ⇒ 순서형, 명목형
    
    - 순서형 : 값의 순서가 존재
    - 명목형 : 값의 순서가 없음
    - 순서형인지 수치형인지가 고민된다면 산술연산의 의미를 두면 된다.(별점 4점은 별점 2점보다 2배 좋은건가?) 경우에 따라 순서형도 수치형처럼 치환하여 계산해볼 수 있다.(평균 별점)
- 범주형 - 집단 간 분석에 사용하기 좋음, 각 집단의 대푯값을 이용
    
- 명목형 - 일반적으로는 값이 텍스트로 구분 ⇒ 학습에 이용하려면 숫자로 바꾸어줘야함 ⇒ Encoding
    
    - Label Encoding - 값에 1,2,3,… 으로 치환 → 순서형에는 적합, 명목형에는 안좋음
    - One Hot Encoding - 여러개의 열을 만들어 [1, 0 0] 식으로 치환 → 순서 정보 X, 데이터가 커져 학습 속도나 성능에 악영향
    - Binary Encoding - 레이블 인코딩 후 2진수로 치환 → 범주의 의미가 거의 사라짐
    - Embedding:자연어 처리에 있어 적절한 임베딩 모델을 사용하는 것도 하나의 방법
    - Hashing:랜덤 해시 값으로 순서정보를 없앨 수도 있음
    - 특정 값에 따른 인코딩:해당 범주가 가진 통계값의 사용 (예시로 빈도수로 한다면 여자가 30명,남자가 70명이라면 =>[여자,남자]=>[30,70])
- 순서형 - 앞의 인코딩 방법 모두 사용 가능
    
    - 순환형 데이터 : 순서가 있지만 계속 반복되는 데이터 ⇒ 요일, 월, 각도, sin함수, cos함수 등
- 수치형 데이터 ⇒ 이산형(정수), 연속형(실수) or 구간형, 비율형
    
    - 구간형 데이터 ⇒ 값들 간의 차이가 일정한 데이터입니다. 즉, 연속적인 수치 사이의 간격이 동일, 절대 영점X → 온도, 시간 등
    - 비율형 데이터 ⇒ 비율형 데이터는 절대 영점이 존재하며, 수치 간의 비율 계산이 가능한 데이터 → 인구수, 횟수, 밀도 등

> 집단의 대푯값을 사용해도 언제나 잘못된 정보를 얻을 수 있음을 인지해야함

## 전처리

- 정규화 - 데이터를 특정 범위로 변환

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/42c3eefd-1bf3-4210-b809-361ae6887723/image.png)

- 표준화 - 평균 0, 표준편차 1로 만들어 표준정규분포로 만듦

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/c5838ef9-d73d-4113-aa63-238ce0fe2f84/image.png)

- NegativeSkewness (오른쪽에 치우처진 데이터라면?)
    
    1. Square/PowerTransformation:제곱 변환 또는 거듭 제곱
    2. ExponentialTransformation:지수 함수
    
    - 부호에 유의해서 사용해야함
- PositiveSkewness (왼쪽에 치우쳐진 데이터라면?)
    
    1. LogTransformation:로그
    2. Square-rootTransformation:제곱근
    
    - (1)0이상 실수 (2)양수 라는 조건 필요
    
    1. Box-CoxTransformation:범용적인 LogTransformation변형 방법
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/b0701ace-0b00-4675-bc07-30b591fa785a/image.png)
        

## 결측치

말그대로 비어있는 값을 나타냄 ⇒ 이 값을 어떻게 처리하느냐에 따라 성능이 달라질 수 있음

아래 커맨드를 통해 결측치를 시각화 할 수 있는 라이브러리 설치가능

```
pip install missingno
```

1. 결측치가 과반수 이상 ⇒ 결측치 유무만 사용하거나, 결측치가 있는 열을 삭제
    
2. 결측치가 유의미하게(>5%) 많은 경우 ⇒ 유의미하기 때문에 결측치를 채우려는 대책 마련
    
3. 결측치의 개수가 매우 적은 경우 ⇒ 결측치가 있는 행을 삭제, 대푯값으로 채움
    
4. 규칙 기반 – 도메인 지식이나 논리적 추론에 의한 방법 – 지나치게 복잡하거나 단순한 경우 잘못된 편향이 반영될 수 있음
    
5. 집단 대푯값 – 특정 집단의 대푯값(평균,중앙값,최빈값)등 사용 – 집단 설계가 중요하고 이상치에 민감할 수 있음
    
6. 모델 기반 – 회귀모델을 포함한 다양한 모델을 통해 예측 – 복잡한 패턴을 예측할 수 있으나 과적합 이슈 발생 가능
    

## 이상치

데이터의 범위에서 과하게 벗어난 값 → 기궂은 없음

대표기준 : IQR, 표준편차, z-score

그외 : DBSCAN, Isolated Forest

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e473be93-4d55-4a8c-9441-dc90b08277d4/image.png)

IQR - IQR=3분위수와 1분위수의 차이

1분위수 - 1.5 * IQR 이상

3분위수 ­ 1.5 * IQR 이상

- Boxplot을 그리면 한번에 확인 가능
- DBSCAN: 밀도기반으로 클러스터링
- Isolated Forest: 결정 트리를 이용해 그룹을 분리

## Feature Engineering

- 특성 추출 ⇒ 기존의 특성으로 새로은 특성을 만듦
- 특성 선택 ⇒ 기존의 특성 중 중요한 특성들만 고름

> 두 가지 모두 도메인 지식이 더 중요할 수 있음

## Clustering

: 유사한 성격을 가진 데이터를 그룹으로 분류

- K-Mean:그룹을 K개로 그룹화하여,각 클러스터의 중심점을 기준으로 데이터 분리
    
- HierarchicalClustering:데이터를 점진적으로 분류하는 방법
    
- DBSCAN:밀도 기반 클러스터링
    
- GMM:가우시안 분포가 혼합된 것으로 모델링
    
- 차원축소(dimensionreduction)란 특성 추출 방법 중 하나로 데이터의 특성(feature)N개를 M개로 줄이는 방법 ⇒ 가까운 데이터 더 가깝게, 먼 데이터는 더 멀게
    
    - 데이터의 복잡성 감소:고차원 데이터에서
    - 시각화:패턴 발견에 용이 =>클러스터링과 매우 밀접한 관련
    - 모델 성능 향상
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/8add295c-3b8d-4224-932b-dd860d1eba2a/image.png)
    

# Day 3

## 시계열 데이터

: 하나의 변수를 시간에 따라 여러 번 관측한 데이터

- 특징
    
    1. 추세: 장기적인증가또는감소
    2. 계절성: 특정 요일/계절에 따라 영향
    3. 주기: 고정된 빈도가 아니지만 형태적으로 유사하게 나타나는 패턴
    4. 노이즈: 왜곡
- 성분분석: 추세, 계절성, 주기, 노이즈를 통해 시계열을 분석
    
    - 가법모델(additivemodel): 추세 + 계절성 + 주기 + 노이즈
    - 승법모델(multiplicativemodel): 추세 * 계절성 * **주기 + 노이즈
    - 시간에 따라 변동폭이 비교적 일정하다면 **가법모델**, 변동폭이 커진다면 **승법 모델** 사용
- 정상성: 시간에 따라 **통계적 특성이 변하지 않음**
    
- 비정상성: 시간에 따라 **통계적 특성이 변함**
    
- 통계 모델에 사용하려면 비정상성을 없애야함
    
    - 차분(Differencing): 이웃된 두 값의 차이값을 사용
        - [1,2,4,7,8,10]=>[None,1,2,3,1,2]
        - 경우에 따라 2차 차분도 가능
        - 바로 직전 데이터가 아닌 **계절성 주기에 따라 차이를 두는 계절성 차분도 존재**
    - 로그연산: 로그연산을 통해 비정상성일부 제거가능
- **평활(Smooting)** ⇒ 불필요한 변동을 제거하여 쉬운 해석을 도움
    
    - 추세 밒 계절성 파악
    - 데이터 시각화 개선

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/aba8497f-6bc0-4110-9fc3-29290a5f216f/image.png)

- 평활 방법
    
    구간별 평균, 구간별 통계 사용, 이동평균→ 이상치에 영향이 존재 ⇒가중 이동 평균, 지수 이동 평균
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/cc31713b-5a1d-43c5-a941-72c9504a8f5c/image.png)
    
    급격하게 변하는 값에 대해서도 추세를 보기 쉬움 다만 구간 길이에 따라 다르게 해석될 수 있으니 주의
    

## 이미지 데이터

- 도메인: 데이터가 어디서 왔는지 (분야)
- Task: 분류, 객체 탐지, 세그멘테이션 등등
- 퀄리티: 그에 맞는 적절한 데이터 셋

데이터 수집과정과 Fine Tuning을 위한 데이터 전처리도 중요

- EDA 진행 순서
    1. Target 중심: 이미지의 상태, 데이터 셋의 상태 분석 후 조치
    2. Input 중심: 이미지 데이터 개별 비교(도메인 지식)
    3. Process 중심: (전처리-모델-결과해서) 반복

### Color space

- RGB: 빛의 삼원색으로 표현, 컴퓨터 그래픽에서 사용
- HSV: 색상, 채도, 명도롤 표현, 인간의 색인지와 유사 RGB 이미지나 HSV 이미지나 모델 성능에는 큰 영향이 X
- CMY(K): RGB의 보색 이용, 인쇄에 용이
- YCbCr: 밝기/ 파랑에 대한 색차/ 빨강에 대한 색차로 구분하여 디지털영상에 용이

### 이미지 포맷

- JPEG(JoinPhotographicExpertGroup): 손실 압축 방법론.웹게시용에 사용
    - 색상공간을 YCbCr을 사용하며, 양자화를 통해 일부 손실 압축
- PNG(PortableNetworkGraphics): 무손실압축방법.
    - 투명도를 포함할 수 있음, 투명도에 의해 4차원일 수 있음

### 이미지 데이터 전처리

- 색상공간(ColorSpace) :RGB, HSV, Grayscale
- 노이즈삽입(Noise)
- 사이즈조정(Resizing): Crop & Interpolation
- 아핀변환(AffineTransformation): 회전, 왜곡, 평행이동 등
- 특성추출(FeatureExtraction): SIFT, SURF, ORB, FAST등

### 이미지 데이터 라이브러리

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/6eb10c33-1d5b-47a8-b73c-db6ffa449f6b/image.png)

# Day 4

### **1. Polar Plot (극좌표 플롯)**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e2124d31-3d2d-4f6e-9e0c-18fe4ca6f987/image.png)

- **Polar Plot**: 극좌표계를 사용하여 데이터를 시각화하며, 회전이나 주기성을 표현하기에 적합합니다.
- **데이터 변환**: 직교 좌표계에서 극좌표계로 변환할 수 있으며, `X = R cos θ`, `Y = R sin θ`의 식을 사용합니다.

### **2. Radar Chart (레이더 차트)**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/36e04475-03c5-4f5d-a5d0-c37da4e05244/image.png)

- **Radar Chart**: 중심점을 기준으로 N개의 변수 값을 표현하는 차트로, 개별 데이터 분석에 용이합니다.
- **주의점**: 모든 변수의 척도가 동일해야 하며, 변수의 순서에 따라 차트의 모양이 달라질 수 있습니다. 변수의 수가 많아지면 가독성이 떨어질 수 있습니다.

### **3. Pie Chart (파이 차트)**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/e14ffbd2-91a5-48e7-b2f3-bb5139458b42/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/47168b10-6fac-445f-86ee-a4dc938da6e7/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/42bb5e44-2b81-4fc6-b054-3e76e74f7f0d/image.png)

- **Pie Chart**: 원을 부채꼴로 분할하여 전체를 백분율로 표현합니다. 그러나 비교가 어려워 오히려 bar plot이 더 유용할 수 있습니다.
- **응용**: Donut Chart(도넛 차트), Sunburst Chart(선버스트 차트) 등이 있으며, 각각 중간이 비어있는 형태, 계층적 데이터를 표현하는 데 유용합니다.

### **4. Treemap (트리맵)**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/1ebfbdbe-87ca-417e-9a1c-6d602304722b/image.png)

- **Treemap**: 계층 데이터를 직사각형으로 표현하여 포함 관계를 나타내는 시각화 방법입니다. 타일링 알고리즘에 따라 다양한 형태로 표현될 수 있습니다.
- **Python 사용법**: `pip install squarify` 또는 Plotly의 treemap을 사용할 수 있습니다.

### **5. Waffle Chart (와플 차트)**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/fc85b680-01fe-4139-978c-5f2c86aae144/image.png)

- **Waffle Chart**: 와플 형태로 값을 나타내는 차트로, 인포그래픽에서 자주 사용됩니다. 정사각형뿐만 아니라 원하는 벡터 이미지로도 표현할 수 있습니다.
- **Icon을 사용한 Waffle Chart**: Pictogram Chart라고도 하며, 시각적 표현을 강조할 때 유용합니다.

### **6. Venn Diagram (벤 다이어그램)**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/7481d49f-7619-4568-bedb-4d1215b89a68/image.png)

- **Venn Diagram**: 집합 간의 관계를 시각적으로 표현하는 다이어그램으로, 출판이나 프레젠테이션에 주로 사용됩니다.
- **Python 사용법**: `pip install pyvenn` 또는 `pip install matplotlib-venn`을 통해 사용할 수 있습니다.

### **7. Facet (분할된 시각화)**

- **Facet의 개념**: 화면을 여러 개의 시각화로 분할하여 다양한 관점을 동시에 보여주는 기법입니다. 동일한 데이터셋에 대해 여러 인사이트를 제공하거나, 전체적으로 볼 수 없는 세부적인 부분을 시각화할 때 유용합니다.
- **Matplotlib에서의 구현**:
    - **Figure와 Axes**: Figure는 전체 틀을 의미하며, 각 subplot이 들어가는 공간을 Axes라고 합니다. 하나의 Figure에 여러 개의 Axes를 포함할 수 있습니다.
        
    - **NxM Subplots**: `plt.subplot()`, `plt.figure() + fig.add_subplot()`, `plt.subplots()` 등으로 쉽게 구현할 수 있으며, 크기, 해상도, 축 공유 등을 조정할 수 있습니다.
        
    - **Grid Spec 활용**: Grid 형태로 subplots을 만들고, slicing 또는 (x, y), dx, dy를 사용하여 subplot의 위치와 크기를 조정할 수 있습니다.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/0b7c77ea-1b87-4a66-8614-8d3391345e8b/image.png)
        
    - **내부에 그리기**: Axes 내부에 서브플롯을 추가하는 방법으로, 미니맵이나 외부 정보를 적은 비중으로 추가할 수 있습니다.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/4a773d2e-85b7-451b-8c25-e8486511e25e/image.png)
        

### **8. More Tips (추가 팁)**

- **Grid 이해하기**:
    - **기본 Grid**: 축과 평행한 선을 사용하여 거리 및 값 정보를 보조적으로 제공합니다. 보통 무채색을 사용해 다른 표현들을 방해하지 않도록 하며, 큰 격자와 세부 격자를 동시에 조정할 수 있습니다.
    - **다양한 타입의 Grid**: X+Y=C, Y=CX, 동심원을 사용하는 Grid 등, 데이터의 특성에 따라 다양한 Grid를 구현할 수 있습니다.
    - **구현 예시**: Python의 numpy와 matplotlib을 사용하여 다양한 Grid를 쉽게 구현할 수 있습니다.
- **보조 도구**:
    - **선 추가하기**: 특정 기준선(예: 평균선, 상한선)을 추가하여 시각화를 보완할 수 있습니다.
    - **면 추가하기**: 특정 범위(예: Netflix 영화 등급 분포)를 강조하기 위해 면을 추가할 수 있습니다.
- **Setting 바꾸기**:
    - **Theme**: 테마를 변경하여 시각화의 스타일을 조정할 수 있습니다.
    - **핵심 수정 사항**: 기본 색상, 텍스트 폰트, 크기와 위치, 테두리 제거 등을 통해 시각화를 세밀하게 조정할 수 있습니다.

### **1. UX와 HCI**

- **HCI(인간-컴퓨터 상호작용)**:
    - 인간과 디지털 기기 및 시스템 간의 상호작용을 연구하는 분야입니다.
    - HCI의 목표는 사용자에게 최적의 경험을 제공하는 것입니다.
- **HCI의 3요소**:
    - **유용성(Usefulness)**: 사용자가 하고자 하는 일을 효과적으로 달성할 수 있도록 해야 합니다.
    - **사용성(Usability)**: 시스템이 일반 사용자도 쉽게 이해하고 사용할 수 있어야 합니다.
    - **감성(Affect)**: 사용 과정에서 적절한 느낌을 제공하여 감성적인 만족을 제공해야 합니다.
- **심성 모형(Mental Model)**:
    - 특정 개념이나 사물에 대해 사용자가 인식하는 이해 구조입니다.
    - 예를 들어, 사용자는 전자레인지를 "버튼을 누르면 동작한다"는 구조로 인식합니다. 사용자가 쉽게 이해할 수 있는 방식으로 제품이나 알고리즘을 전달해야 합니다.

### **2. 다양한 HCI 이론**

- **Schneiderman's Mantra**:
    
    - Ben Shneiderman이 제안한 데이터 탐색 및 분석을 위한 3단계 접근 방식입니다.
        1. **Overview first**: 사용자가 데이터의 전체적인 개요를 먼저 볼 수 있어야 합니다.
        2. **Zoom and filter**: 사용자가 관심 있는 부분을 확대하고 불필요한 정보를 필터링할 수 있어야 합니다.
        3. **Details-on-demand**: 사용자가 필요할 때 세부 정보를 요청할 수 있어야 합니다.
- **게슈탈트 원리**:
    
    - 인간의 인지는 개별 요소들의 합이 아닌 전체적 구조에 기반합니다.
        
        1. **근접성의 원리**: 서로 가까운 요소들이 그룹으로 인식됩니다.
        2. **유사성의 원리**: 외관이 유사한 요소들이 그룹화됩니다.
        3. **연속성의 원리**: 연속적인 패턴을 가진 요소들이 하나의 경로로 인식됩니다.
        4. **폐쇄성의 원리**: 불완전한 형태도 완전한 형태로 인식하려는 경향이 있습니다.
        5. **그림-배경 분리**: 어떤 요소를 '그림'으로, 나머지를 '배경'으로 구분하는 능력입니다.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/b2c5ba24-e1c4-42bf-8366-4fa0b2855835/image.png)
        
- **인지편향**:
    
    - 정보 처리와 의사결정 시 발생하는 체계적인 사고 오류입니다.
        - **확증 편향**: 자신의 믿음을 뒷받침하는 정보만 선택적으로 수집하고 해석하는 경향.
        - **가용성 휴리스틱**: 최근 경험이나 쉽게 떠올릴 수 있는 정보를 기반으로 판단하는 경향.
        - **앵커링**: 최초에 제시된 정보에 지나치게 의존하여 판단하는 경향.
        - **프레임 효과**: 문제나 상황의 제시 방식에 따라 다른 결정을 내리는 경향.

### **3. 사용자와 데이터 분석**

- 사용자 퍼널(AARRR)
    - 유입(Acquisition)
    - 활동(Activation)
    - 재방문(Retention)
    - 구매(Revenue)
    - 추천(Referral)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/dd64a3b6-4f6d-4629-98ec-eefd7e6a74c4/173ff3fc-439e-453f-82b7-0bf740dec95e/image.png)

- **사용자 지표**:
    - **DAU**: 일별 활성 사용자 수.
    - **WAU**: 주간 활성 사용자 수.
    - **MAU**: 월간 활성 사용자 수.
    - **잔존율(Retention Rate)**: 재사용 비율, 고객 유지 비율.
    - **이탈률(Churn Rate)**: 서비스 종료 또는 경쟁사로 이동 비율.
    - **고착도(Stickiness)**: 반복적 방문 지표(DAU/MAU).
    - **신규 이용자 수**: 일정 기간 내 신규 유입된 이용자 수.
    - **동시 접속자 수(CCU)**: 서비스 동시 접속자 수.
    - **평균 체류 시간**: 서비스 방문자의 평균 체류 시간.
- **구매 및 매출 지표**:
    - **CAC(고객 확보 비용)**: 신규 이용자 확보 평균 비용.
    - **구매자 수(PU)**: 유료 결제 사용자 수.
    - **구매 전환율**: 전체 사용자 대비 유료 결제 사용자 비율.
    - **ARPU**: 사용자 당 평균 결제 금액.
    - **ARPPU**: 결제 사용자 당 평균 결제 금액.
    - **ARPDAU**: 일별 활성 사용자 당 평균 결제 금액.
    - **LTV(고객 생애 가치)**: 고객이 서비스 이용 기간 동안 발생시키는 순이익.
- **전환 지표**:
    - **CPM(Cost Per Mille)**: 1000회 노출 당 비용.
    - **CPC(Cost Per Click)**: 클릭 당 비용.
    - **CPI(Cost Per Install)**: 설치 당 비용.
    - **CPA(Cost Per Action)**: 행동 당 비용.

# Day 5

월욜에…..