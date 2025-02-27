---
{"dg-publish":true,"permalink":"/naver-boost-camp/week-6/","created":"2025-02-26T15:44:19.021+09:00","updated":"2025-01-08T20:21:33.720+09:00"}
---

[[NaverBoostCamp/Week 7 학습 정리\|Week 7 학습 정리]]
> [!NOTE]
> > **1. 소프트웨어엔지니어링과 AI엔지니어링
> > 2. Linux, 쉘 스크립트
> > 3. Streamlit을 활용한 웹프로토타입 구현하기
> > 4. 파이썬 환경 설정과 디버깅**

# 1. **소프트웨어엔지니어링과 AI엔지니어링**

> 소프트웨어 엔지니어링이란? ⇒ 소프트웨어를개발하는과정에서 체계적이고효율적인방법을사용하여 소프트웨어의품질과유지보수성을보장하는학문분야

- 좋은 소프트웨어 설계 시 필요한 개념
    - 모듈성: 큰 프로그램을 작고 독립적인 부분으로 나누는 것, 레고로 조립하듯이
    - 응집도: 하나의 모듈 안에서 서로 다른 함수가 엮이는 정도
    - 결합도: 모듈 간 상호의존성 정도

⇒ 높은응집도(모듈내교류)와느슨한결합도(모듈끼리덜교류)를가진소프트웨어를지향
![Pasted image 20250108201933.png](/img/user/images/Pasted%20image%2020250108201933.png)
- 테스트 ⇒ 프로그램이 예상대로 작동하고 문제가 없는지 확인하는 과정
    - Unit Test: 개별단위테스트
    - Integration Test: 다른단위, 구성요소 동작 테스트
    - End to End Test: 처음부터 끝까지 모두 테스트
    - Performance Test: 성능,부하 테스트

![Pasted image 20250108201948.png](/img/user/images/Pasted%20image%2020250108201948.png)

- 문서화

⇒ 소프트웨어를 이용하기 위한 README, API문서, 아키텍처 문서

좋은 소프트웨어는 좋은 문서가 존재 ⇒ 개인 프로젝트에서고 문서화를 신경 쓰면 좋음

한눈에 전체적인 소프트웨어를 확인할 수 있음

- AI 엔지니어링

소프웨어 엔지니어링과 AI 엔지니어링은 비슷해 보이지만 역활과 관심사가 다름을 인지하는 것이 좋음

![Pasted image 20250108202000.png](/img/user/images/Pasted%20image%2020250108202000.png)

![Pasted image 20250108202012.png](/img/user/images/Pasted%20image%2020250108202012.png)

# **2. Linux, 쉘 스크립트**

- Linux

⇒ 특히 서버에 관련된 직군이면 Linux를 다룰 일이 많음

- 쉘
    - sh - 최초의 쉘
    - bash - Linux 표준 쉘
    - zsh - Mac 카탈리나 OS 기본 쉘

⇒ 사용자가 문자를 입력해 컴퓨터에 명령할 수 있도록 하는 프로그램

- 쉘 UX

username@hostname:current_folder 형태로 존재

hostname: 컴퓨터 네트워크에 접속된 장치에 할당된 이름. IP 대신 기억하기 쉬운 글자로 저장

- 서버에서 자주 사용하는 쉘 커멘드

⇒ 굉장히 많기 때문에 강의 PPT를 확인하도록 함

# **3. Streamlit을 활용한 웹프로토타입 구현하기**

- 프로토 타입

⇒ AI 모델의 Input ⇒ Output을 확인할 수 있도록 설정

![Pasted image 20250108202026.png](/img/user/images/Pasted%20image%2020250108202026.png)

- Streamlit

⇒ AI 엔지니어가 Python을 사용하여 간단하고 쉽게 프로토 타입을 만들 수 있음

Streamlit Cloud가 존재해 배포도 쉽게 가능, 화면 녹화 기능을 제공해 프로토 타입 녹화 가능

![Pasted image 20250108202035.png](/img/user/images/Pasted%20image%2020250108202035.png)

Streamlit은 화면에서 무언가 업데이트되면 **전체 Streamlit 코드가 다시 실행됨**

ex 1) Code가 수정 되는 경우

ex 2) 사용자가 Streamlit의 위젯과 상호작용하는 경우 (버튼 클릭, 입력상자에 텍스트 입력 시등)

⇒ 이때 Session State를 사용하면 해결 가능

- Session State ⇒ Global Variable처럼 공유할 수 있는 변수를 만들고 저장
    
    - 사용하는경우
        - 상태를 유지하고 싶은 경우
        - 특정변수를 공유하고 싶은 경우
        - 사용자 로그인 상태 유지
        - 채팅에서 대화 히스토리 유지
        - 여러단계의 Form
- Streamlit Caching
    

⇒ 캐싱을 사용해 **함수 호출 결과를 Local에 저장**해, 앱이 더빨라짐, 캐싱된 값은 모든 사용자가 사용할 수 있음. **반면, 사용자마다 다르게 접근해야 하면 Session State에 저장 하는 것을 추천**

> Streamlit을 개발하는 과정에서 제일 어려운 Session State에 대해 꼭 이해하기

# **4. 파이썬 환경 설정과 디버깅**

내가 사용하고자 하는 라이브러리와 호환되는 파이썬 버전은 [Pypi](https://pypi.org/)에서 라이브러리를 검색해 확인 가능

- 파이썬 설치 방법
    - 파이썬 공식 홈페이지에서 파일을 다운받아 설치
    - Conda로 설치
    - Docker로 파이썬 이미지 설치
    - 패키지 관리자(brew, apt, winget)로 설치
    - pyenv로 설치

![Pasted image 20250108202047.png](/img/user/images/Pasted%20image%2020250108202047.png)

- 가상환경
    
    - venv
    - conda
    - pyenv-virtualenv
    - pipenv
- venv 구조
    
    - bin 폴더: shell에서 명령어 입력 시, 이 경로 내에서 제일 먼저 찾음 python 입력 시 어떤 python을 실행시키는 지 알 수 있음
    - lib 폴더: 패키지 설치 시, 이 경로에 설치. import 시 이 경로에서 제일 먼저 찾음
- pip 패키지 매니저
    
    - pip list --­not-required --­format=freeze: 어떠한 패키지를 실행할 때 필요한 패키지를 보여줌
    - pip freeze > requirements.txt: 명령어를 통해 requirements.txt를 만들 수 있음
- 디버깅 Process
    

![Pasted image 20250108202101.png](/img/user/images/Pasted%20image%2020250108202101.png)
