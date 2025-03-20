---
{"dg-publish":true,"permalink":"/2-naver-boost-camp/week-19/","created":"2025-03-11T14:12:26.490+09:00","updated":"2025-03-20T15:58:45.104+09:00"}
---

# Product Serving 개요

모델이나 프로그램을 개발한 후에는 이를 클라이언트가 실제로 사용할 수 있도록 제품으로 배포해야 합니다. 이를 **serving**이라고 하며, 대표적인 예로 ChatGPT가 prompt를 통해 사용자 요청에 응답하는 방식을 들 수 있습니다.  
serving 방식은 크게 **Batch Serving**과 **Online (Real Time) Serving** 두 가지로 구분되며, 문제 상황, 제약 조건, 인력, 데이터 저장 형태, 레거시 시스템 유무 등에 따라 적합한 방식을 선택할 수 있습니다.

---

## 1. Batch Serving

### 1-1. 개념 및 특징

- **정의:**  
    데이터를 일정한 묶음 단위로 처리하여 서빙하는 방식입니다.
- **적용 상황:**
    - 실시간 응답이 필수적이지 않은 경우
    - 대량의 데이터를 한꺼번에 처리할 때
    - 정기적(일별, 월별, n시간 단위)으로 작업이 수행되어도 되는 경우
    - 인력이 부족하거나 RDB, 데이터 웨어하우스를 활용할 때

**예시:**  
Netflix의 콘텐츠 추천 시스템은 n시간 단위로 예측 결과를 생성해 DB에 저장한 후, DB의 예측 결과를 읽어와 서빙하는 방식입니다.

### 1-2. Batch 패턴 구성 요소
![Pasted image 20250311142136.png](/img/user/images/Pasted%20image%2020250311142136.png)
Batch 패턴은 크게 3개의 파트로 구성할 수 있습니다.

- **Job Management Server**
    
    - 작업을 총괄하는 서버
    - Apache Airflow 같은 도구를 활용해 특정 시간에 주기적으로 batch job을 실행
- **Job**
    
    - 특정 작업 실행에 필요한 모든 활동을 포함
    - Python 스크립트, Docker 이미지 등으로 구현
    - 예: 모델 로드, 데이터 로드 등
- **Data**
    
    - 예측 결과를 저장할 DB나 데이터 웨어하우스

**장점:**

- 기존 코드를 재사용할 수 있으며, 별도의 API 서버 개발 없이도 서빙이 가능
- 서버 리소스를 유연하게 관리할 수 있음

**단점:**

- 별도의 스케줄러(예: Apache Airflow)가 필요

---

## 2. Online (Real Time) Serving

### 2-1. 개념 및 특징

- **정의:**  
    클라이언트가 요청할 때마다 즉각적으로 모델이 예측 결과를 생성해 서빙하는 방식입니다.
- **적용 상황:**
    - 즉각적인 응답이 필수적인 경우
    - 개별 요청에 대해 맞춤형 처리가 필요할 때
    - 동적 데이터에 대응해야 할 때

**예시:**

- 유튜브 추천 시스템 (새로고침 시마다 추천 결과가 변동)
- 번역 서비스

### 2-2. Online Serving 패턴: Web Single 패턴
![Pasted image 20250311142148.png](/img/user/images/Pasted%20image%2020250311142148.png)
**Web Single 패턴 구성 요소:**

- **Inference Server**
    
    - FastAPI, Flask 등으로 단일 REST API 서버를 개발하여 배포
    - 서버가 실행될 때 모델을 로드하고, 전처리 로직이 함께 포함됨
- **Client**
    
    - 서비스를 요청하는 사용자 인터페이스(웹, 앱 등)
- **Data**
    
    - 요청 시 함께 제공되는 입력 데이터
- **Load Balancer**
    
    - Nginx, Amazon ELB 등을 사용해 트래픽을 분산, 서버 과부하 방지

**장점:**

- 빠른 출시와 실시간 예측이 가능
- 단일 프로그래밍 언어로 구현할 수 있어 아키텍처가 단순

**단점:**

- 구성 요소 중 하나라도 변경되면 전체 업데이트 필요
- 모델이 크거나 로딩 시간이 오래 걸리는 경우 초기 응답 지연 가능

---

## 3. Serving 처리 방식: Synchronous vs. Asynchronous

### 3-1. Synchronous 패턴

- **특징:**  
    하나의 작업이 완료될 때까지 다음 작업을 시작하지 않고 기다리는 방식
- **적용:**  
    대부분의 REST API가 이 방식을 사용하며, 예측 결과에 따라 클라이언트 로직이 즉각적으로 달라져야 할 때 유용
- **장점:**  
    아키텍처 및 워크플로우가 단순
- **단점:**  
    동시에 다수의 요청이 들어오면 처리에 어려움이 발생

### 3-2. Asynchronous 패턴

- **특징:**  
    하나의 작업을 시작한 후 결과를 기다리는 동안 다른 작업을 수행할 수 있는 방식
    - 클라이언트와 예측 서버 사이에 메시지 큐(예: Apache Kafka)를 도입하여 요청(push)과 결과 회수(pull) 방식으로 병렬 처리
- **장점:**  
    클라이언트와 예측 프로세스가 분리되어, 클라이언트가 예측 결과를 기다리지 않고도 다른 작업을 수행 가능
- **단점:**  
    별도의 큐 시스템 구축이 필요하여 전체 구조가 복잡해지며, 완전한 실시간 예측에는 적합하지 않을 수 있음

---

# 결론

제품 서빙은 개발한 프로그램이나 모델을 실제 서비스에 적용하기 위한 중요한 단계입니다.

- **Batch Serving:**  
    대량 데이터나 정기적 작업에 적합하며, 기존 코드 재사용과 유연한 서버 리소스 관리의 장점이 있습니다.
- **Online (Real Time) Serving:**  
    즉각적인 응답과 맞춤형 처리가 필요한 경우 적합하며, Web Single 패턴, Synchronous, Asynchronous 패턴 등 다양한 구조로 구현할 수 있습니다.

각 방식은 상황과 요구사항에 따라 선택될 수 있으며, 문제 상황, 제약 조건, 데이터 저장 방식, 시스템 환경 등을 종합적으로 고려해 최적의 서빙 방식을 결정할 수 있습니다.

---
# Apache Airflow로 Batch Serving 워크플로우 구축하기

**Apache Airflow**는 워크플로우 관리 및 스케줄링 도구로, 모델 학습이나 예측과 같은 주기적인 작업(batch serving)을 자동화하는 데 유용합니다. **예를 들어, 학습은 1주일에 1번, 예측은 10분마다 실행하는 등의 스케줄링이 가능합니다.**

이번 포스트에서는 Airflow의 기본 개념, 설치 및 환경 설정, DAG 작성 방법, 그리고 Slack 연동을 통한 알림 기능까지 자세히 알아보겠습니다.

---

## 1. Apache Airflow 기본 개념

### DAG (Directed Acyclic Graph)

- **정의:** 작업(Task)들의 흐름과 순서를 정의하는 구조입니다.
- **역할:** Airflow에서 실행할 작업을 어떻게 연결할지 결정합니다.

### Operator

- **정의:** Airflow의 작업 유형을 나타내는 클래스입니다.
- **예시:**
    - **BashOperator:** Bash 명령어 실행
    - **PythonOperator:** Python 함수 호출
    - **SQLOperator:** SQL 쿼리 실행

### Scheduler

- **역할:** DAG를 모니터링하면서, 실행 시점에 맞춰 작업을 예약합니다.

### Executor

- **정의:** 작업이 실제 실행되는 환경입니다.
- **예시:**
    - **LocalExecutor:** 로컬 환경에서 실행
    - **CeleryExecutor:** 분산 환경에서 작업 실행

---

## 2. Apache Airflow 설치 및 초기 설정

### 설치

최신 버전에서 다른 라이브러리와의 충돌 및 버그가 발생할 수 있으므로, 안정적인 버전을 사용하는 것이 좋습니다.  
예시) Python 3.11.7 기준, Airflow 2.6.3

`$ pip3 install apache-airflow==2.6.3`

### 환경 변수 설정

Airflow의 홈 디렉토리를 지정합니다.
`$ vi ~/.bashrc`

파일 맨 아래에 다음 내용을 추가:
`export AIRFLOW_HOME=your_directory`

변경 사항 적용은 터미널 재시작 또는 다음 명령어 실행:
`$ source ~/.bashrc`

### 데이터베이스 초기화

Airflow는 기본적으로 SQLite를 사용합니다.
`$ airflow db init`

이 명령으로 `airflow.cfg`와 `airflow.db` 파일이 생성됩니다.

### Admin 계정 생성

Airflow Web UI에 로그인할 계정을 생성합니다.
`$ airflow users create --username admin --password your_password --firstname gildong --lastname hong --role Admin --email id@gmail.com`

### Webserver 및 Scheduler 실행

- **Airflow Webserver 실행:**
    `$ airflow webserver --port 8080`
    
    브라우저에서 [http://localhost:8080](http://localhost:8080)을 열고 위 계정으로 로그인합니다.
    
- **Airflow Scheduler 실행:**  
    별도의 터미널에서 실행:
    `$ airflow scheduler`
    

---

## 3. DAG 작성하기

DAG는 Airflow에서 작업의 흐름과 순서를 정의합니다.

1. **DAG 파일 저장 폴더:**  
    `AIRFLOW_HOME` 내에 `dags` 디렉토리를 생성합니다.
2. **DAG 파일 작성:**  
    아래는 날짜 출력과 "Hello world!" 메시지를 출력하는 간단한 DAG 예제입니다.


```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    "owner": "gildong",
    "depends_on_past": False,    # 이전 DAG의 task 성공 여부에 따라서 현재 task를 실행할지 결정. False는 과거 task의 성공 여부와 상관 없이 실행
    "start_date": datetime(2024, 1, 1),
    "end_date": datetime(2024, 1, 8)
}

def print_hello():
	print("Hello world!")

#####################################################################
# Part 1. DAG 정의

with DAG(
    dag_id = "basic_dag",
    default_args=default_args,
    schedule_interval="30 0 * * *",     # 매일 UTC 00:30 AM에 실행 / 한 번만 실행하고 싶다면 "@once"
    tags=["my_dags"]
) as dag:

#####################################################################
# Part 2. task 정의
    task1 = BashOperator(
        task_id="print_date",   
        bash_command="date"   # 실행할 bash command
    )
    
    task2 = PythonOperator(
        task_id="print_hello",
        python_callable=print_hello
    )
        
#####################################################################
# Part 3. task 순서 정의    
    task1 >> task2
```

### Cron 표현식 간단 정리
![Pasted image 20250311142550.png](/img/user/images/Pasted%20image%2020250311142550.png)
- **구성:**
    - 1번째 자리: 분 (0-59)
    - 2번째 자리: 시 (0-23)
    - 3번째 자리: 일 (1-31)
    - 4번째 자리: 월 (1-12)
    - 5번째 자리: 요일 (0-6; 0=일요일)
- **예시:**
    - `"30 0 * * *"`: 매일 00:30 AM에 실행
    - `"@once"`: 한 번만 실행

작성한 DAG 파일을 `basic.py`로 저장 후, Airflow Web UI에서 확인할 수 있습니다.

---

## 4. Slack 연동으로 알림 받기

Airflow에서 task 실패 시 Slack으로 알림을 받아 즉각적으로 문제를 확인할 수 있습니다.

### 4-1. Airflow Slack Provider 설치

Python 3.11.7, Airflow 2.6.3과 호환되는 버전(8.6.0) 설치:

```bash
$ pip3 install 'apache-airflow-providers-slack[http]'==8.6.0
```

### 4-2. Slack API Key 발급 및 Webhook 설정

1. Slack API Apps 페이지에서 "Create New App > From scratch"를 선택합니다.
2. App Name과 Workspace를 설정한 후, **Basic Information** 탭에서 **Incoming Webhooks**를 활성화합니다.
3. "Add New Webhook to Workspace"를 통해 특정 채널에 대한 Webhook URL(예: `https://hooks.slack.com/services/~~~~~~~~/1234567`)을 발급받습니다.

### 4-3. Airflow에 Webhook 등록

Airflow Web UI에서 **Admin > Connections**로 이동 후, 새 Connection을 추가합니다.

- **Connection id:** 예) `slack_webhook`
- **Connection type:** HTTP
- **Host:** `https://hooks.slack.com/services`
- **Password:** `/~~~~~~~~~~/1234567` (비밀 정보)

### 4-4. Slack 알림 코드 작성

아래 코드를 `utils/slack_alert.py` 파일로 저장합니다.

```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

SLACK_DAG_CONN_ID = "slack_webhook"    # Connection id에 입력한 본인이 식별 가능한 이름

def send_message(slack_msg):
    return SlackWebhookOperator(
        task_id="slack_webhook",
        slack_webhook_conn_id=SLACK_DAG_CONN_ID,
        message=slack_msg,
        username="Airflow-alert"
    )
    
def fail_alert(context):
    slack_msg = """
            Task Failed!
            Task: {task}
            Dag: `{dag}`
            Execution Time: {exec_date}
            """.format(
                task=context.get("task_instance").task_id, 
                dag=context.get("task_instance").dag_id,
                exec_date=context.get("execution_date")
            )
            
    alert = send_message(slack_msg)
    
    return alert.execute(context=context)
```

이제 DAG 파일에서 아래와 같이 Slack 알림 함수를 import하고, `on_failure_callback` 인자로 등록합니다.

```python
with DAG(
    dag_id = "basic_dag",
    default_args=default_args,
    schedule_interval="30 0 * * *",     
    tags=["my_dags"],
    on_failure_callback=fail_alert
) as dag:
```

성공 시 알림을 받고 싶다면, 메시지를 성공으로 변경한 후 `on_success_callback` 인자를 사용하면 됩니다.

---

## 결론

Apache Airflow는 복잡한 워크플로우를 쉽게 정의하고, 배치 스케줄링을 통해 반복적인 작업을 자동화할 수 있는 강력한 도구입니다.

- **설치 및 환경 설정:** 안정적인 버전과 환경 변수 설정, 데이터베이스 초기화, Admin 계정 생성
- **DAG 작성:** 작업의 흐름, Operator를 이용한 Task 정의, Cron 표현식을 활용한 스케줄링
- **Slack 연동:** 작업 실패 시 즉각적인 알림으로 문제 대응

---

# 서버 아키텍처와 Web API 이해하기

현대 웹/모바일 애플리케이션의 핵심은 서버 아키텍처와 API 설계입니다. 이 포스트에서는 서버 아키텍처의 종류와 각각의 특징, 그리고 API의 기본 개념 및 REST API의 구성 요소와 URL, HTTP 요청에 대한 기본 개념을 살펴보겠습니다.

---

## 1. 서버 아키텍처

### 1-1. 모놀리식 아키텍처 (Monolithic Architecture)

- **정의:**  
    데이터베이스, 모델 등 모든 로직이 하나의 큰 서버에서 구현되는 형태입니다.
- **특징:**
    - 클라이언트는 단일 서버에 요청을 보내고, 서버 내부에서 모든 요청을 처리 후 반환합니다.
    - 개발 초기에는 구조가 단순하고 직관적이지만, 서비스 확장 시 복잡도가 증가하여 수정이나 추가 개발이 어려워질 수 있습니다.

### 1-2. 마이크로서비스 아키텍처 (Microservice Architecture)

- **정의:**  
    기능별로 여러 개의 작은 서버(예: DB 서버, 모델 서버 등)로 로직을 분리하여 개발하는 형태입니다.
- **특징:**
    - 클라이언트가 요청하면 중앙 서버가 이를 각각의 내부 서버로 분배하여 처리 후 결과를 모아 응답합니다.
    - 서버 단위로 독립적인 의존성 및 환경 설정이 가능하지만, 전체 구조는 상대적으로 복잡해질 수 있습니다.

---

## 2. API의 기본 개념

### API란?

- **API (Application Programming Interface):**  
    소프트웨어 응용 프로그램들이 서로 상호작용할 수 있도록 정의한 인터페이스를 총칭합니다.

### Web API

- **정의:**  
    HTTP(Hyper Text Transfer Protocol)를 기반으로 웹 기술을 활용해 데이터를 주고받는 API입니다.
- **주요 종류:**
    - **REST:** 자원(Resource) 기반의 상태 전송
    - **GraphQL:** 클라이언트가 필요한 데이터만 쿼리할 수 있는 방식
    - **RPC:** 원격 프로시저 호출 방식

---

## 3. REST API

REST API는 자원(Resource)을 표현하고 상태를 전송하는 데 중점을 둔 아키텍처입니다. HTTP를 기반으로 하며, 요청을 통해 어떤 작업을 수행하는지 쉽게 파악할 수 있습니다.

### 3-1. REST API 구성 요소

- **Resource:**
    
    - 시스템에서 관리하는 모든 데이터나 개체
    - 예시: 전자상거래 사이트의 `users`, `products`, `orders`
    - URL endpoint 예시:
        - `www.shopping.com/users`
        - `www.shopping.com/products`
        - `www.shopping.com/orders`
- **Method (HTTP 메서드):**
    
    - **GET:** 리소스 조회
    - **POST:** 리소스 생성
    - **PUT:** 리소스 전체 업데이트
    - **PATCH:** 리소스 일부 업데이트
    - **DELETE:** 리소스 삭제
- **Representation of Resource:**
    
    - 클라이언트와 서버가 주고받는 리소스의 "표현" (보통 JSON 또는 XML)
    - 예시:
        - 요청: `GET /users/1`
        - 응답:	
        ```json
		{   "id": 1,   "name": "Alice",   "email": "alice@mail.com" }
		```

---

## 4. URL의 구성 요소

URL은 클라이언트가 서버에 요청할 리소스의 위치를 나타내며, 다음과 같이 구성됩니다.

- **Schema:**
    - 사용 프로토콜 (예: `http`, `https`)
- **Host:**
    - 서버의 IP 주소나 도메인 이름
- **Port:**
    - 통신 포트 번호 (예: `8080`)
- **Path 또는 Endpoint:**
    - 반환할 리소스의 경로
    - 예시:
        - `/users`, `/predict`, `/train` 등
- **Query Parameter:**
    - 특정 리소스에 대한 추가 정보 제공이나 필터링
    - 예시:
    ```bash
     http://localhost:8080/users?name=alice
```
        
- **Path Parameter:**
    - 리소스의 정확한 위치를 지정
    - 예시:
        ```bash
        https://localhost:8080/users/alice
		```
        

---

## 5. HTTP Header와 Payload

CLI 환경에서는 HTTP 요청 시 Header와 Payload를 함께 사용할 수 있습니다. 예를 들어, `curl`을 사용한 POST 요청은 다음과 같습니다.
```bash
curl -X POST -H "Content-Type: application/json" -d "{'name':'alice'}" http://localhost:8080/users
```
- **curl:**
    - CLI 환경에서 HTTP 요청을 수행하는 명령어
- **-X POST:**
    - HTTP 메서드 POST를 사용 (리소스 생성)
- **-H "Content-Type: application/json":**
    - 전송 데이터가 JSON 형식임을 명시하여 HTTP Header에 key-value 형태로 저장
- **-d "{'name':'alice'}":**
    - 실제 전송하려는 데이터가 HTTP Payload에 포함됨
- **[http://localhost:8080/users](http://localhost:8080/users):**
    - 요청을 보낼 목적지 URL

---

## 6. HTTP Status Code

서버는 클라이언트 요청에 따라 상태 코드를 반환하여 요청 처리 결과를 알려줍니다.

- **1XX (정보):** 요청을 받았으며, 프로세스가 진행 중임
- **2XX (성공):** 요청이 성공적으로 처리됨
- **3XX (리다이렉션):** 요청 완료를 위해 추가 작업이 필요함
- **4XX (클라이언트 오류):** 요청 문법 오류 또는 요청 처리 불가
- **5XX (서버 오류):** 서버가 요청 처리를 실패함

---

# 결론

이번 포스트에서는 아래의 내용을 정리하였습니다.

- **서버 아키텍처**: 모놀리식과 마이크로서비스 아키텍처의 개념 및 장단점
- **API 기본 개념**: API와 Web API, 그리고 REST API의 구성 요소
- **URL 구성 요소**: Schema, Host, Port, Path, Query 및 Path Parameter
- **HTTP 요청**: Header, Payload 사용 방법과 `curl` 예제
- **Status Code**: 각 코드의 의미와 역할

---

# FastAPI를 활용한 Online Serving API 개발

**FastAPI는 파이썬 기반의 웹 프레임워크**로, 빠르고 효율적으로 API를 개발할 수 있게 해줍니다. 본 포스트에서는 간단한 머신러닝(ML) 모델을 로드해 예측 결과를 반환하는 **API 웹 프로젝트를 예제로 소개하며, 프로젝트 구조부터 설치, 설정, 각 기능 구현(예측, 파일 업로드, 라우터, 백그라운드 작업 등)까지 전반적인 내용**을 다룹니다.

---

## 1. 프로젝트 구조

아래와 같이 디렉토리를 구성하여 각 기능을 모듈별로 분리합니다.

```
project_root/
 ┣ app/
 ┃  ┣ __init__.py    
 ┃  ┣ main.py         # FastAPI 애플리케이션, 라우터 설정 및 lifespan 함수
 ┃  ┣ config.py       # 데이터베이스, 모델 경로, 실행 환경 등의 설정
 ┃  ┣ api.py          # 모델 예측 기능 등 API 엔드포인트 구현
 ┃  ┣ schemas.py      # 요청/응답 데이터 스키마 정의 (Pydantic 모델)
 ┃  ┣ database.py     # 데이터베이스 연결 및 테이블 생성 (SQLModel 등)
 ┃  ┣ model.py        # ML 모델 로드 및 관련 함수 정의 (예: predict)
 ┣ router.py          # (필요 시) 추가 API 라우터 파일 (예: user, order 등 분리)
 ┣ requirements.txt   # 필요한 패키지 목록
 ┗ README.md
```

간단한 ML 모델을 로드하여 입력 데이터를 예측한 후 결과를 반환하고, DB에 예측 결과를 저장하는 API 웹을 구현하는 예제입니다.  
더 자세한 프로젝트 구조 참고 자료는 아래 링크들을 확인해보세요.

- [cookiecutter](https://github.com/cookiecutter/cookiecutter)
- [cookiecutter-data-science](https://github.com/drivendataorg/cookiecutter-data-science)
- [cookiecutter-fastapi](https://github.com/arthurhenrique/cookiecutter-fastapi)

---

## 2. FastAPI 설치
```bash
$ pip install fastapi uvicorn
```

uvicorn은 FastAPI 애플리케이션 실행에 필요한 ASGI 서버입니다.

---

## 3. 각 파일별 구현 내용

### 3-1. config.py

FastAPI 프로젝트에서 데이터베이스 URL, 모델 경로, 실행 환경 등 설정을 관리합니다.

```python
from pydantic import BaseSettings, Field

class Config(BaseSettings):
    db_url: str = Field(default="sqlite:///./db.sqlite3", env="DB_URL")
    model_path: str = Field(default="model.joblib", env="MODEL_PATH")
    app_env: str = Field(default="local", env="APP_ENV")

config = Config()
```

> **참고:**  
> Pydantic을 활용해 환경 변수로부터 설정값을 불러오므로, 배포 환경에 맞춰 쉽게 조정할 수 있습니다.

---

### 3-2. database.py

SQLModel과 같은 ORM을 사용하여 데이터베이스 연결 및 테이블 생성을 위한 코드를 작성합니다.

```python
import datetime
from sqlmodel import SQLModel, Field, create_engine
from config import config

class PredictionResult(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    result: int
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

engine = create_engine(config.db_url)
```

---

### 3-3. model.py

모델 로드와 관련된 함수들을 정의합니다.

```python
def load_model(model_path: str):
    import joblib
    return joblib.load(model_path)

# 필요 시 get_model, predict 등 다른 함수도 정의 가능
```

---

### 3-4. schemas.py

API에서 주고 받을 데이터의 스키마(형태)를 Pydantic을 이용해 정의합니다.

```python
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    features: list  # 상황에 맞게 input 데이터의 형식을 정의

class PredictionResponse(BaseModel):
    id: int
    result: int
```

---

### 3-5. api.py

모델 예측 및 결과 조회 기능을 구현하는 API 엔드포인트를 작성합니다.

```python
from fastapi import APIRouter, HTTPException, status
from schemas import PredictionRequest, PredictionResponse
from model import load_model  # 혹은 get_model 함수로 변경
from database import PredictionResult, engine
from sqlmodel import Session

router = APIRouter()

def get_model():
    # 실제 운영에서는 모델을 메모리에 캐싱하는 전략을 사용
    return load_model("model.joblib")

@router.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = get_model()
    # 예측: 예시로 첫 번째 예측값을 정수형으로 변환
    prediction = int(model.predict([request.features])[0])
    
    # 예측 결과를 DB에 저장
    prediction_result = PredictionResult(result=prediction)
    with Session(engine) as session:
        session.add(prediction_result)
        session.commit()
        session.refresh(prediction_result)
        
    return PredictionResponse(id=prediction_result.id, result=prediction)

@router.get("/predict/{id}", response_model=PredictionResponse)
def get_prediction(id: int) -> PredictionResponse:
    with Session(engine) as session:
        prediction_result = session.get(PredictionResult, id)
        if not prediction_result:
            raise HTTPException(
                detail="Not Found", status_code=status.HTTP_404_NOT_FOUND
            )
        return PredictionResponse(id=prediction_result.id, result=prediction_result.result)
```

---

### 3-6. main.py

FastAPI 애플리케이션 생성, 라우터 등록, 그리고 애플리케이션의 lifespan(시작/종료 시 수행할 작업)을 정의합니다.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
from sqlmodel import SQLModel

from config import config
from database import engine
from model import load_model
from api import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 전: 데이터베이스 테이블 생성
    logger.info("Creating database tables")
    SQLModel.metadata.create_all(engine)
    
    # 앱 시작 전: 모델 로드 (여기서 load_model를 호출)
    logger.info("Loading model")
    load_model(config.model_path)
    
    yield
    # 앱 종료 전: 필요한 정리 작업 추가 가능

app = FastAPI(lifespan=lifespan)
app.include_router(router)

# 간단한 기본 루트 엔드포인트
@app.get("/")
def root():
    return {"message": "Hello World!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

> **동작 확인:**
> 
> - 웹 브라우저에서 [http://localhost:8000](http://localhost:8000/) 접속 시 "Hello World!" 메시지 확인
> - `localhost:8000/docs` 에서 자동 생성된 Swagger 문서를 통해 API 테스트 가능
> - CLI에서 예측 요청 예시:
>     
>     ```bash
>     curl -X POST "http://0.0.0.0:8000/predict" -H "Content-Type: application/json" -d "{'features': [5.1, 3.5, 1.4, 0.2]}"
>     ```
>     
>     예측 결과와 함께 DB에 저장된 결과를 확인할 수 있습니다.

---

## 4. 추가 기능 구현

FastAPI는 다양한 HTTP 메서드와 데이터 전송 방식, 그리고 배경 작업, 파일 업로드, 프론트엔드 렌더링 등 여러 기능을 지원합니다.

### 4-1. Path & Query Parameter

- **Path Parameter 예제:**
    
    ```python
    @app.get("/users/{user_id}")
    def get_user(user_id: str):
        return {"user_id": user_id}
    ```
    
- **Query Parameter 예제:**
    
    ```python
    items_db = [{"item_name": "Apple"}, {"item_name": "Banana"}, {"item_name": "Cake"}]
    
    @app.get("/items/")
    def read_item(skip: int = 0, limit: int = 10):
        return items_db[skip: skip + limit]
    ```
    

### 4-2. Form 데이터 처리

Form 데이터 처리를 위해 `python-multipart` 패키지를 설치합니다.

```bash
pip install python-multipart
```

그리고 아래와 같이 Form 데이터를 처리할 수 있습니다.

```python
from fastapi import FastAPI, Form

@app.post("/login/")
def login(username: str = Form(...), password: str = Form(...)):
    return {"username": username}
```

### 4-3. 프론트엔드 렌더링 (Jinja2)

프론트엔드 페이지를 렌더링하려면 Jinja2를 사용합니다.

```bash
pip install Jinja2
```

```python
from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory='./')

@app.get("/login/")
def get_login_form(request: Request):
    return templates.TemplateResponse('login_form.html', context={'request': request})
```

_`login_form.html` 파일은 프로젝트 루트나 지정한 디렉토리에 위치해야 합니다._
![Pasted image 20250311144549.png](/img/user/images/Pasted%20image%2020250311144549.png)

### 4-4. 파일 업로드
![Pasted image 20250311144606.png](/img/user/images/Pasted%20image%2020250311144606.png)
파일 업로드 기능도 쉽게 구현할 수 있습니다. Form을 구현할 때처럼 `python-multipart` 설치가 필요하다.

```python
from typing import List
from fastapi import File, UploadFile
from fastapi.responses import HTMLResponse

@app.post("/files/")
def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}

@app.post("/uploadfiles/")
def create_upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}

@app.get("/upload/")
def main():
    content = """
    <body>
    <form action="/files/" enctype="multipart/form-data" method="post">
      <input name="files" type="file" multiple>
      <input type="submit">
    </form>
    <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
      <input name="files" type="file" multiple>
      <input type="submit">
    </form>
    </body>
    """
    return HTMLResponse(content=content)
```

### 4-5. API Router 분리

프로젝트가 커지면 여러 엔드포인트를 효율적으로 관리하기 위해 별도의 라우터 모듈로 분리할 수 있습니다.

```python
# router.py
from fastapi import APIRouter

user_router = APIRouter(prefix="/users")
order_router = APIRouter(prefix="/orders")

@user_router.get("/{username}", tags=["users"])
def read_user(username: str):
    return {"username": username}

@order_router.get("/{order_id}", tags=["orders"])
def read_order(order_id: str):
    return {"order_id": order_id}
```

그리고 main.py에 아래와 같이 등록합니다.

```python
from router import user_router, order_router

app.include_router(user_router)
app.include_router(order_router)
```

### 4-6. 백그라운드 작업 (Background Task)

긴 작업을 백그라운드에서 실행하여 즉각 응답을 주기 위해 BackgroundTasks를 사용할 수 있습니다.

```python
from uuid import UUID, uuid4
from time import sleep
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field

class TaskInput(BaseModel):
    id_: UUID = Field(default_factory=uuid4)
    wait_time: int

task_repo = {}

def cpu_bound_task(id_: UUID, wait_time: int):
    sleep(wait_time)
    result = f"task done after {wait_time}"
    task_repo[id_] = result

@app.post("/task", status_code=202)
async def create_task_in_background(task_input: TaskInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(cpu_bound_task, id_=task_input.id_, wait_time=task_input.wait_time)
    return {"task_id": task_input.id_}

@app.get("/task/{task_id}")
def get_task_result(task_id: UUID):
    return task_repo.get(task_id, None)
```

HTTP 202 (Accepted) 코드를 리턴해 비동기 작업이 등록되었음을 알릴 수 있습니다.

---

## 결론

FastAPI를 사용하면 간단한 웹 API부터 복잡한 온라인 서빙 시스템까지 빠르게 구축할 수 있습니다.

- **프로젝트 구조 설계:** 각 모듈을 분리하여 유지보수성과 확장성을 높입니다.
- **환경 설정 및 구성 파일:** Pydantic을 활용해 환경변수 및 설정을 관리합니다.
- **주요 기능 구현:** 예측 기능, 데이터베이스 저장, 파일 업로드, 라우터 분리, 백그라운드 작업 등 다양한 기능을 쉽게 추가할 수 있습니다.
- **프론트엔드 및 문서화:** Swagger UI(`/docs`)를 기본으로 제공하며, Jinja2를 활용해 프론트엔드도 구축 가능합니다.

---

# Poetry를 이용한 파이썬 패키지 및 의존성 관리

**Poetry는 파이썬 프로젝트의 패키지 설치, 가상환경 생성, 의존성 관리, 그리고 배포를 위한 패키징 작업(build, publish)을 한 곳에서 처리할 수 있는 도구**입니다. 전통적인 pip나 anaconda와 달리, Poetry는 프로젝트의 전체 생명주기를 관리할 수 있어 **점점 더 많은 개발자들이 채택하고 있습니다.**

---

## 1. Poetry란?

Poetry는 다음과 같은 작업을 수행합니다.

- **패키지 설치:** 프로젝트에서 필요한 라이브러리와 의존성을 설치
- **가상환경 생성:** 독립적인 개발 환경을 자동으로 생성하여 관리
- **의존성 관리:** 프로젝트 의존성을 `pyproject.toml` 파일에 정의하고, 정확한 버전을 `poetry.lock` 파일에 기록
- **패키징 및 배포:** 프로젝트를 빌드하고, PyPI와 같은 저장소에 배포할 수 있는 기능 제공

---

## 2. Poetry 설치

Poetry는 Python 2.7 또는 3.5 이상에서 설치할 수 있습니다.

### Mac / Linux

터미널에서 다음 명령어를 실행합니다.

```bash
$ curl -sSL https://install.python-poetry.org | python3 -
```

### Windows

CMD 창 혹은 PowerShell에서 아래 명령어를 실행합니다.

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

---

## 3. 프로젝트 초기화 및 설정

### 프로젝트 초기화

현재 디렉토리에서 Poetry 프로젝트를 초기화하려면 다음 명령어를 사용합니다.

```bash
$ poetry init
```

이 명령어는 대화형 인터페이스를 통해 프로젝트의 의존성, 버전, 설명 등을 입력받아 `pyproject.toml` 파일을 생성합니다. 이를 통해 기존 디렉토리에서 새로운 Poetry 프로젝트를 시작할 수 있습니다.

### 가상환경 활성화

Poetry는 프로젝트별 가상환경을 자동으로 관리합니다. 다음 명령어로 가상환경을 활성화할 수 있습니다.

```bash
$ poetry shell
```

---

## 4. 라이브러리 설치 및 관리

### 라이브러리 설치

`pyproject.toml` 파일에 정의된 의존성을 기반으로 필요한 라이브러리를 설치하려면 아래 명령어를 실행합니다.

```bash
$ poetry install
```

특정 패키지를 추가하려면 `poetry add` 명령어를 사용합니다.

```bash
$ poetry add pandas
```

이렇게 하면 `pyproject.toml` 파일에 해당 패키지가 추가되고, 동시에 `poetry.lock` 파일에도 기록됩니다.

### 잠금파일 (Lock File)

Poetry는 `poetry.lock` 파일을 생성하여 현재 프로젝트에 필요한 모든 의존성과 그 정확한 버전을 기록합니다.

- 이 파일을 GitHub와 같은 버전 관리 시스템에 커밋해두면, 다른 개발자가 동일한 환경에서 작업할 수 있습니다.
- `poetry.lock` 파일을 통해 모든 팀원이 동일한 의존성 버전을 보장받게 됩니다.

---

## 결론

Poetry는 패키지 설치부터 가상환경 관리, 의존성 및 배포까지 파이썬 프로젝트의 전반적인 관리를 하나의 도구로 해결할 수 있는 강력한 도구입니다.

- **설치 및 초기화:** 간단한 명령어로 시작하여, `pyproject.toml` 파일로 프로젝트 설정
- **가상환경 관리:** `poetry shell`을 통해 손쉽게 가상환경 활성화
- **의존성 관리:** `poetry.lock` 파일로 모든 의존성의 버전을 확정하여 협업 환경에서 안정성을 보장