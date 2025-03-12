---
{"dg-publish":true,"permalink":"/2-naver-boost-camp/week-20/","created":"2025-03-11T14:50:03.039+09:00","updated":"2025-03-11T15:09:29.975+09:00"}
---

# pydantic: 데이터 검증 및 설정 관리를 위한 강력한 파이썬 라이브러리

**pydantic은 파이썬에서 데이터 모델링과 설정 관리를 손쉽게 할 수 있도록 돕는 라이브러리입니다.** 타입 힌트를 기반으로 작성된 클래스를 통해 데이터를 검증하고, JSON 직렬화/역직렬화를 자연스럽게 지원합니다. 특히 FastAPI와 같은 프레임워크에서 널리 사용되어, API 요청/응답 데이터의 검증 및 환경 설정 관리에 큰 도움을 주고 있습니다.

---

## 1. pydantic의 핵심 기능

### 데이터 검증 (Validation)

- **타입 검증:**  
    선언된 타입 힌트에 따라 입력 데이터가 올바른지 확인합니다. 예를 들어, URL, 정수 범위, 디렉토리 경로 등의 검증을 기본 제공하는 타입을 활용할 수 있습니다.
- **사용자 친화적 에러 메시지:**  
    검증에 실패할 경우, 어느 필드에서 어떤 이유로 실패했는지 자세한 정보를 JSON 형태로 제공하여 디버깅에 용이합니다.

### 데이터 직렬화 및 역직렬화

- **JSON 변환:**  
    pydantic 모델은 쉽게 JSON으로 직렬화할 수 있으며, JSON 데이터를 Python 객체로 역직렬화하는 과정이 간편합니다.

---

## 2. 데이터 검증 예제

다음 예제는 입력 데이터가 올바른 URL, 1~10 사이의 정수, 그리고 존재하는 디렉토리인지 검증하는 pydantic 모델을 보여줍니다.

```python
from pydantic import BaseModel, HttpUrl, Field, DirectoryPath

class Validation(BaseModel):
    url: HttpUrl              # 올바른 URL인지 검증
    rate: int = Field(ge=1, le=10)  # 1~10 사이의 정수인지 검증
    target_dir: DirectoryPath # 실제 존재하는 디렉토리인지 검증
```

### 검증 실행 예제

```python
import os
from pydantic import ValidationError

VALID_INPUT = {
    "url": "https://content.presspage.com/uploads/2658/c800_logo-stackoverflow-square.jpg?98978",
    "rate": 4,
    "target_dir": os.path.join(os.getcwd(), "examples"),  # 현재 디렉토리 내 "examples" 폴더가 있어야 함
}

INVALID_INPUT = {"url": "WRONG_URL", "rate": 11, "target_dir": "WRONG_DIR"}

# 올바른 입력을 검증하는 경우
valid_model = Validation(**VALID_INPUT)
print("검증 성공:", valid_model)

# 잘못된 입력은 ValidationError 발생
try:
    invalid_model = Validation(**INVALID_INPUT)
except ValidationError as exc:
    print("pydantic model input validation error:", exc.json())
```

위 코드는 VALID_INPUT의 경우 정상적으로 객체를 생성하며, INVALID_INPUT의 경우 에러 메시지를 출력합니다. 에러 메시지에는 어떤 필드에서 어떤 값이 문제였는지 상세히 안내해줍니다.

---

## 3. 설정 관리 (Configuration) with BaseSettings

프로젝트에서 상수를 코드에 하드코딩하거나 별도의 파일(yaml 등)로 관리하는 대신, pydantic의 BaseSettings를 활용하면 환경 변수로부터 설정 값을 안전하게 불러올 수 있습니다. 이를 통해 배포 환경에 따라 설정을 손쉽게 변경할 수 있으며, 민감한 정보가 코드에 노출되지 않도록 할 수 있습니다.

### 설정 파일 예제

```python
from pydantic import BaseSettings, Field
from enum import Enum

class ConfigEnv(str, Enum):
    DEV = "dev"
    PROD = "prod"

class DBConfig(BaseSettings):
    host: str = Field(default="localhost", env="db_host")
    port: int = Field(default=3306, env="db_port")
    username: str = Field(default="user", env="db_username")
    password: str = Field(default="user", env="db_password")
    database: str = Field(default="dev", env="db_database")

class AppConfig(BaseSettings):
    env: ConfigEnv = Field(default="dev", env="env")
    db: DBConfig = DBConfig()

# 예를 들어, dev_config.yaml 파일을 로드하여 기본 설정을 구성할 수도 있음
# (여기서는 환경 변수를 통한 설정 오버라이딩 예제를 함께 설명)
```

### 환경 변수로 설정 오버라이딩

아래 예제는 환경 변수를 이용해 설정 값을 오버라이딩하는 방법을 보여줍니다.

```python
import os

# 환경 변수 설정 (실제 배포 환경에서는 시스템 환경 변수로 관리)
os.environ["env"] = "prod"
os.environ["db_host"] = "mysql"
os.environ["db_username"] = "admin"
os.environ["db_password"] = "PASSWORD"

# 환경 변수 기반 설정 객체 생성
prod_config_with_pydantic = AppConfig()
print("환경:", prod_config_with_pydantic.env)
print("DB 설정:", prod_config_with_pydantic.db.dict())

# cleanup: 필요 시 환경 변수 정리
# os.environ.pop("env")
# os.environ.pop("db_host")
# os.environ.pop("db_username")
# os.environ.pop("db_password")
```

이와 같이 pydantic의 BaseSettings를 사용하면 YAML 파일이나 별도의 설정 파일 없이도, 환경 변수만으로 손쉽게 설정을 관리할 수 있어 보안과 유연성을 동시에 확보할 수 있습니다.

---

## 결론

pydantic은 다음과 같은 이유로 많은 파이썬 개발자와 FastAPI 사용자에게 사랑받고 있습니다.

- **데이터 검증:**  
    간결한 코드로 입력 데이터를 자동으로 검증하고, 상세한 에러 메시지 제공
- **직렬화/역직렬화:**  
    JSON과 같은 포맷으로 손쉽게 데이터를 변환
- **설정 관리:**  
    BaseSettings를 활용해 환경 변수 기반의 안전하고 유연한 설정 관리

---
# Docker로 컨테이너 기반 애플리케이션 개발 및 배포

**Docker는 컨테이너 기술을 기반으로 애플리케이션을 개발, 배포, 실행할 수 있도록 지원하는 오픈 소스 플랫폼입니다.** Docker를 활용하면 개발 환경과 배포 환경 간의 차이로 인한 문제를 해결할 수 있으며, 가상 머신보다 가볍고 빠른 컨테이너를 이용해 일관된 실행 환경을 보장할 수 있습니다.

---

## 1. 컨테이너와 Docker의 필요성

### 컨테이너란?

- **정의:**  
    컨테이너는 애플리케이션과 그 실행에 필요한 라이브러리, 설정 등을 하나의 패키지로 묶어 격리된 환경에서 실행하는 기술입니다.
- **왜 사용할까?**
    - **환경 일관성:**  
        개발할 때의 서버와 배포할 서버는 OS, 환경 변수, 퍼미션 등이 다를 수 있는데, 컨테이너는 이를 모두 소프트웨어화하여 동일한 환경에서 실행할 수 있게 합니다.
    - **가상 머신과의 차이:**  
        가상 머신은 호스트 OS 위에 OS 전체를 실행하기 때문에 무겁지만, 컨테이너는 커널을 공유하여 훨씬 가볍고 빠른 환경을 제공합니다.

### Docker의 역할

- **Docker image:**  
    컨테이너를 실행할 때 사용하는 "템플릿" (Read Only)
- **Docker container:**  
    이미지를 실행하여 만들어진 인스턴스 (Write 가능)

---

## 2. Docker 기본 명령어 정리

- **이미지 다운로드 및 목록 확인**
    
    - `docker pull image_name:tag`  
        → Docker Hub에서 해당 이미지 다운로드
    - `docker images`  
        → 다운로드한 Docker image 목록 확인
- **컨테이너 실행**
    
    - `docker run image_name:tag`  
        → 해당 이미지를 기반으로 컨테이너 생성 및 실행
        - `--name` : 컨테이너 이름 지정
        - `-e` : 환경 변수 설정 (예: `-e MYSQL_ROOT_PASSWORD=1234`)
        - `-d` : 데몬(백그라운드) 모드로 실행
        - `-p host_port:container_port` : 포트 포워딩 (예: `-p 8000:8000`)
        - `-v host_directory:container_directory` : 볼륨 마운트로 저장소 공유
- **컨테이너 상태 확인 및 관리**
    
    - `docker ps`  
        → 현재 실행 중인 컨테이너 목록 확인
    - `docker ps -a`  
        → 중지된 컨테이너까지 모두 확인
    - `docker exec -it CONTAINER_ID /bin/bash`  
        → 실행 중인 컨테이너 내부로 진입
    - `docker stop CONTAINER_ID`  
        → 컨테이너 중지
    - `docker rm CONTAINER_ID`  
        → 중지된 컨테이너 삭제 (실행 중이면 `-f` 옵션 사용)

---

## 3. Docker 설치 및 Docker Hub

### Docker 설치

- **설치 방법:**  
    Docker 공식 홈페이지에서 자신의 운영체제에 맞는 Docker Desktop을 다운로드 및 설치합니다.  
    (설치 완료까지는 시간이 다소 소요될 수 있습니다.)

### Docker Hub

- **개념:**  
    Docker Hub는 Docker image를 저장, 공유, 배포할 수 있는 클라우드 기반 레지스트리 서비스입니다.  
    필요한 이미지를 검색해서 받아 사용할 수 있으며, 자신이 만든 이미지를 업로드하여 공유할 수 있습니다.

---

## 4. Docker Image 생성

Docker 이미지는 Dockerfile을 기반으로 빌드합니다. Dockerfile에는 이미지 빌드에 필요한 모든 설정 정보가 포함됩니다.

### Dockerfile 주요 명령어

- `FROM image_name:tag`  
    → 베이스 이미지 지정 (예: `python:3.9.13-slim`)
- `COPY host_directory container_directory`  
    → 호스트 파일/디렉토리를 컨테이너로 복사
- `WORKDIR container_directory`  
    → 작업 디렉토리 설정 (CMD, RUN 등의 명령어 실행 경로)
- `ENV 환경변수=값`  
    → 컨테이너 내 환경 변수 설정
- `RUN 리눅스_명령어`  
    → 이미지 빌드 시 실행할 명령어 (예: 소프트웨어 설치)
- `CMD ["명령어", "인자"]`  
    → 컨테이너 실행 시 기본으로 실행할 명령어 지정

### 예시 Dockerfile

```dockerfile
FROM python:3.9.13-slim

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

- **설명:**
    - Python 3.9.13-slim 이미지를 베이스로 사용
    - `/code` 디렉토리를 작업 공간으로 설정
    - `requirements.txt` 파일을 복사하여 의존성 설치
    - 전체 코드를 복사 후, uvicorn을 통해 FastAPI 앱 실행

### Docker Image 빌드 및 실행

- **빌드:**
    
    ```bash
    $ docker build -t APIweb:first .
    ```
    
- **이미지 확인:**
    
    ```bash
    $ docker images
    ```
    
- **컨테이너 실행:**
    
    ```bash
    $ docker run -p 8000:8000 APIweb:first
    ```
    
    → 브라우저에서 [http://localhost:8000](http://localhost:8000/) 접속하여 웹 페이지 확인

---

## 5. Docker Image Push

Docker Hub에 이미지를 업로드(푸시)하는 방법은 다음과 같습니다.

1. **로그인:**
    
    ```bash
    $ docker login
    ```
    
2. **태그 변경:**
    
    ```bash
    $ docker tag image_name:tag_name my_id/image_name:tag_name
    ```
    
3. **푸시:**
    
    ```bash
    $ docker push my_id/image_name:tag_name
    ```
    

---

## 6. Docker Image 최적화

ML 모델이 포함된 Docker image는 종종 용량이 크기 때문에 최적화가 필요합니다.

### 최적화 방법

1. **작은 베이스 이미지 사용:**
    - Python 이미지 중 `slim` 또는 `alpine`을 사용  
        (예: `FROM python:3.9.13-slim`)
2. **Multi-stage build:**
    - 빌드에 필요한 단계와 최종 이미지에서 필요한 부분을 분리하여 불필요한 파일 제거
3. **컨테이너 패키징 최적화:**
    - `.dockerignore` 파일로 빌드 시 포함하지 않을 파일 지정
    - 변경 가능성이 낮은 명령어는 Dockerfile 상단에 배치해 캐싱 활용

---

## 7. Docker Compose 사용하기

여러 Docker container를 동시에 실행해야 할 때 Docker Compose를 사용하면 편리합니다. 예를 들어, 데이터베이스 컨테이너와 웹 애플리케이션 컨테이너를 동시에 실행할 수 있습니다.

### 예시 docker-compose.yml 파일

```yaml
version: '3'

services:
  db:
    image: mysql:5.7.12
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: my_database
    ports:
      - "3306:3306"

  app:
    build:
      context: .
    environment:
      DB_URL: mysql+mysqldb://root:root@db:3306/my_database?charset=utf8mb4
    ports:
      - "8000:8000"
    depends_on:
      - db
    restart: always
```

- **설명:**
    - `db` 서비스: MySQL 컨테이너 실행, 환경 변수 설정 및 포트 매핑
    - `app` 서비스: 현재 디렉토리의 Dockerfile을 이용해 빌드, DB URL 환경 변수 지정, `db` 컨테이너에 의존
- **실행:**
    
    ```bash
    $ docker-compose up
    ```
    
    → 여러 컨테이너가 동시에 실행되며, `app` 컨테이너는 `db`가 실행된 이후에 시작됨

---

## 결론

Docker를 이용하면 개발 환경과 배포 환경의 차이를 극복하며, 애플리케이션을 일관된 환경에서 손쉽게 실행할 수 있습니다.

- **컨테이너 기술:**  
    가상 머신보다 가볍고 빠른 실행 환경 제공
- **Docker 명령어:**  
    이미지 다운로드, 컨테이너 실행, 관리 및 삭제 등 다양한 기능 제공
- **Dockerfile과 이미지 최적화:**  
    베이스 이미지 선택, Multi-stage build, .dockerignore 활용 등으로 최적화 가능
- **Docker Compose:**  
    복수의 컨테이너를 쉽게 관리하고 실행할 수 있음
---
# 클라우드 컴퓨팅과 GCP 활용 가이드

클라우드는 인터넷을 통해 IT 자원(서버, 저장소, 데이터베이스, 네트워킹, 소프트웨어 등)을 제공하는 서비스입니다. AWS, GCP, Azure와 같이 다양한 클라우드 서비스 제공업체가 있으며, 각 업체마다 서버, 서버리스 컴퓨팅, 오브젝트 스토리지, 데이터베이스 등 공통 개념은 유사하지만 이름이나 세부 기능이 다릅니다.

이번 포스트에서는 클라우드 서비스의 기본 개념, 특히 object storage와 database의 차이, 그리고 Google Cloud Platform(GCP)의 대표 기능(Compute Engine, Cloud Storage, 방화벽, Cloud Composer)을 실제 사용하는 방법과 Python을 활용한 파일 업로드/다운로드 예제를 살펴보겠습니다.

---

## 1. 클라우드 서비스의 기본 개념

### 주요 서비스 비교

|**서비스**|**AWS**|**GCP**|**Azure**|
|---|---|---|---|
|**Server**|Elastic Compute (EC2)|Compute Engine|Virtual Machine|
|**Serverless**|Lambda|Cloud Function|Azure Function|
|**Stateless container**|ECS|Cloud Run|Container Instance|
|**Object storage**|S3|Cloud Storage|Blob Storage|
|**Database**|Amazon RDS|Cloud SQL|Azure SQL|
|**Data Warehouse**|Redshift|BigQuery|Synapse Analytics|
|**AI platform**|SageMaker|Vertex AI|Azure Machine Learning|
|**Kubernetes**|EKS|GKE|AKS|

- **Server (Computing Service):** 연산(CPU, 메모리, GPU 등) 수행을 위한 서버
- **Serverless computing:** 코드를 제출하면 클라우드에서 자동으로 서버를 실행하는 형태
- **Stateless container:** Docker 이미지를 기반으로 별도의 서버 없이 컨테이너를 실행
- **Object storage vs. Database:**
    - **Object storage:** 비구조적 데이터(파일, 객체)를 대용량(TB~PB급)으로 저장, 전체 파일 단위 접근, 수정 시 파일 전체를 덮어씀 (미디어 파일, 백업 등)
    - **Database:** 구조적 데이터(행, 열 또는 키-값 쌍)를 저장하며, 필드별로 세밀하게 접근 및 수정 가능, 실시간 트랜잭션 및 쿼리 최적화 (고객 데이터, 주문 기록 등)

---

## 2. Google Cloud Platform (GCP) 사용법
![Pasted image 20250311145442.png](/img/user/images/Pasted%20image%2020250311145442.png)
GCP는 대표적인 클라우드 서비스 제공업체로, 다양한 컴퓨팅 및 스토리지 서비스를 제공합니다.

### 2-1. Compute Engine

- **설명:**  
    GCP의 Compute Engine은 VM(가상 머신)을 생성할 수 있는 서비스입니다.
- **사용법:**
    1. GCP에 가입한 후, 메인 대시보드로 이동
    2. 왼쪽 메뉴에서 **Compute Engine**을 클릭하면 초기화 과정이 진행되고, VM 인스턴스 메뉴가 표시됨
    3. **인스턴스 만들기** 버튼을 클릭하여 이름 및 머신 구성(예: e2-micro, 무료 사용 가능 범위)을 선택 후 서버 생성
    4. 생성된 서버의 오른쪽 "연결" 항목에서 **SSH**를 클릭하면 브라우저에서 CLI로 접속할 수 있음
- **주의:**  
    사용하지 않는 서버는 삭제하여 비용 발생을 방지합니다.

또한, GCP에서는 이미 구축된 Docker 이미지 기반의 서버도 사용할 수 있으므로, 원하는 환경(예: PyTorch 등)을 검색 후 인스턴스를 생성할 수 있습니다.

### 2-2. Cloud Storage

- **설명:**  
    GCP의 Cloud Storage는 대용량 파일 및 객체를 저장할 수 있는 서비스입니다.
- **버킷 생성:**
    1. Cloud Storage 탭에서 **버킷 만들기** 클릭
    2. 버킷 이름과 스토리지 클래스를 지정하면 버킷 생성
    3. 생성된 버킷을 클릭하여 파일 및 폴더 업로드 가능

#### Python을 이용한 파일 업로드/다운로드 예제

1. **Google Cloud Storage 파이썬 라이브러리 설치:**
    
    ```bash
    pip install google-cloud-storage
    ```
    
2. **서비스 계정과 키 생성:**
    
    - GCP Cloud Console의 서비스 계정 만들기 페이지에서 대상 프로젝트 선택
    - 서비스 계정 생성 후, 역할을 “소유자”로 설정
    - 서비스 계정 상세 정보에서 키 추가(형식: JSON) 후 다운로드
3. **환경 변수 설정:**
    
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="key_path"
    ```
    
4. **파이썬 코드 예제:**
    
    ```python
    from google.cloud import storage
    
    # 초기화
    bucket_name = "bucket1"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # 파일 업로드
    upload_file_path = "/your/directory/your_file"
    blob = bucket.blob("your_file")
    blob.upload_from_filename(upload_file_path)
    
    # 파일 다운로드
    download_destination = "/your/directory/your_file_downloaded"
    blob.download_to_filename(download_destination)
    ```
    

자세한 내용은 구글 클라우드 공식 문서를 참고하세요.

---

## 3. 방화벽 설정

GCP에서는 VM 인스턴스에 접근할 수 있도록 방화벽 규칙을 설정할 수 있습니다.

1. **방화벽 규칙 만들기:**
    
    - 왼쪽 메뉴의 **네트워킹 > VPC 네트워크 > 방화벽** 클릭
    - 상단 **방화벽 규칙 만들기** 버튼 클릭
    - 이름과 대상 태그 설정, 소스 필터를 IPv4 범위(0.0.0.0/0)로 지정, 프로토콜 TCP의 포트(예: 8888) 설정 후 생성
2. **방화벽 규칙 적용:**
    
    - 규칙을 적용할 VM 인스턴스를 선택 후 수정 버튼 클릭
    - 방화벽 설정에서 **Allow HTTP traffic**, **Allow HTTPS traffic** 체크
    - 네트워크 태그에 생성한 대상 태그를 등록하면 해당 서버에 규칙이 적용됩니다.

---

## 4. Cloud Composer (GCP의 Airflow)

Cloud Composer는 GCP에서 제공하는 관리형 Apache Airflow 서비스입니다. 데이터 파이프라인과 워크플로우를 손쉽게 관리할 수 있습니다.

1. **Cloud Composer 생성:**
    
    - GCP 왼쪽 메뉴에서 **Composer** 선택
    - 상단 **만들기 > Composer 2** 클릭
    - 이름, 서비스 계정, 이미지 버전(안정적인 composer-2.5.5-airflow-2.6.3 권장) 선택
    - 고급 구성에서 Airflow 구성 재정의(예: webserver의 `dag_dir_list_interval`을 30초 등) 입력 후 생성
2. **Composer 환경 확인:**
    
    - 생성된 Composer 환경 목록에서 Airflow 웹서버 링크를 클릭하면 Airflow UI에 접속
    - DAG 폴더 링크를 클릭하면 연결된 Cloud Storage 버킷으로 이동, 여기서 DAG 파일(예: `01-bash-operator.py`)을 저장하면 자동으로 실행됨

---

## 결론

클라우드 서비스는 IT 자원을 손쉽게 제공하여 개발, 배포, 운영의 효율성을 극대화합니다.

- **서비스 비교:** AWS, GCP, Azure는 각각 서버, 서버리스, 오브젝트 스토리지, 데이터베이스 등 공통 개념은 유사하지만 세부 기능과 이름이 다릅니다.
- **GCP 활용:** Compute Engine, Cloud Storage, 방화벽, Cloud Composer 등을 통해 실제 서버 생성부터 파일 저장, 네트워크 보안, 워크플로우 관리까지 다양한 기능을 사용할 수 있습니다.
- **Python 활용:** Cloud Storage와의 연동을 비롯한 다양한 작업을 Python 코드로 손쉽게 자동화할 수 있습니다.

---
# MLflow로 머신러닝 라이프사이클 관리하기

AI 서비스를 개발할 때는 데이터 수집·라벨링·정제부터 다양한 모델과 파라미터로 실험을 진행하고, 가장 성능이 좋은 모델을 배포하는 일련의 과정이 필요합니다. 이 모든 과정은 메타 데이터, 모델 아티팩트, 그리고 사용한 feature나 데이터를 꼼꼼히 기록하고 관리해야 합니다.  
**MLflow**는 머신러닝 모델의 실험, 배포, 관리를 효율적으로 도와주는 플랫폼으로, 다음과 같은 핵심 기능을 제공합니다.

---

## MLflow의 핵심 기능

- **실험 실행/관리/기록:**  
    실험(run)마다 실행 환경, 파라미터, 메트릭, 아티팩트 등을 기록하여 언제, 어떻게 모델이 만들어졌는지 추적할 수 있습니다.
    
- **Registry:**  
    여러 실험에서 생성된 모델을 저장하고 버전 관리할 수 있으며, 팀원과 공유하거나 재사용할 수 있습니다.
    
- **Serving:**  
    Registry에 등록된 모델을 REST API 서버로 배포하여 실시간 예측 서비스를 제공할 수 있습니다.
    

---

## 1. MLflow 설치 및 UI 실행

### 설치

MLflow는 아래 명령어로 설치할 수 있습니다.

```bash
$ pip install mlflow==2.10.0
```

### MLflow UI 실행

다음 명령어로 MLflow 서버를 실행한 후 브라우저에서 [http://localhost:8080](http://localhost:8080/)을 접속하면 MLflow 웹페이지를 확인할 수 있습니다.

```bash
$ mlflow server --host 127.0.0.1 --port 8080
```

---

## 2. Experiment 생성 및 관리

### Experiment란?

Experiment는 하나의 프로젝트를 의미하며, 여러 run(실행)으로 구성됩니다. 예를 들어 'hand_bone_segmentation' 프로젝트를 만들면, 해당 experiment 내에서 여러 번의 모델 학습(run) 기록이 저장됩니다.

### Experiment 생성

아래 명령어를 이용해 새로운 experiment를 생성할 수 있습니다.

```bash
$ mlflow experiments create --experiment-name hand_bone_segmentation
```

experiment가 성공적으로 생성되면, MLflow는 `mlruns`라는 폴더에 관련 기록을 저장하고, 웹 UI에서도 해당 experiment를 확인할 수 있습니다. 생성된 experiment 목록은 다음 명령어로 검색할 수 있습니다.

```bash
$ mlflow experiments search
```

---

## 3. 프로젝트 메타 정보 저장

실험 기록 외에도 프로젝트의 실행 환경, 의존성 등을 관리하기 위해 **MLProject** 파일과 **python_env.yaml** 파일을 작성할 수 있습니다.

### MLProject 파일 예시

```yaml
name: project1

python_env: python_env.yaml

entry_points:
  main:
    parameters:
      regularization: {type: float, default: 0.1}
    command: "python train.py"
```

### python_env.yaml 파일 예시

```yaml
python: "3.9.13"

# 빌드 시 필요한 의존성
build_dependencies:
  - pip
  - setuptools
  - wheel==0.37.1
  
# 프로젝트 실행 시 필요한 의존성
dependencies:
  - mlflow==2.10.0
  - scikit-learn==1.4.0
  - pandas==2.2.0
  - numpy==1.26.3
  - matplotlib==3.8.2
```

이렇게 메타 정보를 기록해두면, 실험 재현성과 관리가 쉬워집니다.

---

## 4. Experiment 실행 (Run)

실제 실험을 실행할 때는 MLflow가 제공하는 커맨드를 사용하여 실행합니다.

```bash
$ mlflow run experiment_directory --experiment-name hand_bone_segmentation
```

- `experiment_directory`에는 MLProject 파일과 `train.py` 등의 코드가 포함된 디렉토리 경로를 지정합니다.
- `-P regularization=0.01`과 같이 `-P` 옵션을 사용하여 파라미터를 전달할 수도 있습니다.

실행(run)이 완료되면 MLflow UI에 run 기록이 추가되고, 아래와 같은 정보가 자동으로 기록됩니다.

- **source:** 실행한 프로젝트의 이름
- **version:** 실행 시의 Git hash 등
- **start & end time:** 실행 시간 기록
- **parameters:** 입력한 파라미터들
- **metrics:** 성능 지표들
- **tags:** 추가 태그 정보
- **artifacts:** 실행 과정에서 생성된 파일들 (예: 모델 파일, 이미지 등)

---

## 5. 로깅 및 Autolog

`train.py` 내에서 직접 로깅할 수 있으며, MLflow는 다양한 프레임워크에 대해 자동 로깅(autolog) 기능을 제공합니다. 예를 들어 scikit-learn의 경우:

```python
import mlflow
import mlflow.sklearn

# 개별 항목 로깅
mlflow.log_param("l1_ratio", 0.1)

# autolog 활성화
mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    model.fit(x, y)
```

이렇게 하면 MLflow가 자동으로 파라미터, 메트릭, 모델 아티팩트를 기록해줍니다.

---

## 6. 모델 아티팩트 다운로드

특정 run에서 저장된 모델 아티팩트를 다운로드하려면, 다음 코드를 사용할 수 있습니다.

```python
from mlflow import artifacts

def download_model(run_id, model_name="model"):
    artifact_uri = f"runs:/{run_id}/{model_name}"
    artifacts.download_artifacts(artifact_uri, dst_path=".")
```

이 함수에 run id와 모델 이름을 전달하면, 해당 모델 파일을 로컬에 다운로드할 수 있습니다.

---

## 결론

MLflow는 머신러닝 실험의 실행, 기록, 관리, 배포를 통합하여 효율적으로 관리할 수 있도록 도와줍니다.

- **실험 실행 및 기록:** run 단위로 파라미터, 메트릭, 아티팩트를 체계적으로 관리
- **메타 데이터 관리:** MLProject와 python_env.yaml을 통해 재현 가능한 환경 구성
- **모델 Registry 및 Serving:** 저장된 모델을 쉽게 배포하고 REST API 서버로 서빙 가능

---

# 모델 평가: Offline과 Online 평가를 통한 최적 AI 서비스 서빙

AI 서비스를 배포할 때 가장 중요한 요소 중 하나는 최적의 성능을 갖춘 모델을 서빙하는 것입니다. 이를 위해 모델의 성능을 지속적으로 평가하고 개선해야 하는데, 평가 환경에 따라 크게 **Offline**과 **Online** 평가로 나뉩니다. 이번 포스트에서는 두 평가 방식의 개념, 방법론, 장단점에 대해 살펴보고, 어떻게 지속적으로 모델 성능을 개선할 수 있는지 알아보겠습니다.

---

## 1. Offline 모델 평가

Offline 평가란 모델을 배포하기 전, 학습 결과가 내는 성능을 과거 데이터 셋(hold-out 데이터, k-fold, bootstrap 등)을 이용해 실험적으로 확인하는 과정입니다.  
주요 방법은 다음과 같습니다.

### 1-1. Hold-out Validation

- **설명:**  
    데이터를 학습 데이터와 검증 데이터로 분리하여 모델을 훈련하고, 검증 데이터로 성능을 평가합니다.
- **장점:**  
    간단하며 빠르게 평가할 수 있음

### 1-2. k-Fold Cross Validation

- **설명:**  
    데이터를 k개의 폴드로 나누어, 각 폴드를 한 번씩 검증 데이터로 사용하고 나머지로 학습합니다.
- **장점:**  
    데이터 분할에 따른 편향을 줄이고, 전체 데이터에 대해 모델의 일반화 성능을 평가할 수 있음

### 1-3. Bootstrap Resampling

- **설명:**  
    원본 데이터에서 중복을 허용하여 랜덤 샘플을 여러 번 추출한 후, 각 부분 집합으로 모델을 반복 훈련 및 평가합니다.
- **장점:**  
    데이터의 변동성에 따른 모델의 성능을 보다 견고하게 평가할 수 있음

Offline 평가는 주로 배포 전 모델의 성능을 객관적으로 검증하는 데 사용되며, 다양한 평가 방법을 통해 모델의 일반화 능력을 확보하는 것이 중요합니다.

---

## 2. Online 모델 평가

Online 평가는 모델을 실제 서비스 환경에 배포한 후, 실시간 사용자 데이터를 기반으로 성능을 평가하는 방법입니다. 실제 운영 환경에서의 성능, 사용자 반응, 트랜잭션 등 다양한 요소를 반영할 수 있습니다.

### 2-1. A/B Test
![Pasted image 20250311150337.png](/img/user/images/Pasted%20image%2020250311150337.png)
- **설명:**  
    현재 모델과 새로운 모델(또는 여러 버전)을 동시에 배포하여 사용자 그룹별로 각각 다른 모델을 경험하게 하고, 성능 및 사용자 행동을 비교 분석합니다.
- **장점:**
    - 실제 사용자 데이터를 기반으로 성능 차이를 직접 확인할 수 있음
    - 통계적 분석을 통해 신뢰성 있는 결과 도출 가능
    - 사용자 행동을 직접 관찰하여 실질적인 효과 평가 가능
- **단점:**
    - 충분한 데이터를 수집하는 데 시간이 소요됨
    - 사용자 그룹 분할로 인해 기존 사용자 경험이 일부 방해될 수 있음

### 2-2. Canary Test

- **설명:**  
    새로운 모델이나 시스템을 소규모 사용자 집단(“카나리아 그룹”)에게 먼저 배포하여 안정성을 평가한 후, 문제가 없으면 점진적으로 배포 범위를 확대합니다.
- **장점:**
    - 초기 단계에서 문제를 발견하여 전면적인 서비스 장애를 방지할 수 있음
    - 소규모 그룹으로부터 빠르게 피드백 수집 가능
    - 점진적인 전환으로 리스크가 낮음
- **단점:**
    - 모든 사용자에게 배포하기까지 시간이 오래 걸릴 수 있음
    - 초기 소규모 그룹이 전체 사용자 집단을 대표하지 못할 가능성이 있음(샘플 바이어스)

### 2-3. Shadow Test

- **설명:**  
    새로운 모델을 실제 서비스에 반영하지 않고, 실시간 트래픽을 복사하여 별도로 모델 성능을 평가하는 방법입니다. 기존 모델은 실제 요청을 처리하고, 동시에 새로운 모델도 동일한 요청을 처리하지만 그 결과는 사용자에게 노출되지 않습니다.
- **장점:**
    - 실제 사용자 경험에 전혀 영향을 미치지 않음
    - 실시간 데이터를 기반으로 실제 운영 환경에서 성능 검증 가능
    - 모델의 문제점을 사전에 발견하여 수정할 수 있음
- **단점:**
    - 트래픽 복사 및 평가를 위한 추가 리소스가 필요함
    - 사용자 반응을 직접적으로 피드백받지 못함

---

## 3. 지속적 개선의 중요성

Offline 평가와 Online 평가를 반복하면서 모델을 지속적으로 개선하는 것이 매우 중요합니다.

- **Offline 평가:**  
    개발 단계에서 여러 실험을 통해 모델의 일반화 성능을 확보하고, 초기 성능 기준을 설정합니다.
- **Online 평가:**  
    실제 사용자 데이터를 통해 모델의 성능을 실시간으로 모니터링하고, 배포 후의 문제점을 신속하게 발견할 수 있습니다.

이러한 반복적 평가와 피드백 과정을 통해 최적의 모델을 서빙함으로써, AI 서비스의 품질과 사용자 만족도를 높일 수 있습니다.

---

## 결론

모델 평가 전략은 AI 서비스의 성공적인 배포와 운영에 필수적입니다.

- **Offline 평가**는 모델의 기본 성능과 일반화 능력을 검증하는 데 사용되고,
- **Online 평가**는 실제 서비스 환경에서 모델의 성능을 확인하며,  
    A/B test, Canary test, Shadow test와 같은 다양한 기법을 통해 지속적인 개선과 안정성을 도모합니다.

---