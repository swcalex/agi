# Program Structure: Project MLP (v0.2.1)

## 1. 개요 (Overview)
본 프로젝트는 XOR 논리 문제를 해결하는 MLP를 메뉴 기반 인터페이스를 통해 제어한다. 사용자는 모델 초기화, 학습, 추론을 독립적으로 수행할 수 있으며, 기본 구현을 완료한 후 지속적인 구조 개선을 진행 중이다.

## 2. 업데이트 내역 (Release Notes)
**Current Version: v0.2.1** (Previous: v0.2.0)

이번 업데이트에서는 코드의 실제 구현 형태와 설계 문서 간의 불일치를 해결하고, 협업 효율성을 높이기 위한 문서 구조 개선에 집중하였다.

- **[설계 수정] Module B-2 정의 변경**: 
    - (기존) 독립적인 'Inference' 모듈로 기술.
    - (변경) `main.py` 내부에서 동작하는 **'전방 연산 서브 메뉴(Inference Sub-menu)'**로 구조를 명확히 하여 코드와 설계의 일치성 확보.
- **[문서 최적화] AI 협업 프로세스 분리**: 
    - 프로젝트 외부에 있던 'AI 협업 개발 시스템' 내용을 별도의 규칙 파일(`ai_rules.md`)로 이관하여 본 문서는 프로그램 구조 자체에만 집중하도록 경량화.

## 3. 시스템 구성 요소
- `program_structure.md`: 설계 및 의사결정 기록 (Source of Truth).
- `model.py`: MLP 클래스 (초기화, 전방 연산, 역전파 모듈 정의).
- `main.py`: 루프 기반 메뉴 인터페이스(Initialization, Forward Propagation, Backpropagation & Training) 및 실행 제어.

## 4. 모듈 상세 설계 (v0.2.1 기준)

### 4.1. [Module A] 핵심 연산 로직 (model.py)
- **초기화 (Initialization)**: 가중치($W$)와 편향($b$)을 분리하여 초기화. $W$는 정규분포(std=0.01), $b$는 0으로 설정.
- **전방 연산 (Forward Propagation)**: 시그모이드($\sigma$) 활성화 함수를 사용하여 은닉층(4개 노드)과 출력층(1개 노드) 연산 수행.
- **역전파 (Backpropagation)**: MSE 손실 함수와 시그모이드 미분을 연쇄 법칙으로 결합하여 가중치 업데이트 ($\eta = 0.1$).

### 4.2. [Module B-1] 메인 메뉴 시스템 (main.py)
- **역할**: 사용자로부터 명령을 입력받아 각 기능을 실행하고 루프를 통해 메뉴로 복귀한다.
- **구성**:
    1. **Initialization**: MLP 객체를 생성하여 가중치를 초기 상태로 설정.
    2. **Forward Propagation**: 추론을 위한 서브 메뉴로 진입.
    3. **Backpropagation & Training**: 사용자에게 에폭(Epoch) 수를 입력받아 학습 진행. 각 에폭마다 데이터 셔플링(Shuffling) 적용 및 Loss 로그 출력.
    4. **Exit**: 프로그램 종료.

### 4.3. [Module B-2] 전방 연산 서브 메뉴 (main.py)
- **설명**: `main.py` 내부의 중첩 루프로 구현되며, 모델 추론 기능을 담당한다.
- **구성**:
    1. **Input**: 사용자로부터 두 개의 숫자(0 또는 1)를 입력받아 예측값 출력. 0.5 이상은 1, 미만은 0으로 이진화하여 표시.
    2. **Exit to Main Menu**: 메인 메뉴로 복귀.