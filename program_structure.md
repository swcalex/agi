# Program Structure: Project MLP (v0.2.0)

## 1. 개요 (Overview)
본 프로젝트는 XOR 논리 문제를 해결하는 MLP를 메뉴 기반 인터페이스를 통해 제어한다. 사용자는 모델 초기화, 학습, 추론을 독립적으로 수행할 수 있으며, v0.2.0에서는 설계와 Python(NumPy) 기반의 실제 코드 구현을 모두 완료한다.

## 2. 개발 로드맵 (Milestones)
- **Phase 0: Initial Design (v0.1.0)**: 프로젝트 방향성 및 기본 설계 문서 확정.
- **Phase 1: Design & Implementation (v0.2.0)**: [현재 단계] XOR Solver 설계 및 메뉴 시스템을 포함한 코드 구현.
- **Phase 2: Optimization & Refactoring (v1.0.0)**: 코드의 구조적 개선 및 최적화 기법 적용.

## 3. 시스템 구성 요소
- `program_structure.md`: 설계 및 의사결정 기록 (Source of Truth).
- `model.py`: MLP 클래스 (초기화, 전방 연산, 역전파 모듈 정의).
- `main.py`: 루프 기반 메뉴 인터페이스(Initialization, Forward Propagation, Backpropagation & Training) 및 실행 제어.

## 4. 모듈 상세 설계 (v0.2.0 기준)

### 4.1. [Module A] 핵심 연산 로직 (model.py)
- **초기화 (Initialization)**: 가중치($W$ )와 편향($b$ )을 분리하여 초기화. $W$ 는 정규분포(std=0.01), $b$ 는 0으로 설정.
- **전방 연산 (Forward Propagation)**: 시그모이드($\sigma$ ) 활성화 함수를 사용하여 은닉층(4개 노드)과 출력층(1개 노드) 연산 수행.
- **역전파 (Backpropagation)**: MSE 손실 함수와 시그모이드 미분을 연쇄 법칙으로 결합하여 가중치 업데이트 ($\eta = 0.1$ ).

### 4.2. [Module B-1] 메인 메뉴 시스템 (main.py)
- **역할**: 사용자로부터 명령을 입력받아 각 기능을 실행하고 루프를 통해 메뉴로 복귀한다.
- **구성**:
    1. **Initialization**: MLP 객체를 생성하여 가중치를 초기 상태로 설정.
    2. **Forward Propagation**: 추론을 위한 서브 메뉴로 진입.
    3. **Backpropagation & Training**: 사용자에게 에폭(Epoch) 수를 입력받아 학습 진행. 각 에폭마다 데이터 셔플링(Shuffling) 적용 및 Loss 로그 출력.
    4. **Exit**: 프로그램 종료.

### 4.3. [Module B-2] 전방 연산 서브 메뉴 (Inference)
- **구성**:
    1. **Input**: 사용자로부터 두 개의 숫자(0 또는 1)를 입력받아 예측값 출력. $0.5$ 이상은 1, 미만은 0으로 이진화하여 표시.
    2. **Exit to Main Menu**: 메인 메뉴로 복귀.
