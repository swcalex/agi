# Program Structure: Project MLP (Evolutionary Development)

## 1. 개요 (Overview)
본 프로젝트는 MLP의 수학적 기초부터 프레임워크 최적화까지의 과정을 단계별로 정복한다. 각 단계는 Git Tag 또는 커밋 버전으로 관리하며, 이전 단계의 코드를 기반으로 점진적으로 업데이트한다. 특히 본 문서는 프로젝트의 설계 및 의사결정의 핵심 지표로서 모든 파일 생성의 기준점이 된다.

## 2. 개발 로드맵 및 버전 전략 (Milestones)

### Phase 0: Initial Design (v0.1.0) - [현재 단계]
- **목표**: 프로젝트 방향성 설정 및 설계 문서 확정
- **주요 작업**: `program_structure.md` 작성 및 초기화

### Phase 1: Math & Scratch (v1.0.0)
- **목표**: 외부 라이브러리 최소화(NumPy) 및 핵심 수학 원리 구현
- **주요 모듈**:
    - `core.nn`: Linear Layer, Activation (Sigmoid, ReLU)
    - `core.loss`: MSE, CrossEntropy
    - `core.optimizer`: SGD (Basic)
- **작업 내용**: Forward/Backward Propagation 로직 완성 및 간단한 Logic Gate(XOR 등) 테스트

### Phase 2: Optimization & Scaling (v1.1.0)
- **목표**: 규모 확장에 따른 성능 문제 극복
- **추가 내용**: 가중치 초기화(Xavier, He), Batch Normalization, Adam Optimizer 등 적용

### Phase 3: Framework Integration (v2.0.0)
- **목표**: PyTorch/JAX 등 현대적 프레임워크로의 전환 및 성능 비교
- **추가 내용**: GPU 가속 적용 및 대용량 데이터셋(MNIST 등) 처리

## 3. 시스템 구성 요소
- `program_structure.md`: 프로젝트의 설계 및 버전 관리 기준 문서
- `main.py`: 프로그램 실행 및 테스트 진입점
- `model.py`: MLP 네트워크 구조 정의
- `utils.py`: 데이터 전처리 및 시각화 도구
