# Project AI: Mathematical Foundation for AGI

## 1. 개요 (Overview)
본 프로젝트는 코드 작성을 넘어, **"지능의 알고리즘을 수학적으로 규명하고 구현하는 것"**을 목표로 한다. 
PyTorch 기반의 신경망을 다루지만, 핵심 가치는 코드가 아닌 **수리적 모델링(`algorithm_spec.md`)**에 있다.

## 2. 프로젝트 구조 (Structure)

### 📄 문서 (Documents)
* **`README.md`**: 프로젝트 개요 및 지도.
* **`algorithm_spec.md`**: **[핵심]** AI 모델의 수학적 정의 및 논리 설계도. (Code is just a tool, Math is the core.)
* **`ai_rule.md`**: Architect(User)와 Engine(AI)의 협업 프로토콜.

### 🛠 구현체 (Implementation)
* `model.py`: `algorithm_spec.md`의 수식을 PyTorch로 번역한 신경망 모듈.
* `main.py`: 학습 및 실험을 수행하는 실행 환경.
* `visualization_utils.py`: 내부 동작을 검증하기 위한 시각화 도구 (관측 창).

## 3. 현재 단계: v0.5.1 (Monitoring & Transparency)
* **Objective**: 학습 상태의 명확한 인지와 블랙박스(Hidden Layer) 내부의 정보 처리 과정을 시각화한다.
* **Key Feature**:
    * **Metric**: BCE Loss(학습용)와 MAE(관측용)의 이원화.
    * **Visualization**: Hidden Layer Activation Map (Input vs Neuron Heatmap).
    * **Target**: 4비트 정수 회귀 문제의 완전한 해석 가능성 확보, 최적의 default epoch 값 재설정