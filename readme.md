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

## 3. 현재 단계: v0.5.0 (4-Bit Binary to Decimal)
* **Objective**: 4비트 입력($0000_2 \sim 1111_2$)을 받아 10진수($0 \sim 15$)로 변환하는 회귀적 능력을 검증한다.
* **Key Feature**:
    * 4D Input Space Visualization (Slicing Hypercubes)
    * Regression via Probability (BCE Loss Experiment)