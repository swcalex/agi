# Program Structure: Project AI (v0.4.0)

## 1. 개요 (Overview)
본 프로젝트('AI')는 AGI(Artificial General Intelligence) 구현을 위한 기초 단계로, PyTorch 프레임워크를 기반으로 확장 가능한 신경망 아키텍처를 구축한다. 기존의 'MLP' 프로젝트를 계승하되, `numpy` 기반의 수동 구현에서 벗어나 `PyTorch`의 자동 미분(Autograd)과 모듈화된 구조를 도입하여 향후 대규모 모델 및 다양한 감각 입력(Multimodal) 확장에 대비한다.

## 2. 업데이트 내역 (Release Notes)
**Current Version: v0.4.0 (Planning)**

* **[Project Rename]**: 프로젝트 명을 `MLP`에서 `AI`로 변경하여 범용성을 강조.
* **[Core Engine 교체]**: 핵심 연산 라이브러리를 `NumPy`에서 `PyTorch`로 전면 교체.
* **[Module A 리팩토링]**: 수동 역전파 함수를 제거하고, `torch.nn.Module` 기반의 클래스 구조로 전환.
* **[Wide Layer 전략]**: 은닉층(Hidden Layer)의 노드를 **200개**로 대폭 확장하고, 이에 따른 학습 불안정을 방지하기 위해 **Xavier Initialization**을 적용.

## 3. 시스템 구성 요소
* `ai_rule.md`: AI 협업 작업 가이드라인 (Source of Truth for Collaboration).
* `program_structure.md`: 설계 및 의사결정 기록 (Source of Truth for Architecture).
* `model.py`: [Module A] PyTorch 기반의 신경망 모델 클래스 정의.
* `main.py`: [Module B] CLI 메뉴 인터페이스, 학습 루프 제어, 데이터 텐서 관리.
* `visualization_utils.py`: [Module C] 시각화 도구 (PyTorch 모델 호환).

## 4. Module Detail Design

### 4.1. [Module A] 신경망 모델 (model.py)
* **역할**: `torch.nn.Module`을 상속받아 신경망의 구조(Layers)와 순전파(Forward) 흐름을 정의한다.
* **구조 (Class: `SimpleMLP`)**:
    * **Hyperparameters**:
        * Input Size: 2 (XOR)
        * Hidden Size: **200 (Fixed)** - 폭이 넓은 신경망(Wide Network) 실험.
        * Output Size: 1
    * `__init__(self)`:
        * `nn.Linear` 레이어를 사용하여 네트워크 층(Input -> Hidden -> Output)을 정의.
        * **초기화 전략**: 노드가 200개로 늘어남에 따라 출력 분산이 커지는 것을 막기 위해, 가중치에 **Xavier (Glorot) Initialization**을 명시적으로 적용한다.
    * `forward(self, x)`:
        * 입력 텐서 `x`를 받아 은닉층과 활성화 함수(`Sigmoid`)를 거쳐 최종 출력(Probability)을 반환.
* **특이사항**: 역전파(Backward) 로직은 작성하지 않으며, PyTorch 내부의 Autograd를 활용한다.

### 4.2. [Module B] 메인 컨트롤러 (main.py)
* **역할**: 사용자 인터페이스(CLI)를 제공하고, 데이터 전처리(Tensor 변환), 모델 인스턴스 생성, 최적화 도구(Optimizer) 설정 및 학습 루프를 관리한다.
* **구성**:
    1.  **Initialization**:
        * `SimpleMLP` 모델 객체 생성.
        * 데이터셋을 `torch.Tensor` (float32) 형태로 변환.
        * 손실 함수: `nn.BCELoss` (Binary Cross Entropy).
        * 옵티마이저: `torch.optim.Adam` (학습 효율 증대).
    2.  **Forward Propagation**: 입력 텐서를 모델에 통과시켜 예측값 확인. `torch.no_grad()` 컨텍스트 활용.
    3.  **Backpropagation & Training**:
        * 표준 학습 루프 구현: `optimizer.zero_grad()` -> `loss.backward()` -> `optimizer.step()`.
        * 에폭(Epoch) 진행 상황 및 Loss 출력.
    4.  **Visualization**: Module C를 호출. 이때 모델 객체와 텐서 데이터를 넘김.
    5.  **Exit**: 종료.

### 4.3. [Module C] 시각화 모듈 (visualization_utils.py)
* **역할**: 학습된 PyTorch 모델의 결정 경계를 시각화한다.
* **변경 사항**:
    * PyTorch 모델은 `Tensor` 입력을 받으므로, `numpy` 데이터를 `torch.Tensor`로 변환하여 주입하는 래퍼(Wrapper) 로직이 필요하다.
    * 결과값은 `.detach().numpy()`를 통해 다시 NumPy 배열로 변환하여 `matplotlib`에 전달한다.
* **주요 기능**:
    * `plot_decision_boundary(model, X, y)`:
        * 모델 객체(`model`) 자체를 인자로 받아 내부에서 추론 및 시각화 수행.
        * 디자인: `RdBu` 컬러맵, 임계값(0.5) 검은 실선 강조, 파일 저장(`decision_boundary.png`) 유지.