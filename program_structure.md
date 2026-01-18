# Program Structure: Project MLP (v0.3.0)

## 1. 개요 (Overview)
본 프로젝트는 XOR 논리 문제를 해결하는 MLP를 메뉴 기반 인터페이스를 통해 제어한다. 사용자는 모델 초기화, 학습, 추론을 독립적으로 수행할 수 있으며, 기본 구현을 완료한 후 지속적인 구조 개선을 진행 중이다.

## 2. 업데이트 내역 (Release Notes)
**Current Version: v0.3.0**

[기능 추가] 결정 경계 시각화 (Visualization)**: `matplotlib`을 이용해 학습된 모델의 결정 경계(Decision Boundary)를 시각화.

[구조 변경] Module C 신설**: 시각화 로직 분리 (`visualization_utils.py`).

## 3. 시스템 구성 요소
- 'ai_rule.md': ai 협업 작업 가이드라인. 이는 관리자 및 ai 모두 지켜야 한다.
- 'program_structure.md': 설계 및 의사결정 기록 (Source of Truth).
- 'model.py': MLP 클래스 (초기화, 전방 연산, 역전파 모듈 정의).
- 'main.py': 루프 기반 메뉴 인터페이스(Initialization, Forward Propagation, Backpropagation & Training) 및 실행 제어.
- visualization_utils.py # [Module C] 시각화 도구

## 4. Module Detail Design

### 4.1. [Module A] 핵심 모델 로직 (model.py)
* **역할**: MLP의 연산(수학적 계산)만을 담당한다. 상태를 저장하는 클래스는 사용하지 않으며, 가중치(weights)를 인자로 받아 결과를 반환하는 순수 함수(Pure Function) 형태를 지향한다.
* **주요 함수**:
    * `init_params(input_size, hidden_size, output_size)`: 가중치와 편향을 랜덤 초기화하여 딕셔너리로 반환.
    * `forward(params, x)`: 입력 `x`에 대해 예측값 `y_pred`와 은닉층 값(역전파용) 반환.
    * `backward(params, grads, x, y, y_pred, hidden)`: 기울기(gradient) 계산.
    * `update_params(params, grads, learning_rate)`: 경사 하강법 적용.

### 4.2. [Module B-1] 메인 메뉴 시스템 (main.py)
* **역할**: 사용자와 상호작용(CLI)하고 전체 프로그램의 흐름을 제어한다. `model.py`의 함수들을 호출하여 학습 상태를 관리한다.
* **구성**:
    1.  **Initialization**: 파라미터 초기화 (XOR 문제 기준: 2-input, n-hidden, 1-output).
    2.  **Forward Propagation**: 현재 가중치로 입력값 테스트.
    3.  **Backpropagation & Training**: 에폭(Epoch) 수를 입력받아 학습 루프 실행.
    4.  **Visualization (New)**: 현재 모델의 결정 경계를 시각화하여 팝업 출력.
    5.  **Exit**: 종료.

### 4.3. [Module B-2] 데이터 핸들링 (main.py 내부 혹은 별도)
* 현재는 XOR 데이터(`[[0,0], [0,1], [1,0], [1,1]]`)를 하드코딩하여 사용.
* 추후 파일 입출력 등이 필요할 경우 분리 고려.

### 4.4. [Module C] 시각화 모듈 (visualization_utils.py)
* **역할**: `matplotlib` 라이브러리를 사용하여 모델의 추론 결과와 결정 경계를 시각적으로 표현하고 이미지 파일로 저장한다.
* **설계 고려사항 (확장성)**:
    * 현재는 2D 입력(XOR)에 최적화된 등고선(Contour plot)을 그린다.
    * **고차원 대응 전략**: 추후 입력 노드가 3개 이상으로 늘어날 경우, 사용자가 선택한 두 개의 차원(축)을 기준으로 단면(Slice)을 시각화하거나, 나머지 차원을 0으로 고정하는 방식을 채택한다.
* **주요 기능**:
    * `plot_decision_boundary(params, predict_func, X, y)`:
        * **격자 생성**: 입력 공간(Input Space) 전체를 커버하는 메쉬그리드(Meshgrid)를 생성한다.
        * **예측 수행**: 최종 출력 노드(Output Node)의 확률값(0~1)을 계산한다.
        * **시각화 스타일 (Design Update)**:
            * **배경색**: 기존의 복잡한 색상 대신 **'RdBu' (Red-Blue)** 컬러맵을 사용하여 0(Red)과 1(Blue)의 영역을 직관적이고 심플하게 표현한다.
            * **결정 경계선**: 출력값이 정확히 **0.5(임계값)**가 되는 지점에 **두꺼운 검은색 실선**을 그려 경계를 명확히 표시한다.
            * **데이터 오버레이**: 실제 학습 데이터(`X`)를 산점도로 표시하여 모델의 분류 현황과 비교한다.
        * **파일 저장**: 결과물은 화면 출력 대신 `decision_boundary.png`라는 이름의 파일로 저장하며, 실행 시마다 **기존 파일을 덮어쓴다(Overwrite)**.
* **특이사항 및 제약 (Behavioral Notes)**:
    * **학습 전 시각화**: 초기화 직후(학습 전)에는 가중치가 작아 모델의 모든 출력값이 0.5 부근에 머물 수 있다. 이 경우 0.5 임계선을 넘나들지 못해 **검은색 결정 경계선이 나타나지 않을 수 있으며**, 이는 정상적인 동작이다. (학습이 진행됨에 따라 경계선이 생성됨)