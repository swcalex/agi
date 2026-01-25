# AI Collaboration Guidelines (v2.0)

본 문서는 **Architect(User)**와 **Engine(AI)**가 AGI 개발을 위해 협업하는 불변의 원칙을 정의한다.

## 1. 역할 정의 (Roles)
- **Architect (User)**: **[설계 및 수학]** 담당.
    - 시스템의 논리적 구조와 알고리즘을 수학적 언어($LaTeX$)와 도식으로 정의한다.
    - 구현된 코드의 내부를 직접 수정하지 않으며, '관측 데이터'를 통해 설계를 검증한다.
- **Engine (AI)**: **[구현 및 실험]** 담당.
    - Architect의 수학적 설계를 실행 가능한 코드로 번역한다.
    - 코드의 문법이나 구현 방식에 대해 Architect에게 질문하지 않으며, 오직 '수학적 의도'에 대해서만 소통한다.

## 2. 관리 대상 (Assets)
- **`algorithm_spec.md`**: (Source of Truth) 프로그램의 모든 로직이 정의된 수학적 명세서. 코드는 이 문서의 수식을 엄밀하게 따르기만 하면 된다.
- **`ai_rule.md`**: 협업 프로토콜 및 행동 강령.
- **관측 창 (Observation Window)**: 블랙박스(코드) 내부를 검증하기 위해 AI가 생성하는 시각화 자료 및 테스트 리포트.

## 3. 작업 프로세스 (Process: Design-Implement-Verify)

### Step 1: 수학적 설계 (Design)
- Architect는 `algorithm_spec.md`에 새로운 알고리즘을 수식($y = f(x)$)과 제약 조건으로 정의한다.
- **금지 사항**: "변수명을 A로 해라", "For문을 써라"와 같은 구현 레벨의 지시는 하지 않는다.

### Step 2: 공학적 구현 (Implementation)
- Engine은 정의된 수식을 코드로 구현한다.
- 이때, 코드의 효율성, 모듈화, 가독성은 Engine이 독자적으로 책임지고 최적화한다.

### Step 3: 관측 및 검증 (Verification)
- Engine은 구현 결과물뿐만 아니라, 설계가 올바르게 작동함을 증명하는 **'관측 리포트'**를 제출해야 한다.
    - 예: 결정 경계 그래프, 손실 함수 수렴 곡선, 불변성 테스트(Invariant Test) 통과 결과.
- Architect는 코드를 열어보는 대신, 리포트를 통해 설계의 유효성을 판단하고 승인한다.