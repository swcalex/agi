# Algorithm Specification: v0.5.1 (Monitoring & Transparency)

## 1. 모델 정의 (Model Definition)
본 모델은 4비트 이진 입력 $\mathbf{x} \in \{0, 1\}^4$를 받아, 10진수 값에 대응하는 확률 $y \in (0, 1)$을 출력하는 MLP이다.

### 1.1 구조 (Architecture)
* **Input Layer**: $D_{in} = 4$
    * Input Vector: $\mathbf{x} = [x_3, x_2, x_1, x_0]$ (MSB -> LSB 순서)
* **Hidden Layer**: $D_{hidden} = 200$ (Wide Strategy)
* **Output Layer**: $D_{out} = 1$

### 1.2 순전파 수식 (Forward Propagation)
$$
\begin{aligned}
\mathbf{h} &= \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \\
y &= \sigma(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)
\end{aligned}
$$
* $\sigma(z) = \frac{1}{1 + e^{-z}}$ (Sigmoid)

### 1.3 출력의 해석 (Interpretation)
$$
\hat{d} = \text{round}(y \times 15)
$$

## 2. 학습 알고리즘 (Learning Algorithm)

### 2.1 역전파 (Backpropagation)
$$
\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial \mathbf{W}}
$$

### 2.2 최적화 목표 (Optimization Goal)
* **Objective**: Binary Cross Entropy (BCE)
  $$L_{BCE} = - \frac{1}{N} \sum_{i=1}^{N} [t_i \log(y_i) + (1-t_i) \log(1-y_i)]$$
* **Target Scaling**: $t_i = \frac{d_{true}}{15}$

### 2.3 관측 지표 (Monitoring Metric) [NEW]
학습의 직관적 파악을 위해 MAE(Mean Absolute Error)를 보조 지표로 사용한다. 역전파에는 관여하지 않는다.
$$
L_{MAE} = \frac{1}{N} \sum_{i=1}^{N} | y_i - t_i |
$$
* **판단 기준**: $L_{MAE} \approx 0$이면 학습 완료로 간주한다.

## 3. 초기화 전략 (Initialization)
* **Weights**: Xavier (Glorot) Normal Initialization.
* **Bias**: Zero Initialization.

## 4. 시각화 명세 (Visualization Spec) [NEW]
블랙박스 내부를 해석하기 위해 은닉층의 활성화 패턴을 시각화한다.

### 4.1 Hidden Layer Activation Map
입력 공간 전체($\mathbf{X}_{all} \in \mathbb{R}^{16 \times 4}$)에 대한 은닉층 반응($\mathbf{H}$)을 히트맵으로 표현한다.
$$
\mathbf{H} = \sigma(\mathbf{X}_{all} \mathbf{W}_1^T + \mathbf{b}_1) \in \mathbb{R}^{16 \times 200}
$$
* **Axis X**: Hidden Units ($0 \sim 199$)
* **Axis Y**: Input Integers ($0 \sim 15$)
* **Value**: Activation Degree ($0.0 \sim 1.0$)