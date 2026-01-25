# Algorithm Specification: v0.5.0 (4-Bit Binary to Decimal)

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
* $\mathbf{W}_1 \in \mathbb{R}^{200 \times 4}, \mathbf{b}_1 \in \mathbb{R}^{200}$
* $\mathbf{W}_2 \in \mathbb{R}^{1 \times 200}, \mathbf{b}_2 \in \mathbb{R}^{1}$

### 1.3 출력의 해석 (Interpretation)
모델의 출력 $y$는 $0$부터 $15$까지의 정수 값 $d$로 변환된다.
$$
\hat{d} = \text{round}(y \times 15)
$$
* 즉, 결정 경계(Decision Boundary)는 $y$ 공간에서 $1/30, 3/30, ..., 29/30$ 지점에 형성된다.

## 2. 학습 알고리즘 (Learning Algorithm)
본 모델은 **역전파 알고리즘(Backpropagation)**을 통해 파라미터를 갱신한다.

### 2.1 역전파 (Backpropagation)
$$
\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial \mathbf{W}}
$$

### 2.2 최적화 목표 (Optimization Goal)
* **Objective**: Binary Cross Entropy (BCE) 최소화 (Experimental)
  $$L = - \frac{1}{N} \sum_{i=1}^{N} [t_i \log(y_i) + (1-t_i) \log(1-y_i)]$$
* **Target Scaling**: 정수 정답 $d_{true} \in \{0, ..., 15\}$는 다음과 같이 스케일링되어 $t_i$로 사용된다.
  $$t_i = \frac{d_{true}}{15}$$

## 3. 초기화 전략 (Initialization)
* **Weights**: Xavier (Glorot) Normal Initialization.
* **Bias**: Zero Initialization.