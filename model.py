import numpy as np

class MLP:
    def __init__(self, n_in=2, n_h=4, n_out=1, lr=0.1):
        """
        초기 설정 및 하이퍼파라미터 정의
        """
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = n_out
        self.lr = lr
        
        # 모델 초기화 실행 (Module A-1)
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        [Module A-1] 가중치 및 편향 초기화
        - 가중치: 정규분포 (std=0.01)
        - 편향: 0
        """
        np.random.seed(42)  # 재현성을 위한 시드 고정
        
        # Layer 1 (Input -> Hidden)
        self.W1 = np.random.normal(0, 0.01, (self.n_in, self.n_h))
        self.b1 = np.zeros((1, self.n_h))
        
        # Layer 2 (Hidden -> Output)
        self.W2 = np.random.normal(0, 0.01, (self.n_h, self.n_out))
        self.b2 = np.zeros((1, self.n_out))

    def sigmoid(self, x):
        """
        활성화 함수: 시그모이드
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        시그모이드 함수의 미분 (역전파용)
        """
        return x * (1 - x)

    def forward(self, x):
        """
        [Module A-2] 전방 연산
        """
        # 은닉층 연산: z1 = W1*x + b1, a1 = sigmoid(z1)
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # 출력층 연산: z2 = W2*a1 + b2, a2 = sigmoid(z2)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2

    def backward(self, x, y, output):
        """
        [Module A-4] 역전파 및 가중치 업데이트
        """
        # 1. 출력층 오차 계산 (MSE 미분 기반)
        error_output = y - output
        delta_output = error_output * self.sigmoid_derivative(output)
        
        # 2. 은닉층 오차 계산
        error_hidden = delta_output.dot(self.W2.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.a1)
        
        # 3. 가중치 및 편향 업데이트 (W = W + lr * delta * input)
        # Note: 설계의 'W - lr * Grad'와 동일한 방향
        self.W2 += self.a1.T.dot(delta_output) * self.lr
        self.b2 += np.sum(delta_output, axis=0, keepdims=True) * self.lr
        
        self.W1 += x.T.dot(delta_hidden) * self.lr
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * self.lr

    def get_loss(self, y_true, y_pred):
        """
        [Module A-3] 손실 함수 (MSE)
        """
        return 0.5 * np.mean((y_true - y_pred)**2)
