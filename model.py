import numpy as np

def sigmoid(x):
    """시그모이드 활성화 함수"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """시그모이드 함수의 미분: f(x) * (1 - f(x))"""
    return x * (1 - x)

def init_params(input_size, hidden_size, output_size):
    """
    가중치와 편향을 랜덤 초기화하여 딕셔너리로 반환
    """
    np.random.seed(42) # 재현성을 위한 시드 고정
    params = {
        'W1': np.random.randn(input_size, hidden_size),
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size),
        'b2': np.zeros((1, output_size))
    }
    return params

def forward(params, x):
    """
    전방 연산 (Forward Propagation)
    반환값: y_pred (최종 출력), hidden (역전파용 은닉층 출력)
    """
    # 은닉층 계산
    z1 = np.dot(x, params['W1']) + params['b1']
    a1 = sigmoid(z1) # 은닉층 활성화 값
    
    # 출력층 계산
    z2 = np.dot(a1, params['W2']) + params['b2']
    y_pred = sigmoid(z2) # 최종 출력
    
    return y_pred, a1

def backward(params, x, y, y_pred, hidden):
    """
    역전파 (Backpropagation) - 기울기(Gradient) 계산
    """
    # 데이터 개수
    m = x.shape[0]
    a1 = hidden
    
    # 출력층 오차 (MSE 가정: Loss = 1/2 * (y - y_pred)^2 의 미분 -> -(y - y_pred))
    # 여기에 시그모이드 미분까지 포함하면: (y_pred - y) * y_pred * (1 - y_pred)
    # 단순화를 위해 (y_pred - y) * sigmoid_derivative(y_pred) 로 계산
    output_error = (y_pred - y) * sigmoid_derivative(y_pred)
    
    dW2 = np.dot(a1.T, output_error)
    db2 = np.sum(output_error, axis=0, keepdims=True)
    
    # 은닉층 오차
    hidden_error = np.dot(output_error, params['W2'].T) * sigmoid_derivative(a1)
    
    dW1 = np.dot(x.T, hidden_error)
    db1 = np.sum(hidden_error, axis=0, keepdims=True)
    
    grads = {
        'dW1': dW1, 'db1': db1,
        'dW2': dW2, 'db2': db2
    }
    return grads

def update_params(params, grads, learning_rate):
    """
    경사 하강법을 적용하여 파라미터 업데이트 (새로운 파라미터 반환)
    """
    new_params = params.copy()
    new_params['W1'] -= learning_rate * grads['dW1']
    new_params['b1'] -= learning_rate * grads['db1']
    new_params['W2'] -= learning_rate * grads['dW2']
    new_params['b2'] -= learning_rate * grads['db2']
    
    return new_params