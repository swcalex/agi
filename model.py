import torch
import torch.nn as nn
import torch.nn.init as init

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 하이퍼파라미터 설정
        self.input_size = 2
        self.hidden_size = 200  # Wide Network 전략
        self.output_size = 1
        
        # 레이어 정의
        # fc1: Input -> Hidden
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # fc2: Hidden -> Output
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        
        # 활성화 함수
        self.activation = nn.Sigmoid()
        
        # 가중치 초기화 (Xavier Initialization)
        self._init_weights()

    def _init_weights(self):
        """
        Xavier (Glorot) Initialization 적용
        노드 수가 많아짐(200개)에 따른 출력 분산 폭발 방지
        """
        init.xavier_normal_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_normal_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        순전파 (Forward Propagation)
        """
        # 은닉층: 선형 변환 -> 활성화 함수
        out = self.fc1(x)
        out = self.activation(out)
        
        # 출력층: 선형 변환 -> 활성화 함수 (확률값 0~1)
        out = self.fc2(out)
        out = self.activation(out)
        
        return out