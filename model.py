import torch
import torch.nn as nn
import torch.nn.init as init

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 하이퍼파라미터 설정 (Spec v0.5.1)
        self.input_size = 4   # 4-bit input
        self.hidden_size = 200  # Wide Network
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
        """
        init.xavier_normal_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_normal_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        순전파 (Forward Propagation)
        """
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        return out

    def get_hidden_features(self, x):
        """
        [v0.5.1] 은닉층의 활성화 패턴(Activation Map)을 관측하기 위한 메서드
        Input -> Hidden (Activated) 값을 반환
        """
        out = self.fc1(x)
        out = self.activation(out)
        return out