import torch
import torch.nn as nn
import torch.nn.init as init

class SimpleCNN(nn.Module):
    """
    Stage 2: SimpleCNN 이진 분류 신경망 모델 클래스.
    - Input: (B, 3, 112, 112)
    - Output: (B, 1), 특정 인물일 확률 y in [0.0, 1.0]
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 1. Feature Extractor (합성곱 및 풀링 계층)
        # Conv Layer 1: (3, 112, 112) -> (16, 112, 112)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # Max Pooling 1: (16, 112, 112) -> (16, 56, 56)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2: (16, 56, 56) -> (32, 56, 56)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # Max Pooling 2: (32, 56, 56) -> (32, 28, 28)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Classifier (완전연결 계층)
        # Flatten Dimension: 32 * 28 * 28 = 25,088
        self.flatten = nn.Flatten()
        # FC Layer 1: 25,088 -> 200
        self.fc1 = nn.Linear(in_features=32 * 28 * 28, out_features=200)
        self.sigmoid1 = nn.Sigmoid()
        
        # Output Layer: 200 -> 1
        self.fc2 = nn.Linear(in_features=200, out_features=1)
        self.sigmoid2 = nn.Sigmoid()
        
        # 3. 가중치 및 편향 초기화 실행
        self._init_weights()

    def _init_weights(self):
        """
        초기화 전략 적용 (Spec 4.0):
        - Conv Layers (ReLU): He (Kaiming) Normal Initialization
        - FC Layers (Sigmoid): Xavier (Glorot) Normal Initialization
        - Bias: Zero Initialization
        """
        # Conv1 초기화
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            init.zeros_(self.conv1.bias)
            
        # Conv2 초기화
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.conv2.bias is not None:
            init.zeros_(self.conv2.bias)
            
        # FC1 초기화
        init.xavier_normal_(self.fc1.weight)
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
            
        # FC2 (Output) 초기화
        init.xavier_normal_(self.fc2.weight)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (Forward Pass) 연산 수행.

        Args:
            x (torch.Tensor): 전처리된 이미지 텐서 (B, 3, 112, 112)

        Returns:
            torch.Tensor: 인물 확률값 (B, 1), 범위 [0.0, 1.0]
        """
        # Feature Extraction
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten & Classification
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid1(x)
        
        x = self.fc2(x)
        x = self.sigmoid2(x)
        
        return x