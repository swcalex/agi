import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

class ModelTrainer:
    """
    Stage 2 SimpleCNN 모델의 학습, 손실/MAE 계산 및 역전파 관리 클래스.
    """
    def __init__(self, model: nn.Module, lr: float = 0.01, device: torch.device = torch.device('cpu')):
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # 1. 학습 손실 함수: Binary Cross Entropy (BCE)
        self.criterion_bce = nn.BCELoss()
        
        # 2. 모니터링 보조 지표: Mean Absolute Error (L1 Loss)
        self.criterion_mae = nn.L1Loss()
        
        # 3. 옵티마이저 설정 (기본: Adam 또는 SGD 활용 가능)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def compute_loss_and_metrics(self, y_pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        BCE Loss 및 MAE 모니터링 지표를 계산합니다.

        Args:
            y_pred (torch.Tensor): 모델의 예측 확률값 (B, 1)
            target (torch.Tensor): 정답 타겟 레이블 (B, 1), t in {0.0, 1.0}

        Returns:
            Tuple[torch.Tensor, float]: (bce_loss_tensor, mae_value_float)
        """
        loss = self.criterion_bce(y_pred, target)
        with torch.no_grad():
            mae = self.criterion_mae(y_pred, target).item()
        return loss, mae

    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        1회 학습 스텝 (Forward -> Loss -> Backward -> Optimizer Step)을 수행합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서 (B, 3, 112, 112)
            target (torch.Tensor): 타겟 레이블 텐서 (B, 1)

        Returns:
            Dict[str, float]: 해당 스텝의 {'loss': BCE손실값, 'mae': MAE오차값}
        """
        self.model.train()
        x = x.to(self.device)
        target = target.to(self.device)

        # 1. 그레디언트 초기화
        self.optimizer.zero_grad()

        # 2. 순전파
        y_pred = self.model(x)

        # 3. 손실 및 지표 계산
        loss, mae = self.compute_loss_and_metrics(y_pred, target)

        # 4. 역전파 및 가중치 업데이트
        loss.backward()
        self.optimizer.step()

        return {
            'loss': float(loss.item()),
            'mae': float(mae)
        }