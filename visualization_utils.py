import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

# 리눅스 환경(GUI 없음) 대응
matplotlib.use('Agg') 

def plot_decision_boundary(model, X_tensor, y_tensor):
    """
    PyTorch 모델의 결정 경계 시각화 함수
    
    Args:
        model: 학습된 PyTorch 모델 (nn.Module)
        X_tensor: 입력 데이터 (torch.Tensor)
        y_tensor: 정답 레이블 (torch.Tensor)
    """
    print(">> Generating graph...")
    
    # 텐서를 넘파이로 변환 (시각화 범위 설정용)
    X_numpy = X_tensor.detach().numpy()
    y_numpy = y_tensor.detach().numpy()

    # 1. 그래프 범위 설정
    x_min, x_max = X_numpy[:, 0].min() - 0.5, X_numpy[:, 0].max() + 0.5
    y_min, y_max = X_numpy[:, 1].min() - 0.5, X_numpy[:, 1].max() + 0.5
    h = 0.01 # 격자 간격
    
    # 2. 격자(Meshgrid) 생성
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 3. 예측 수행 (PyTorch 로직)
    # 격자 데이터를 텐서로 변환
    grid_data = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid_data).float()
    
    model.eval() # 평가 모드 전환
    with torch.no_grad():
        # 추론 후 넘파이로 다시 변환
        Z = model(grid_tensor).detach().numpy()
        
    Z = Z.reshape(xx.shape)
    
    # 4. 시각화 그리기
    plt.figure(figsize=(8, 6))
    
    # 배경: RdBu (Red-Blue) 컬러맵
    plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
    plt.colorbar(label='Output Probability (Red=0, Blue=1)')
    
    # 결정 경계선 (0.5) 강조 - 검은색 굵은 실선
    contour_line = plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)
    plt.clabel(contour_line, inline=True, fontsize=12, fmt='%1.2f')
    
    # 데이터 포인트 (테두리 흰색으로 강조)
    plt.scatter(X_numpy[:, 0], X_numpy[:, 1], c=y_numpy.flatten(), 
                s=150, cmap='RdBu', edgecolors='white', linewidths=2)
    
    plt.title("PyTorch MLP Decision Boundary (Hidden=200, Xavier Init)")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    
    # 파일 저장
    filename = 'decision_boundary.png'
    plt.savefig(filename)
    plt.close()
    
    print(f">> Graph saved successfully as '{filename}'.")