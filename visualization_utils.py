import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

# 리눅스 환경(GUI 없음) 대응
matplotlib.use('Agg') 

def plot_decision_boundary(model, X_tensor, y_tensor):
    """
    4비트 입력 공간을 2x2 격자로 시각화 (Slicing Hypercubes)
    """
    print(">> Generating 4D visualization (2x2 Grid)...")
    
    # 설정: 격자 간격 및 범위
    h = 0.02
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    
    # 2x2 Subplot 생성
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # 상위 비트 고정값 정의: (x3, x2) -> (0,0), (0,1), (1,0), (1,1)
    fixed_bits_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    subplot_titles = ["MSB: 00 (0~3)", "MSB: 01 (4~7)", "MSB: 10 (8~11)", "MSB: 11 (12~15)"]
    
    # 격자(Meshgrid) 생성 (하위 2비트 x1, x0 용)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()] # (N, 2)
    
    model.eval()
    
    for idx, (ax, fixed_bits, title) in enumerate(zip(axes.flat, fixed_bits_list, subplot_titles)):
        x3_val, x2_val = fixed_bits
        
        # 1. 4차원 입력 데이터 생성
        # [x3(고정), x2(고정), x1(변수), x0(변수)]
        N_grid = grid_points.shape[0]
        x3_arr = np.full((N_grid, 1), x3_val)
        x2_arr = np.full((N_grid, 1), x2_val)
        
        # 전체 입력 결합
        input_numpy = np.hstack((x3_arr, x2_arr, grid_points))
        input_tensor = torch.from_numpy(input_numpy).float()
        
        # 2. 추론
        with torch.no_grad():
            Z = model(input_tensor).detach().numpy()
        Z = Z.reshape(xx.shape)
        
        # 3. 그리기
        # 등고선 (Probabilities)
        contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6, vmin=0, vmax=1)
        
        # 데이터 포인트 표시 (현재 상위 비트에 해당하는 데이터만 필터링)
        X_numpy = X_tensor.detach().numpy()
        y_numpy = y_tensor.detach().numpy()
        
        # 마스킹: x3, x2가 현재 고정값과 일치하는 데이터만 찾기
        mask = (np.abs(X_numpy[:, 0] - x3_val) < 0.1) & (np.abs(X_numpy[:, 1] - x2_val) < 0.1)
        X_subset = X_numpy[mask]
        y_subset = y_numpy[mask]
        
        if len(X_subset) > 0:
            # 정답 라벨 표시 (0~15 정수값으로 복원하여 표시)
            true_vals = (y_subset * 15).flatten().round().astype(int)
            for i in range(len(X_subset)):
                ax.text(X_subset[i, 2], X_subset[i, 3], str(true_vals[i]), 
                        fontsize=12, ha='center', va='center', 
                        color='white', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
        
        ax.set_title(title)
        ax.set_xlabel("Bit 1 ($x_1$)")
        ax.set_ylabel("Bit 0 ($x_0$)")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    # 컬러바 추가 (전체 공유)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(axes[0,0].collections[0], cax=cbar_ax, label='Output Probability (0~1)')
    
    filename = 'decision_boundary_4bit.png'
    plt.savefig(filename)
    plt.close()
    print(f">> Graph saved successfully as '{filename}'.")