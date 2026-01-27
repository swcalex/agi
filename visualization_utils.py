import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

# 리눅스 환경(GUI 없음) 대응
matplotlib.use('Agg') 

def plot_decision_boundary(model, X_tensor, y_tensor):
    """
    [v0.5.0] 4비트 입력 공간을 2x2 격자로 시각화 (Slicing Hypercubes)
    """
    print(">> Generating 4D visualization (2x2 Grid)...")
    
    # 설정: 격자 간격 및 범위
    h = 0.02
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    
    # 2x2 Subplot 생성
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fixed_bits_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    subplot_titles = ["MSB: 00 (0~3)", "MSB: 01 (4~7)", "MSB: 10 (8~11)", "MSB: 11 (12~15)"]
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()] 
    
    model.eval()
    
    for idx, (ax, fixed_bits, title) in enumerate(zip(axes.flat, fixed_bits_list, subplot_titles)):
        x3_val, x2_val = fixed_bits
        N_grid = grid_points.shape[0]
        x3_arr = np.full((N_grid, 1), x3_val)
        x2_arr = np.full((N_grid, 1), x2_val)
        
        input_numpy = np.hstack((x3_arr, x2_arr, grid_points))
        input_tensor = torch.from_numpy(input_numpy).float()
        
        with torch.no_grad():
            Z = model(input_tensor).detach().numpy()
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6, vmin=0, vmax=1)
        
        X_numpy = X_tensor.detach().numpy()
        y_numpy = y_tensor.detach().numpy()
        mask = (np.abs(X_numpy[:, 0] - x3_val) < 0.1) & (np.abs(X_numpy[:, 1] - x2_val) < 0.1)
        X_subset = X_numpy[mask]
        y_subset = y_numpy[mask]
        
        if len(X_subset) > 0:
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
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(axes[0,0].collections[0], cax=cbar_ax, label='Output Probability (0~1)')
    
    filename = 'decision_boundary_4bit.png'
    plt.savefig(filename)
    plt.close()
    print(f">> Graph saved successfully as '{filename}'.")

def plot_hidden_activation_map(model, X_tensor):
    """
    [v0.5.1] 은닉층 활성화 히트맵 시각화 (Input vs Hidden Neurons)
    """
    print(">> Generating Hidden Layer Activation Map...")
    
    model.eval()
    with torch.no_grad():
        # (16, 200) 크기의 은닉층 활성화 값 추출
        hidden_features = model.get_hidden_features(X_tensor).numpy()
    
    plt.figure(figsize=(15, 8))
    
    # 히트맵 그리기
    # aspect='auto': 셀을 정사각형으로 강제하지 않고 화면에 꽉 차게 늘림
    plt.imshow(hidden_features, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    plt.colorbar(label='Activation Value (Sigmoid: 0~1)')
    plt.title(f"Hidden Layer Activation Map (Hidden Size: {hidden_features.shape[1]})")
    plt.xlabel("Hidden Neurons Index (0 ~ 199)")
    plt.ylabel("Input Integer Value (0 ~ 15)")
    
    # Y축 눈금을 0~15 정수로 설정
    plt.yticks(range(16), labels=[str(i) for i in range(16)])
    
    plt.tight_layout()
    
    filename = 'hidden_activation_map.png'
    plt.savefig(filename)
    plt.close()
    print(f">> Graph saved successfully as '{filename}'.")