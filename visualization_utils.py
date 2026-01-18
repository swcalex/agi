import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 리눅스 환경(GUI 없음) 대응
matplotlib.use('Agg') 

def plot_decision_boundary(params, predict_func, X, y):
    """
    결정 경계 시각화 함수 (디자인 개선 및 파일 저장 버전)
    """
    print(">> Generating graph...")

    # 1. 그래프 범위 설정
    # 데이터 포인트 주변에 여백을 둠
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01 # 격자 간격 (작을수록 곡선이 부드러움)
    
    # 2. 격자(Meshgrid) 생성
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 3. 예측 수행
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_func(params, grid_input)
    Z = Z.reshape(xx.shape)
    
    # 4. 시각화 그리기
    plt.figure(figsize=(8, 6))
    
    # [배경 색상]
    # RdBu: 빨강(0) <-> 파랑(1) 대비 (중간은 흰색에 가까움)
    # alpha=0.6: 색을 연하게 해서 눈을 편안하게 함
    plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
    plt.colorbar(label='Output Probability (Red=0, Blue=1)')
    
    # [임계값 등고선 강조] - 요청하신 핵심 기능
    # levels=[0.5]: 정확히 0.5인 지점만 그림
    # colors='black', linewidths=3: 검고 굵은 선으로 강조
    contour_line = plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)
    
    # 등고선에 라벨 추가 (선 위에 '0.50' 이라고 표시됨)
    plt.clabel(contour_line, inline=True, fontsize=12, fmt='%1.2f')
    
    # [데이터 포인트]
    # 산점도 테두리를 진하게(edgecolors='k') 하여 배경과 구분
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=150, cmap='RdBu', edgecolors='white', linewidths=2)
    
    plt.title("MLP Decision Boundary (Threshold = 0.5)")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    
    # 파일 저장
    filename = 'decision_boundary.png'
    plt.savefig(filename)
    plt.close()
    
    print(f">> Graph saved successfully as '{filename}'.")