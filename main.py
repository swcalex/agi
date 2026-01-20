import sys
import torch
import torch.nn as nn
import torch.optim as optim
import model              # Module A
import visualization_utils # Module C

def main():
    # 데이터 설정 (XOR 문제) - PyTorch Tensor로 변환 (float32)
    # GPU 사용 가능 시 .to('cuda')를 붙일 수 있음 (현재는 CPU)
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    # 모델 및 학습 도구 변수
    net = None
    criterion = None
    optimizer = None
    
    print("=== Project AI: PyTorch MLP (v0.4.0) ===")
    
    while True:
        print("\n[Menu]")
        print("1. Initialization (Build Model)")
        print("2. Forward Propagation (Test)")
        print("3. Backpropagation & Training")
        print("4. Visualization (Decision Boundary)")
        print("5. Exit")
        
        choice = input("Select Number: ")
        
        if choice == '1':
            # 모델 인스턴스 생성
            net = model.SimpleMLP()
            
            # 손실 함수: 이진 교차 엔트로피 (Binary Cross Entropy)
            criterion = nn.BCELoss()
            
            # 옵티마이저: Adam (학습률 0.01 권장)
            # SGD보다 수렴 속도가 훨씬 빠르고 안정적임
            optimizer = optim.Adam(net.parameters(), lr=0.01)
            
            print(">> Model Built with PyTorch.")
            print(f"   Structure: {net}")
            print("   Optimizer: Adam, Loss: BCELoss")
            
        elif choice == '2':
            if net is None:
                print("!! Error: Model not initialized. Please run step 1.")
                continue
            
            # 추론 시에는 기울기 계산 불필요 (no_grad)
            with torch.no_grad():
                outputs = net(X)
                
            print("\n>> Test Results (Input -> Prediction):")
            for i in range(len(X)):
                input_val = X[i].tolist()
                pred_val = outputs[i].item()
                print(f"   {input_val} -> {pred_val:.4f}")
                
        elif choice == '3':
            if net is None:
                print("!! Error: Model not initialized. Please run step 1.")
                continue
                
            epochs_input = input("Enter Epochs (default: 1000): ")
            epochs = int(epochs_input) if epochs_input else 1000
            
            print(f">> Training started for {epochs} epochs...")
            
            net.train() # 학습 모드 전환
            
            for i in range(epochs):
                # 1. 기울기 초기화 (누적 방지)
                optimizer.zero_grad()
                
                # 2. 순전파
                outputs = net(X)
                
                # 3. 손실 계산
                loss = criterion(outputs, y)
                
                # 4. 역전파 (Autograd가 자동으로 기울기 계산)
                loss.backward()
                
                # 5. 가중치 갱신
                optimizer.step()
                
                if i % 100 == 0:
                    print(f"   Epoch {i}: Loss {loss.item():.6f}")
            
            print(">> Training Complete.")
            
        elif choice == '4':
            if net is None:
                print("!! Error: Model not initialized. Please run step 1.")
                continue
            
            # PyTorch 모델 객체와 텐서 데이터를 그대로 전달
            visualization_utils.plot_decision_boundary(net, X, y)
            
        elif choice == '5':
            print("Exiting program.")
            sys.exit()
            
        else:
            print("!! Invalid choice. Try again.")

if __name__ == "__main__":
    main()