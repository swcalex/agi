import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import model              
import visualization_utils 

def main():
    print("=== Project AI: 4-Bit Binary to Decimal (v0.5.0) ===")
    
    # 1. 데이터 설정 (0~15 모든 케이스)
    # X: [x3, x2, x1, x0]
    X_data = []
    y_data = []
    
    for i in range(16):
        # 정수를 4비트 리스트로 변환 (예: 3 -> [0, 0, 1, 1])
        # format(i, '04b') returns string '0011'
        binary_str = format(i, '04b')
        bin_list = [int(b) for b in binary_str]
        
        X_data.append(bin_list)
        # y는 0~1 사이로 정규화 (i / 15.0)
        y_data.append([i / 15.0])
        
    X = torch.tensor(X_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)
    
    net = None
    criterion = None
    optimizer = None
    
    while True:
        print("\n[Menu]")
        print("1. Initialization")
        print("2. Forward Test")
        print("3. Training (Backpropagation)")
        print("4. Visualization (2x2 Grid)")
        print("5. Exit")
        
        choice = input("Select: ")
        
        if choice == '1':
            net = model.SimpleMLP()
            criterion = nn.BCELoss()
            optimizer = optim.Adam(net.parameters(), lr=0.01)
            print(">> Model Initialized (4-In, 200-Hidden, 1-Out).")
            
        elif choice == '2':
            if net is None:
                print("!! Initialize first.")
                continue
            with torch.no_grad():
                outputs = net(X)
            
            print("\n>> Prediction Result:")
            print("Input (Bin) -> Target (Dec) -> Pred (Raw) -> Pred (Dec)")
            for i in range(len(X)):
                bin_input = X[i].tolist() # [0,0,1,1]
                target_dec = int(y[i].item() * 15)
                pred_raw = outputs[i].item()
                pred_dec = int(round(pred_raw * 15))
                
                # Check accuracy
                mark = "O" if target_dec == pred_dec else "X"
                print(f"{bin_input} -> {target_dec:2d} -> {pred_raw:.4f} -> {pred_dec:2d} [{mark}]")
                
        elif choice == '3':
            if net is None:
                print("!! Initialize first.")
                continue
            epochs_input = input("Epochs (default 1000): ")
            epochs = int(epochs_input) if epochs_input else 5000
            
            net.train()
            print(f">> Training started for {epochs} epochs...")
            for i in range(epochs):
                optimizer.zero_grad()
                out = net(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
                if i % 1000 == 0:
                    print(f"Epoch {i}: Loss {loss.item():.6f}")
            print(">> Training Done.")
            
        elif choice == '4':
            if net is None:
                print("!! Initialize first.")
                continue
            visualization_utils.plot_decision_boundary(net, X, y)
            
        elif choice == '5':
            sys.exit()

if __name__ == "__main__":
    main()