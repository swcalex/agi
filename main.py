import sys
import numpy as np
import model  # Module A
import visualization_utils  # Module C

def main():
    # 데이터 설정 (XOR 문제)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    params = None
    learning_rate = 0.1
    
    print("=== Simple MLP Project (v0.3.0) ===")
    
    while True:
        print("\n[Menu]")
        print("1. Initialization")
        print("2. Forward Propagation (Test)")
        print("3. Backpropagation & Training")
        print("4. Visualization (Decision Boundary)")
        print("5. Exit")
        
        choice = input("Select Number: ")
        
        if choice == '1':
            input_size = 2
            hidden_size = int(input("Enter Hidden Layer Size (default: 4): ") or 4)
            output_size = 1
            params = model.init_params(input_size, hidden_size, output_size)
            print(">> Model Initialized.")
            
        elif choice == '2':
            if params is None:
                print("!! Error: Model not initialized. Please run step 1.")
                continue
            
            # 전체 데이터에 대해 테스트
            y_pred, _ = model.forward(params, X)
            print("\n>> Test Results (Input -> Prediction):")
            for i in range(len(X)):
                print(f"   {X[i]} -> {y_pred[i][0]:.4f}")
                
        elif choice == '3':
            if params is None:
                print("!! Error: Model not initialized. Please run step 1.")
                continue
                
            epochs = int(input("Enter Epochs (e.g., 10000): "))
            
            print(f">> Training started for {epochs} epochs...")
            for i in range(epochs):
                # 1. Forward
                y_pred, hidden = model.forward(params, X)
                
                # 2. Backward
                grads = model.backward(params, X, y, y_pred, hidden)
                
                # 3. Update
                params = model.update_params(params, grads, learning_rate)
                
                if i % 1000 == 0:
                    loss = np.mean(np.square(y - y_pred))
                    print(f"   Epoch {i}: Loss {loss:.6f}")
            
            print(">> Training Complete.")
            
        elif choice == '4':
            if params is None:
                print("!! Error: Model not initialized. Please run step 1.")
                continue
            
            print(">> Generating Decision Boundary Plot...")
            
            # 시각화 모듈 호출
            # model.forward는 (y_pred, hidden)을 반환하므로, y_pred만 반환하는 래퍼 함수 전달
            predict_wrapper = lambda p, x: model.forward(p, x)[0]
            
            visualization_utils.plot_decision_boundary(params, predict_wrapper, X, y)
            
        elif choice == '5':
            print("Exiting program.")
            sys.exit()
            
        else:
            print("!! Invalid choice. Try again.")

if __name__ == "__main__":
    main()