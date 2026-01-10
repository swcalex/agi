import numpy as np
from model import MLP

def main():
    # 학습 데이터 셋 고정
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    model = None

    while True:
        print("\n=== MLP Project Main Menu ===")
        print("1. Initialization")
        print("2. Forward Propagation (Inference)")
        print("3. Backpropagation & Training")
        print("4. Exit")
        
        choice = input("원하는 메뉴를 선택하세요: ")

        if choice == '1':
            # 초기화 모듈 실행
            model = MLP(n_in=2, n_h=4, n_out=1, lr=0.1)
            print("\n[알림] 모델 가중치 및 편향이 초기화되었습니다.")

        elif choice == '2':
            # 전방 연산 서브 메뉴
            if model is None:
                print("\n[오류] 먼저 모델을 초기화(1번)해 주세요.")
                continue
            
            while True:
                print("\n--- Forward Propagation Sub-Menu ---")
                print("1. Input Data")
                print("2. Exit to Main Menu")
                sub_choice = input("선택하세요: ")
                
                if sub_choice == '1':
                    try:
                        user_input = input("두 개의 입력을 입력하세요 (예: 0 1): ")
                        x_user = np.array([list(map(int, user_input.split()))])
                        raw_val = model.forward(x_user)
                        final_val = 1 if raw_val >= 0.5 else 0
                        print(f"결과: {final_val} (실수값: {raw_val[0][0]:.4f})")
                    except Exception as e:
                        print(f"입력 형식이 잘못되었습니다. (에러: {e})")
                elif sub_choice == '2':
                    break
                else:
                    print("올바른 메뉴를 선택해 주세요.")

        elif choice == '3':
            # 학습 모듈 실행
            if model is None:
                print("\n[오류] 먼저 모델을 초기화(1번)해 주세요.")
                continue
            
            try:
                epochs = int(input("학습할 에폭(Epoch) 횟수를 입력하세요: "))
                print(f"\n학습 진행 중 (Total: {epochs})...")
                
                for epoch in range(epochs):
                    # 데이터 셔플링
                    indices = np.arange(X.shape[0])
                    np.random.shuffle(indices)
                    for i in indices:
                        x_i, y_i = X[i:i+1], Y[i:i+1]
                        output = model.forward(x_i)
                        model.backward(x_i, y_i, output)
                    
                    if (epoch + 1) % 1000 == 0:
                        loss = model.get_loss(Y, model.forward(X))
                        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss:.6f}")
                print("학습이 완료되었습니다.")
            except ValueError:
                print("숫자를 입력해 주세요.")

        elif choice == '4':
            print("프로그램을 종료합니다.")
            break
        else:
            print("올바른 메뉴를 선택해 주세요.")

if __name__ == "__main__":
    main()
