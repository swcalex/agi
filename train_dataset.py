import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from face_detector import FaceDetector
from model import SimpleCNN
from trainer import ModelTrainer

def augment_tensor(tensor_img: torch.Tensor) -> list:
    """
    (1, 3, 112, 112) 규격의 정규화된 텐서에 6배 데이터 증강을 적용합니다.
    1. 원본 (Original)
    2. 수평 반전 (Horizontal Flip)
    3. 밝기 증가 (+25%)
    4. 밝기 감소 (-25%)
    5. 대비 증가
    6. 복합 증강 (반전 + 밝기 +15%)
    """
    aug_list = []
    
    # 1. 원본
    aug_list.append(tensor_img)
    
    # 2. 수평 반전 (W축 반전, dim=3)
    flipped = torch.flip(tensor_img, dims=[3])
    aug_list.append(flipped)
    
    # 3. 밝기 증가 (clip [0.0, 1.0])
    bright = torch.clamp(tensor_img * 1.25, 0.0, 1.0)
    aug_list.append(bright)
    
    # 4. 밝기 감소
    dark = tensor_img * 0.75
    aug_list.append(dark)
    
    # 5. 대비 증가: ((I - 0.5) * 1.3) + 0.5
    contrast = torch.clamp((tensor_img - 0.5) * 1.3 + 0.5, 0.0, 1.0)
    aug_list.append(contrast)
    
    # 6. 복합 증강 (수평 반전 + 밝기 증가)
    flip_bright = torch.clamp(flipped * 1.15, 0.0, 1.0)
    aug_list.append(flip_bright)
    
    return aug_list

def load_and_augment_data(detector: FaceDetector, data_dir: str = "data"):
    """
    data/target 및 data/non_target 폴더의 이미지를 로드하여 얼굴 검출 및 증강 데이터셋을 생성합니다.
    """
    categories = [("target", 1.0), ("non_target", 0.0)]
    all_tensors = []
    all_labels = []
    
    print(">> 데이터셋 로드 및 Stage 1 전처리/증강 시작...")
    
    for folder_name, label_val in categories:
        folder_path = os.path.join(data_dir, folder_name)
        image_paths = glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]"))
        
        valid_face_count = 0
        total_augmented_count = 0
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Stage 1: 얼굴 검출 및 크롭
            cropped = detector.detect_and_crop(img)
            if cropped is None:
                print(f"   [얼굴 미검출 제외] {os.path.basename(img_path)}")
                continue
                
            valid_face_count += 1
            
            # 전처리 (112x112 텐서화)
            base_tensor = detector.preprocess(cropped)  # (1, 3, 112, 112)
            
            # 6배 증강 적용
            augmented_tensors = augment_tensor(base_tensor)
            for aug_t in augmented_tensors:
                all_tensors.append(aug_t)
                all_labels.append(label_val)
                total_augmented_count += 1
                
        print(f">> [{folder_name}] 원본 검출 성공: {valid_face_count}장 -> 증강 후: {total_augmented_count}장 생성")
        
    if len(all_tensors) == 0:
        raise ValueError("유효한 얼굴 데이터가 검출되지 않았습니다. data 폴더를 확인해주세요.")
        
    # 배치 텐서 결합: (N, 3, 112, 112), (N, 1)
    X = torch.cat(all_tensors, dim=0)
    y = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)
    
    return X, y

def train_model():
    print("=== Stage 2 CNN 오프라인 배치 학습 시작 ===")
    
    # 1. 디바이스 및 컴포넌트 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">> 연산 디바이스: {device}")
    
    detector = FaceDetector(min_detection_confidence=0.5)
    model = SimpleCNN().to(device)
    trainer = ModelTrainer(model=model, lr=0.0005, device=device)
    
    try:
        # 2. 데이터셋 로드 및 증강
        X, y = load_and_augment_data(detector, data_dir="data")
        print(f">> 최종 학습 데이터셋 규격: X={X.shape}, y={y.shape}")
        
        # DataLoader 생성 (배치 16, 셔플 적용)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # 3. 학습 전 초기 오차 관측
        model.eval()
        with torch.no_grad():
            init_preds = model(X.to(device))
            init_loss, init_mae = trainer.compute_loss_and_metrics(init_preds, y.to(device))
            print(f"\n>> [학습 전] Initial Loss: {init_loss.item():.4f} | Initial MAE: {init_mae:.4f}")
            
        # 4. 에포크 학습 진행 (100 Epochs)
        epochs = 100
        print(f">> 총 {epochs} Epochs 학습 시작...\n")
        
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            epoch_maes = []
            
            for batch_x, batch_y in loader:
                metrics = trainer.train_step(batch_x, batch_y)
                epoch_losses.append(metrics['loss'])
                epoch_maes.append(metrics['mae'])
                
            avg_loss = np.mean(epoch_losses)
            avg_mae = np.mean(epoch_maes)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch:03d}/{epochs}] -> Avg BCE Loss: {avg_loss:.6f} | Avg MAE: {avg_mae:.6f}")
                
        # 5. 최종 성능 평가 및 가중치 저장
        model.eval()
        with torch.no_grad():
            final_preds = model(X.to(device))
            final_loss, final_mae = trainer.compute_loss_and_metrics(final_preds, y.to(device))
            print(f"\n>> [학습 완료] Final BCE Loss: {final_loss.item():.6f} | Final MAE: {final_mae:.6f}")
            
        # 모델 파라미터 저장
        save_path = "weights.pth"
        torch.save(model.state_dict(), save_path)
        print(f">> 최적 가중치가 성공적으로 저장되었습니다: '{save_path}'")
        
    finally:
        detector.close()
        print("=== 학습 파이프라인 정상 종료 ===")

if __name__ == "__main__":
    train_model()