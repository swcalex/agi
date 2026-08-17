import numpy as np
import torch
from typing import Tuple, Optional
from face_detector import FaceDetector
from model import SimpleCNN

class FaceRecognitionPipeline:
    """
    Stage 1(FaceDetector)과 Stage 2(SimpleCNN)를 결합한 통합 추론 파이프라인.
    """
    def __init__(self, min_detection_confidence: float = 0.5, device: Optional[str] = None):
        # 1. 실행 디바이스 설정 (기본값: cpu)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # 2. Stage 1 및 Stage 2 모듈 초기화
        self.detector = FaceDetector(min_detection_confidence=min_detection_confidence)
        self.model = SimpleCNN().to(self.device)
        self.model.eval()  # 추론 모드로 설정

    def predict(self, frame: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        웹캠 프레임을 받아 얼굴 검출 여부를 판단하고 최종 특정 인물 확률을 반환합니다.

        Args:
            frame (np.ndarray): OpenCV 프레임 (BGR, H x W x C)

        Returns:
            Tuple[float, Optional[np.ndarray]]: 
                - prob (float): 특정 인물일 확률 (0.0 ~ 1.0)
                - cropped_face (Optional[np.ndarray]): 크롭된 얼굴 이미지 (시각화/디버깅용, 미검출 시 None)
        """
        # Step 1. Stage 1 얼굴 검출 및 크롭
        cropped_face = self.detector.detect_and_crop(frame)
        
        # 엣지 케이스: 얼굴이 검출되지 않았을 경우 즉시 0.0 반환
        if cropped_face is None:
            return 0.0, None
        
        # Step 2. 전처리 (Resize 112x112, Normalize, Tensor 변환)
        input_tensor = self.detector.preprocess(cropped_face).to(self.device)
        
        # Step 3. Stage 2 CNN 순전파 추론
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = float(output.item())
            
        return prob, cropped_face

    def close(self):
        """
        내부 리소스를 해제합니다.
        """
        self.detector.close()