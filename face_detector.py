import cv2
import mediapipe as mp
import numpy as np
import torch
from typing import Optional, Tuple

class FaceDetector:
    """
    Stage 1: MediaPipe 기반 얼굴 검출 및 PyTorch 입력 규격 전처리 클래스.
    """
    def __init__(self, min_detection_confidence: float = 0.5):
        # MediaPipe Face Detection 초기화
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detect_and_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        웹캠 프레임(BGR)에서 얼굴을 검출하고 해당 영역을 크롭하여 반환합니다.
        얼굴이 미검출되거나 바운딩 박스 오차가 발생할 경우 None을 반환합니다.

        Args:
            frame (np.ndarray): OpenCV 웹캠 입력 이미지 (BGR, H x W x C)

        Returns:
            Optional[np.ndarray]: 크롭된 얼굴 이미지 (BGR, H_face x W_face x C) 또는 None
        """
        h, w, c = frame.shape
        # 1. BGR 이미지를 RGB로 변환 (MediaPipe는 RGB 입력을 받음)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. 얼굴 검출 수행
        results = self.face_detection.process(rgb_frame)
        
        # 3. 검출 결과가 없으면 즉시 None 반환
        if not results.detections:
            return None

        # 4. 가장 신뢰도가 높은 첫 번째 얼굴의 바운딩 박스 추출
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # 5. 상대 좌표를 절대 픽셀 좌표로 변환
        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        bbox_w = int(bbox.width * w)
        bbox_h = int(bbox.height * h)

        # 6. 이미지 경계를 벗어나는 예외 상황 방지 (클리핑 처리)
        x1 = max(0, x_min)
        y1 = max(0, y_min)
        x2 = min(w, x_min + bbox_w)
        y2 = min(h, y_min + bbox_h)

        # 유효한 크롭 크기인지 검증 (폭이나 높이가 0 이하이면 무효)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return None

        # 7. 원형 프레임에서 Bounding Box 영역 절단
        cropped_face = frame[y1:y2, x1:x2]
        return cropped_face

    def preprocess(self, cropped_face: np.ndarray) -> torch.Tensor:
        """
        크롭된 얼굴 이미지를 $112 x 112$ 크기로 리사이즈하고, 정규화 및 텐서 변환을 수행합니다.

        Args:
            cropped_face (np.ndarray): 크롭된 얼굴 이미지 (BGR, H_face x W_face x C)

        Returns:
            torch.Tensor: PyTorch 합성곱 신경망 입력 규격 텐서 (1 x C x H_target x W_target, float32)
        """
        # 1. 쌍선형 보간법(Bilinear Interpolation)을 적용하여 $112 x 112$ 크기로 리사이즈
        resized_face = cv2.resize(cropped_face, (112, 112), interpolation=cv2.INTER_LINEAR)
        
        # 2. BGR에서 RGB로 다시 변환 (최종 모델 입력은 RGB 채널 순서)
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        
        # 3. [0, 255] 정수형 값을 [0.0, 1.0] 실수형(float32)으로 정규화
        normalized_face = rgb_face.astype(np.float32) / 255.0
        
        # 4. 차원 변경: H x W x C -> C x H x W
        tensor_face = np.transpose(normalized_face, (2, 0, 1))
        
        # 5. PyTorch 텐서로 변환하고, 배치 차원을 추가하여 4D 텐서화: 1 x C x H x W
        pytorch_tensor = torch.from_numpy(tensor_face).unsqueeze(0)
        
        return pytorch_tensor

    def close(self):
        """
        MediaPipe 리소스를 명시적으로 해제합니다.
        """
        self.face_detection.close()
