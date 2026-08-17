import os
import time
import cv2
import torch
import numpy as np
from pipeline import FaceRecognitionPipeline

def run_main():
    print("=== Project AGI: 실시간 인물 인식 시스템 (v0.6.0) ===")
    
    # 1. 실행 디바이스 및 파이프라인 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">> 연산 디바이스: {device}")
    
    pipeline = FaceRecognitionPipeline(min_detection_confidence=0.5, device=str(device))
    
    # 2. 사전 학습된 가중치(weights.pth) 로드
    weights_path = "weights.pth"
    if os.path.exists(weights_path):
        pipeline.model.load_state_dict(torch.load(weights_path, map_location=device))
        pipeline.model.eval()
        print(f">> 학습된 가중치를 성공적으로 로드했습니다: '{weights_path}'")
    else:
        print(f"!! 경고: '{weights_path}'를 찾을 수 없습니다. 초기화 상태의 모델로 실행합니다.")

    # 3. 웹캠 캡처 초기화 (240p: 320 x 240)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    if not cap.isOpened():
        print("!! 웹캠 장치를 열 수 없습니다.")
        pipeline.close()
        return

    print(">> 웹캠 구동 시작. (종료: 'q' | 가중치 재로드: 'r')\n")

    # 추론 주기 제어 변수 (2fps -> 0.5초 주기)
    prev_time = 0.0
    interval = 0.5
    current_prob = 0.0
    last_cropped_face = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽어올 수 없습니다.")
                break

            current_time = time.time()

            # 0.5초마다 파이프라인 추론 수행
            if (current_time - prev_time) >= interval:
                prev_time = current_time
                prob, cropped = pipeline.predict(frame)
                current_prob = prob
                last_cropped_face = cropped

            # ==========================================
            # UI 시각화 렌더링
            # ==========================================
            display_frame = frame.copy()
            h_frame, w_frame, _ = display_frame.shape

            # 1. 상태 및 확률 텍스트 표시
            if last_cropped_face is None:
                status_text = "No Face Detected"
                color = (128, 128, 128)  # 회색
            elif current_prob >= 0.5:
                status_text = f"TARGET MATCH: {current_prob * 100:.1f}%"
                color = (0, 255, 0)      # 초록색
            else:
                status_text = f"NON-TARGET: {current_prob * 100:.1f}%"
                color = (0, 0, 255)      # 빨간색

            # 텍스트 오버레이
            cv2.rectangle(display_frame, (5, 5), (w_frame - 5, 40), (0, 0, 0), -1)
            cv2.putText(display_frame, status_text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # 2. 크롭된 얼굴 PiP 표시 (우측 하단)
            pip_size = 80
            if last_cropped_face is not None:
                pip_face = cv2.resize(last_cropped_face, (pip_size, pip_size))
            else:
                pip_face = np.zeros((pip_size, pip_size, 3), dtype=np.uint8)

            # 테두리 및 PiP 삽입
            display_frame[h_frame - pip_size - 10:h_frame - 10,
                          w_frame - pip_size - 10:w_frame - 10] = pip_face
            cv2.rectangle(display_frame,
                          (w_frame - pip_size - 10, h_frame - pip_size - 10),
                          (w_frame - 10, h_frame - 10), color, 2)

            cv2.imshow("Project AGI - Real-Time Recognition (240p)", display_frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if os.path.exists(weights_path):
                    pipeline.model.load_state_dict(torch.load(weights_path, map_location=device))
                    pipeline.model.eval()
                    print(f">> 가중치 파일 다시 로드 완료: '{weights_path}'")

    finally:
        cap.release()
        pipeline.close()
        cv2.destroyAllWindows()
        print("=== 시스템이 안전하게 종료되었습니다 ===")

if __name__ == "__main__":
    run_main()