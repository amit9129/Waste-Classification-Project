# pip install opencv-contrib-python
# pip install ultralytics
import cv2
from ultralytics import YOLO

def load_model(model_path='waste_detection\yolo\weights\best.pt'):
    """Load the YOLOv8 model."""
    return YOLO(model_path)

def process_frame(frame, model):
    """Run YOLOv8 inference on the frame and visualize the results."""
    results = model(frame)
    annotated_frame = results.render()
    return annotated_frame

def display_progress(frame, current_frame, total_frames):
    """Display progress bar and percentage on the frame."""
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 5), (int(width * current_frame / total_frames), height), (0, 200, 255), -1)
    percentage = round((current_frame / total_frames) * 100, 2)
    cv2.putText(frame, f"{percentage}%", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return frame

def process_video(video_path=0, model_path='yolov8n.pt'):
    """Process the video with YOLOv8 model."""
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    while cap.isOpened():
        success, frame = cap.read()
        current_frame += 1
        if success:
            annotated_frame = process_frame(frame, model)
            annotated_frame = display_progress(annotated_frame, current_frame, total_frames)
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"E:\q.mp4"  # Set your video path here
    process_video(video_path)

