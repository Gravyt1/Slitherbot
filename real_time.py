import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO

# Path to your trained YOLO model
MODEL_PATH = "runs/detect/train/weights/best.pt"

# Adjust the monitor region based on your screen resolution
MONITOR_REGION = {
    "top": 120,          
    "left": 0,           
    "width": 1920,       
    "height": int(1080 * 0.84)
}

CONF_THRESHOLD = 0.25  # Confidence threshold for detections
DELAY = 0.01           # Delay between frames (in seconds)

def realtime_detection():
    """
    Performs real-time object detection on screen capture.
    Press 'Esc' to quit the detection loop.
    """
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    with mss.mss() as sct:
        print("Starting real-time detection... Press 'Esc' to quit.")
        while True:
            start_time = time.time()

            # Capture screen
            sct_img = sct.grab(MONITOR_REGION)
            frame = np.array(sct_img)

            # Convert color format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Run YOLO prediction
            results = model.predict(source=frame, conf=CONF_THRESHOLD, save=False, verbose=False)

            # Draw bounding boxes and labels
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{cls}:{conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate and display FPS
            fps = 1 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("YOLO Real-Time Detection", frame)

            # Exit on 'Esc' key
            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(DELAY)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_detection()