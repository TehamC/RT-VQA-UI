import cv2
import base64
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv12 model (trained with Ultralytics)
YOLO_MODEL_PATH = "/home/teham/case_study/VQA/yolov12/896_anchor/runs/detect/train/weights/best.pt"
model = YOLO(YOLO_MODEL_PATH)

@app.get("/stream")
async def stream(request: Request, video_name: str):
    video_path = f"/home/teham/case_study/VQA/{video_name}"
    if not os.path.isfile(video_path):
        return {"detail": f"Video {video_name} not found"}

    cap = cv2.VideoCapture(video_path)

    async def event_generator():
        while cap.isOpened():
            if await request.is_disconnected():
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv12 detection
            results = model(frame)[0]
            annotated_frame = results.plot()

            # Parse detections
            detections = []
            for box in results.boxes:
                detections.append({
                    "bbox": [int(x.item()) for x in box.xyxy[0]],
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                })

            # Convert annotated frame to base64
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "image_base64": jpg_as_text,
                "detections": detections,
            }

            try:
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception as e:
                print("Streaming error:", e)
                break

        cap.release()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
