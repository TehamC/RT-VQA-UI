import asyncio
import base64
import cv2
import json
import os
from typing import List, Dict
from fastapi import FastAPI, Request, WebSocket, APIRouter
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from file_selection import router as file_router
from file_selection import download_progress
from yolo_inference import perform_yolo_inference
from llm_inference import generate_llm_response, parse_llm_target_pile, update_context
from config import VIDEO_DIR

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(file_router)

router = APIRouter()

current_llm_question = "Fill the shovel"

class QuestionRequest(BaseModel):
    question: str

@router.websocket("/download_progress/{video_id}")
async def download_progress_websocket(websocket: WebSocket, video_id: str):
    await websocket.accept()
    try:
        while True:
            progress_data = download_progress.get(video_id, {
                'progress': 0,
                'speed': 0,
                'eta': 0,
                'status': 'pending'
            })
            await websocket.send_json(progress_data)
            if progress_data['status'] in ['finished', 'error']:
                break
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@router.post("/set_llm_question")
async def update_task(req: QuestionRequest):
    global current_llm_question
    current_llm_question = req.question
    return {"status": "ok", "task": current_llm_question}

@router.get("/stream/resume")
async def stream_resume(request: Request, video_name: str, frame_index: int, frame_skip: int = 1):
    video_path = os.path.join(VIDEO_DIR, video_name)
    if not os.path.isfile(video_path):
        return {"detail": f"Video {video_name} not found"}

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    async def event_generator():
        current_frame = frame_index
        while cap.isOpened() and current_frame < total_frames:
            if await request.is_disconnected():
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            detections, anchor_bbox, piles_info = perform_yolo_inference(frame)

            if anchor_bbox and piles_info:
                context = update_context(anchor_bbox, piles_info)
                full_llm_prompt = f"<s>[INST] {context}\nTask: {current_llm_question} [/INST]"
                answer = generate_llm_response(full_llm_prompt)
                selected_pile = parse_llm_target_pile(answer)
                selected_pile_bbox = piles_info.get(selected_pile)
                annotated_frame = draw_boxes(frame.copy(), detections, selected_pile_bbox, anchor_bbox) if selected_pile_bbox else frame
            else:
                annotated_frame = frame

            _, buffer = cv2.imencode(".jpg", annotated_frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "image_base64": jpg_as_text,
                "detections": detections,
                "frame_index": current_frame
            }

            try:
                yield f"data: {json.dumps(payload)}\n\n"
                current_frame += frame_skip
            except Exception as e:
                print("Streaming error:", e)
                break

        cap.release()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/frame/infer")
async def infer_single_frame(video_name: str, frame_index: int):
    video_path = os.path.join(VIDEO_DIR, video_name)
    if not os.path.isfile(video_path):
        return {"detail": f"Video {video_name} not found"}

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"detail": f"Could not read frame {frame_index}"}

    detections, anchor_bbox, piles_info = perform_yolo_inference(frame)

    if anchor_bbox and piles_info:
        context = update_context(anchor_bbox, piles_info)
        full_llm_prompt = f"<s>[INST] {context}\nTask: {current_llm_question} [/INST]"
        answer = generate_llm_response(full_llm_prompt)
        selected_pile = parse_llm_target_pile(answer)
        selected_pile_bbox = piles_info.get(selected_pile)
        annotated_frame = draw_boxes(frame.copy(), detections, selected_pile_bbox, anchor_bbox) if selected_pile_bbox else frame
    else:
        annotated_frame = frame
        answer = "Could not find anchor or piles."

    _, buffer = cv2.imencode(".jpg", annotated_frame)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")

    return {
        "image_base64": jpg_as_text,
        "detections": detections,
        "frame_index": frame_index,
        "llm_answer": answer,
        "task": current_llm_question
    }

def draw_boxes(frame, detections: List[Dict], selected_pile_bbox, anchor_bbox):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det["class_name"]
        conf = det["confidence"]

        if det["bbox"] == selected_pile_bbox:
            color = (0, 0, 255)
        elif det["bbox"] == anchor_bbox:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

app.include_router(router)