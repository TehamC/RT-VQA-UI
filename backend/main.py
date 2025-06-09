import cv2
import base64
import json
import os
from typing import List, Tuple, Optional
import time
import re
import asyncio
from fastapi import FastAPI, Request, Body, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ultralytics import YOLO  
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from peft import PeftModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

# --- Constants ---
VIDEO_DIR = "/home/teham/case_study/VQA_UI/backend/uploads"
YOLO_MODEL_PATH = os.path.join("/home/teham/case_study/VQA", "yolov12/896_anchor/runs/detect/train/weights/best.pt")
TRAINED_MODEL_PATH = "/home/teham/case_study/VQA/LLM/Meta-Llama-3.2-1B/Q6_context"
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# --- Global prompt ---
current_llm_question = "Fill the shovel"

# --- Load LLM ---
print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 2048

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
with suppress_output():
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
llm_model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_PATH)
llm_model.eval()

print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

# --- Helper functions ---
def calculate_centroid(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

def calculate_bbox_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)

def update_context(anchor_bbox, piles_info):
    pile_count = 0
    detected_piles = []
    for detection in piles_info:
        x1, y1, x2, y2 = piles_info[detection]
        centroid_x, centroid_y = calculate_centroid(x1, y1, x2, y2)
        area = calculate_bbox_area(x1, y1, x2, y2)
        pile_count += 1
        detected_piles.append({
            "name": f"pile{pile_count}",
            "position": [centroid_x, centroid_y],
            "area": area,
            "bbox": [x1, y1, x2, y2]
        })

    x1, y1, x2, y2 = anchor_bbox
    centroid_x, centroid_y = calculate_centroid(x1, y1, x2, y2)
    detected_anchor = {"position": [centroid_x, centroid_y], "bbox": [x1, y1, x2, y2]}

    llm_context_string = ""
    if detected_anchor and detected_piles:
        context_lines = [
            "You are a caterpillar in a construction site. In the following you will be given geometric data of piles of gravel such as their position and size. Alongside the piles you will be given an anchor (your position), which you can use as a reference to determine distances and relative positions.",
            f"Anchor position: ({detected_anchor['position'][0]:.1f}, {detected_anchor['position'][1]:.1f})",
            "Following piles are present:"
        ]
        for pile in sorted(detected_piles, key=lambda p: int(p['name'][4:])):
            context_lines.append(f"{pile['name']}: position=({pile['position'][0]:.1f}, {pile['position'][1]:.1f}), area={pile['area']:.1f}")
        llm_context_string = "\n".join(context_lines)

    return llm_context_string

def parse_llm_target_pile(llm_response):
    match = re.search(r"(pile\d+)", llm_response, re.IGNORECASE)
    return match.group(1).lower() if match else None

def generate_llm_response(model, tokenizer, prompt_text, max_new_tokens=64):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in decoded_output:
        return decoded_output.split("[/INST]")[-1].strip().replace("</s>", "").strip()
    return decoded_output.replace("</s>", "").strip()

def draw_boxes(frame, detections, selected_pile_bbox, anchor_bbox):
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

def extract_youtube_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

class QuestionRequest(BaseModel):
    question: str

class YouTubeRequest(BaseModel):
    url: str

# Dictionary to store download progress for each video ID
download_progress = {}

def progress_hook(d):
    """Hook to capture yt-dlp download progress."""
    video_id = d.get('info_dict', {}).get('id')
    if not video_id:
        return
    if d['status'] == 'downloading':
        downloaded_bytes = d.get('downloaded_bytes', 0)
        total_bytes = d.get('total_bytes', d.get('total_bytes_estimate', 0))
        speed = d.get('speed', 0) or 0
        eta = d.get('eta', 0) or 0
        progress = (downloaded_bytes / total_bytes) * 100 if total_bytes > 0 else 0
        download_progress[video_id] = {
            'progress': progress,
            'speed': speed / 1024 / 1024,  # Convert to MB/s
            'eta': eta,  # Seconds
            'status': 'downloading'
        }
    elif d['status'] == 'finished':
        download_progress[video_id] = {
            'progress': 100,
            'speed': 0,
            'eta': 0,
            'status': 'finished'
        }
    elif d['status'] == 'error':
        download_progress[video_id] = {
            'progress': 0,
            'speed': 0,
            'eta': 0,
            'status': 'error'
        }

@app.post("/load_youtube")
async def load_youtube(req: YouTubeRequest):
    video_id = extract_youtube_id(req.url)
    if not video_id:
        return {"status": "error", "message": "Invalid YouTube URL"}

    # Check if video exists in uploads folder
    for filename in os.listdir(VIDEO_DIR):
        if video_id in filename and filename.endswith('.mp4'):
            return {"status": "ok", "filename": filename}

    # Download video-only stream if it doesn't exist
    output_path = os.path.join(VIDEO_DIR, f"{video_id}.%(ext)s")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]',  # Best video-only, prefer mp4
        'outtmpl': output_path,
        'progress_hooks': [progress_hook],
        'quiet': False,  # Enable logs for debugging
        'noplaylist': True,
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            ),
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=True)
            filename = ydl.prepare_filename(info).split('/')[-1]
        return {"status": "ok", "filename": filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.websocket("/download_progress/{video_id}")
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

@app.post("/set_llm_question")
async def legacy_update_task(req: QuestionRequest):
    global current_llm_question
    current_llm_question = req.question
    return {"status": "ok", "task": current_llm_question}

@app.get("/stream/resume")
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

            results = yolo_model(frame)[0]
            detections = []
            anchor_bbox = None
            piles_info = {}

            pile_n = 1
            for box in results.boxes:
                bbox = [int(x.item()) for x in box.xyxy[0]]
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]
                conf = float(box.conf[0])

                detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": class_name
                })

                if class_name == "anchor":
                    anchor_bbox = bbox
                elif class_name == "pile of gravel":
                    piles_info["pile" + str(pile_n)] = bbox
                    pile_n += 1

            if anchor_bbox and piles_info:
                context = update_context(anchor_bbox, piles_info)
                full_llm_prompt = f"<s>[INST] {context}\nTask: {current_llm_question} [/INST]"
                answer = generate_llm_response(llm_model, tokenizer, full_llm_prompt)

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

@app.get("/frame/infer")
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

    results = yolo_model(frame)[0]
    detections = []
    anchor_bbox = None
    piles_info = {}

    pile_n = 1
    for box in results.boxes:
        bbox = [int(x.item()) for x in box.xyxy[0]]
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        conf = float(box.conf[0])

        detections.append({
            "bbox": bbox,
            "confidence": conf,
            "class_id": class_id,
            "class_name": class_name
        })

        if class_name == "anchor":
            anchor_bbox = bbox
        elif class_name == "pile of gravel":
            piles_info["pile" + str(pile_n)] = bbox
            pile_n += 1

    if anchor_bbox and piles_info:
        context = update_context(anchor_bbox, piles_info)
        full_llm_prompt = f"<s>[INST] {context}\nTask: {current_llm_question} [/INST]"
        answer = generate_llm_response(llm_model, tokenizer, full_llm_prompt)
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