import cv2
import os
import json
import torch
import time
import math
import warnings
import re
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
from ultralytics import YOLO # Keep for class_id mapping, though not for inference
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

# --- Configuration for LLM ---
TRAINED_MODEL_PATH = "/home/teham/case_study/VQA/LLM/Meta-Llama-3.2-1B/Q6_context"
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# --- Load LLM Model and Tokenizer ---
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

# --- Helper Functions ---
def generate_llm_response(model, tokenizer, prompt_text, max_new_tokens=64):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    inference_time = time.time() - start_time
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in decoded_output:
        answer = decoded_output.split("[/INST]")[-1].strip()
    else:
        answer = decoded_output.strip()
    return answer.replace("</s>", "").strip(), inference_time


# --- Load Precomputed Detections ---
# PRECOMPUTED_DETECTIONS_PATH = "/home/teham/case_study/VQA/testv1_detections.json"
PRECOMPUTED_DETECTIONS_PATH = "/home/teham/case_study/VQA/yolov12/896_anchor/detections/v5_detections/v5_detections.json"

print("Loading precomputed detections from JSON...")
try:
    with open(PRECOMPUTED_DETECTIONS_PATH, 'r') as f:
        precomputed_detections_data = json.load(f)
    print(f"Successfully loaded {len(precomputed_detections_data)} image entries from {PRECOMPUTED_DETECTIONS_PATH}")
except FileNotFoundError:
    print(f"Error: Precomputed detections file not found at {PRECOMPUTED_DETECTIONS_PATH}. Exiting.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {PRECOMPUTED_DETECTIONS_PATH}. Please check file format. Exiting.")
    exit()

frame = 2
frame = "vid5_2_frame_00022_jpg.rf.0cb373d6fa4b4a9dc1c304c67f5fffb9.jpg"
detected_piles = []
detected_anchor = None
# print(precomputed_detections_data[frame_n])
all_detections_data = precomputed_detections_data.get(frame, [])


# Class IDs (ensure these match your JSON's class_id)
PILE_CLASS_ID = 1
ANCHOR_CLASS_ID = 0

def calculate_centroid(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

def calculate_bbox_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)

pile_count = 0 # To assign pile names correctly, starting fresh for each image
for detection in all_detections_data:
    x1, y1, x2, y2 = detection['bbox']
    cls = detection['class_id']
    conf = detection['confidence']

    # Note: We are using the hardcoded PILE_CLASS_ID and ANCHOR_CLASS_ID.
    # If your JSON uses different IDs or relies on 'class_name' string,
    # you might need to adjust this logic.
    if cls == PILE_CLASS_ID:
        centroid_x, centroid_y = calculate_centroid(x1, y1, x2, y2)
        area = calculate_bbox_area(x1, y1, x2, y2)
        pile_count += 1
        detected_piles.append({
            "name": f"pile{pile_count}",
            "position": [centroid_x, centroid_y],
            "area": area,
            "bbox": [x1, y1, x2, y2]
        })
    elif cls == ANCHOR_CLASS_ID:
        centroid_x, centroid_y = calculate_centroid(x1, y1, x2, y2)
        detected_anchor = {"position": [centroid_x, centroid_y], "bbox": [x1, y1, x2, y2]}


# Construct LLM Context
llm_context_string = ""
if detected_anchor and detected_piles:
    llm_context_gen_start_time = time.time()
    context_lines = [
        "You are a caterpillar in a construction site. In the following you will be given geometric data of piles of gravel such as their position and size. Alongside the piles you will be given an anchor (your position), which you can use as a reference to determine distances and relative positions.",
        f"Anchor position: ({detected_anchor['position'][0]:.1f}, {detected_anchor['position'][1]:.1f})",
        "Following piles are present:"
    ]
    for pile in sorted(detected_piles, key=lambda p: int(p['name'][4:])): # Sort by pile number for consistent context
        context_lines.append(f"{pile['name']}: position=({pile['position'][0]:.1f}, {pile['position'][1]:.1f}), area={pile['area']:.1f}")
    llm_context_string = "\n".join(context_lines)



# LLM Questions
LLM_QUESTIONS = [
    "Start at the rightmost pile",
    "Clear a remote pile",
    "Clear a pile as fast as possible",
    "Start at the leftmost pile",
    "Process the largest pile",
    "Fill the shovel"
]

def parse_llm_target_pile(llm_response):
    match = re.search(r"(pile\d+)", llm_response, re.IGNORECASE)
    return match.group(1).lower() if match else None

for current_llm_question in LLM_QUESTIONS:
        llm_task_start_time = time.time() # Time for processing this specific LLM question
        llm_predicted_target_pile_name = None
        llm_response = None
        llm_inf_time = 0

        if llm_context_string: # Only run LLM if context was successfully generated (i.e., detections exist)
            full_llm_prompt = f"<s>[INST] {llm_context_string}\nTask: {current_llm_question} [/INST]"
            llm_response, llm_inf_time = generate_llm_response(llm_model, tokenizer, full_llm_prompt)
            llm_predicted_target_pile_name = parse_llm_target_pile(llm_response)
            # Pre/post processing for LLM inference (e.g., tokenization) will be minimal here
            # as context generation is separate.
            # We'll just add tokenization/decoding time for this specific LLM call
            # which is already part of generate_llm_response if not explicitly separated.

        # Calculate Ground Truth

        # Log Results for this task

        print("\ntask", current_llm_question)
        print("prediction", llm_predicted_target_pile_name)
        print("llm_response", llm_response)

