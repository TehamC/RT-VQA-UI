import re
import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os
from config import TRAINED_MODEL_PATH, BASE_MODEL_NAME

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

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

def calculate_centroid(x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float]:
    return (x1 + x2) / 2, (y1 + y2) / 2

def calculate_bbox_area(x1: int, y1: int, x2: int, y2: int) -> float:
    return (x2 - x1) * (y2 - y1)

def update_context(anchor_bbox: List[int], piles_info: Dict[str, List[int]]) -> str:
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

def parse_llm_target_pile(llm_response: str) -> str:
    match = re.search(r"(pile\d+)", llm_response, re.IGNORECASE)
    return match.group(1).lower() if match else None

def generate_llm_response(prompt_text: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        outputs = llm_model.generate(
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