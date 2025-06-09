import json
import torch
import time
import math
import difflib
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Config ---
MODEL_PATH = "/home/teham/case_study/VQA/LLM/Meta-Llama-3.2-1B/Q6_context"
DETECTIONS_JSON_PATH = "/home/teham/case_study/VQA/testv1_detections.json"
METRICS_OUTPUT_PATH = "frame_metrics"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()

# --- Helper Functions ---
def load_frame_data(frame_index):
    with open(DETECTIONS_JSON_PATH, "r") as f:
        data = json.load(f)
    for frame in data:
        if frame["frame_index"] == frame_index:
            return frame["detections"]
    raise ValueError(f"Frame {frame_index} not found.")

def prepare_context(detections):
    anchor = None
    pile_data = []

    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]

        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if class_name == "anchor":
            anchor = (x, y)
        elif class_name == "pile of gravel":
            pile_data.append((class_name, x, y, area))

    if not anchor:
        raise ValueError("No anchor found in frame.")

    anchor = (round(anchor[0], 1), round(anchor[1], 1))
    rounded_piles = [(f"pile {i+1}", round(x, 1), round(y, 1), round(area, 1)) for i, (name, x, y, area) in enumerate(pile_data)]

    context = (
        f"Anchor position: ({anchor[0]}, {anchor[1]})\n"
        "Following piles are present:\n" +
        "\n".join([f"{name}: position=({x}, {y}), area={area}" for name, x, y, area in rounded_piles])
    )

    return context, rounded_piles, anchor

def euclidean_distance(p1, p2):
    return round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2), 2)

def run_inference(context, task="Fill the shovel"):
    prompt = f"<s>[INST] {context}\nTask: {task} [/INST]"

    t_encode_start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest", max_length=2048).to(DEVICE)
    t_encode_end = time.time()
    input_len = inputs["input_ids"].shape[-1]

    torch.cuda.synchronize()
    t_infer_start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    t_infer_end = time.time()

    t_decode_start = time.time()
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    t_decode_end = time.time()

    after = decoded.split("[/INST]")[-1].strip() if "[/INST]" in decoded else decoded
    first_line = after.split("\n")[0].split(".")[0].strip()

    output_len = outputs.shape[-1]
    new_tokens = output_len - input_len

    print("Inference time:", round(t_infer_end - t_infer_start, 4))

    return {
        "output": first_line,
        "input_tokens": input_len,
        "output_tokens": output_len,
        "new_tokens": new_tokens,
        "encode_time": round(t_encode_end - t_encode_start, 4),
        "inference_time": round(t_infer_end - t_infer_start, 4),
        "decode_time": round(t_decode_end - t_decode_start, 4),
    }

def extract_predicted_pile_name(llm_response, pile_names):
    # Preprocess: normalize and strip special characters
    cleaned = llm_response.lower()
    cleaned = re.sub(r'[^a-z0-9 ]', '', cleaned)  # remove punctuation/symbols
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # collapse whitespace

    # Optional debug
    print(f"🔍 Cleaned LLM response: '{cleaned}'")
    print(f"🔍 Available pile names: {pile_names}")

    matches = difflib.get_close_matches(cleaned, [p.lower() for p in pile_names], n=1, cutoff=0.4)
    return matches[0] if matches else "unknown"

def make_eval(start_frame, end_frame):
    all_results = []

    for frame_index in range(start_frame, end_frame + 1):
        print(f"\n>>> Testing Frame {frame_index} <<<")
        detections = load_frame_data(frame_index)
        context, piles, anchor = prepare_context(detections)
        result = run_inference(context, task="Fill the shovel")

        # Calculate ground-truth: closest pile
        closest_pile = min(piles, key=lambda p: euclidean_distance(anchor, (p[1], p[2]))) if piles else None
        ground_truth_pile = closest_pile[0] if closest_pile else "none"

        # Match predicted pile name from LLM response
        pile_names = [p[0] for p in piles]
        predicted_pile = extract_predicted_pile_name(result["output"], pile_names)

        print("\nContext:")
        print(context)
        print("\nLLM Response:")
        print(result["output"])
        print(f"Predicted: {predicted_pile}")
        print(f"Ground Truth: {ground_truth_pile}")

        image_results = {
            "frame_index": frame_index,
            "tasks": [
                {
                    "task": "Fill the shovel",
                    "prediction": predicted_pile,
                    "calculated": ground_truth_pile,
                    "llm_response": result["output"]
                }
            ],
            "timings": {
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "new_tokens": result["new_tokens"],
                "encode_time": result["encode_time"],
                "inference_time": result["inference_time"],
                "decode_time": result["decode_time"],
            },
            "pile_count": len(piles),
            "avg_area": round(sum(p[3] for p in piles) / len(piles), 2) if piles else 0,
            "avg_anchor_distance": round(sum(euclidean_distance(anchor, (p[1], p[2])) for p in piles) / len(piles), 2) if piles else 0,
            "context_lines": len(context.splitlines())
        }

        all_results.append(image_results)

    output_path = f"{METRICS_OUTPUT_PATH}_{start_frame}_{end_frame}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Metrics written to {output_path}")

# --- Run Eval ---
make_eval(0, 5)
