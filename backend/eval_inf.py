import json
import matplotlib.pyplot as plt

# --- Config ---
METRICS_FILE = "frame_metrics_0_300.json"
THRESHOLD_INFERENCE_TIME = 3.0  # seconds

# --- Load Metrics ---
with open(METRICS_FILE, "r") as f:
    metrics = json.load(f)

frames = [m["frame_index"] for m in metrics]

# Extract metrics
inference_times = [m["inference_time"] for m in metrics]
encode_times = [m["encode_time"] for m in metrics]
decode_times = [m["decode_time"] for m in metrics]
input_tokens = [m["input_tokens"] for m in metrics]
output_tokens = [m["output_tokens"] for m in metrics]
new_tokens = [m["new_tokens"] for m in metrics]
pile_counts = [m["pile_count"] for m in metrics]
avg_areas = [m["avg_area"] for m in metrics]
avg_distances = [m["avg_anchor_distance"] for m in metrics]
context_lines = [m["context_lines"] for m in metrics]

# Determine frames where inference_time > threshold
spike_frames = [m["frame_index"] for m in metrics if m["inference_time"] > THRESHOLD_INFERENCE_TIME]

# --- Plotting ---
# fig, axs = plt.subplots(3, 3, figsize=(18, 12))
# axs = axs.flatten()

# def add_spike_lines(ax):
#     for spike in spike_frames:
#         ax.axvline(spike, color="red", linestyle="--", linewidth=1)

# # Metric plotting
# metric_map = {
#     0: ("Inference Time (s)", inference_times),
#     1: ("Encode Time (s)", encode_times),
#     2: ("Decode Time (s)", decode_times),
#     3: ("Input Tokens", input_tokens),
#     4: ("Output Tokens", output_tokens),
#     5: ("New Tokens", new_tokens),
#     6: ("Pile Count", pile_counts),
#     7: ("Avg Pile Area", avg_areas),
#     8: ("Avg Anchor Distance", avg_distances)
# }

# for idx, (title, values) in metric_map.items():
#     axs[idx].plot(frames, values, marker="o", label=title)
#     add_spike_lines(axs[idx])
#     axs[idx].set_title(title)
#     axs[idx].set_xlabel("Frame Index")
#     axs[idx].set_ylabel(title)
#     axs[idx].grid(True)

# plt.tight_layout()
# plt.suptitle("Frame-Level LLM Metrics with Inference Time Spikes", fontsize=16, y=1.02)
# plt.subplots_adjust(top=0.92)
# plt.savefig("frame_metrics_plot.png")
# plt.show()


# --- Extra Combined Plot with Dual Y-Axis ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary y-axis (left): Output and New Tokens
ax1.plot(frames, output_tokens, marker='s', label="Output Tokens", color="green")
ax1.plot(frames, new_tokens, marker='^', label="New Tokens", color="orange")
ax1.set_xlabel("Frame Index")
ax1.set_ylabel("Tokens", color="black")
ax1.tick_params(axis='y', labelcolor='black')

# Secondary y-axis (right): Inference Time
ax2 = ax1.twinx()
ax2.plot(frames, inference_times, marker='o', label="Inference Time (s)", color="blue")
ax2.set_ylabel("Inference Time (s)", color="blue")
ax2.tick_params(axis='y', labelcolor='blue')

# Add vertical dotted lines for inference spikes
for spike in spike_frames:
    ax1.axvline(spike, color="red", linestyle="--", linewidth=1)

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

# Title and layout
plt.title("Inference Time (Right Axis) vs. Token Counts (Left Axis)")
plt.tight_layout()
plt.savefig("combined_metrics_dual_axis.png")
plt.show()
