import os
import time
import csv
from collections import defaultdict, deque
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ----------------------------
# C3D model (same as training)
# ----------------------------
class C3D(nn.Module):
    def __init__(self, num_classes=2):
        super(C3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------------
# Utilities
# ----------------------------
def safe_crop(img, x1, y1, x2, y2, pad=8):
    h, w = img.shape[:2]
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(w, int(x2 + pad))
    y2 = min(h, int(y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def draw_label(img, text, x, y, color, font_scale=0.6, thickness=2):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

# ----------------------------
# Main loop with tracking + classification
# ----------------------------
def run(
    source=0,
    yolo_weights="yolov8n.pt",
    tracker_yaml="bytetrack.yaml",
    c3d_weights="checkpoints/best_c3d.pth",
    clip_len=16,
    stride=1,
    input_size=112,
    prob_threshold=0.60,       # threshold for fight
    smooth_k=3,                # moving average over last k clip probs per track
    cooldown_sec=5.0,          # per-track alert cooldown
    save_alert_snaps=True,
    save_video=False,
    out_video_path="tracked_output.mp4",
    conf=0.25,
    iou=0.45
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # YOLO with built-in tracking (ByteTrack via YAML)
    yolo = YOLO(yolo_weights)

    # Load C3D
    c3d = C3D(num_classes=2).to(device)
    # Safe load: you saved state_dict, so weights_only=True is fine
    try:
        state = torch.load(c3d_weights, map_location=device, weights_only=True)
    except TypeError:
        # older torch versions don't support weights_only
        state = torch.load(c3d_weights, map_location=device)
    c3d.load_state_dict(state)
    c3d.eval()
    softmax = nn.Softmax(dim=1)

    # Per-track buffers and stats
    frame_buffers = defaultdict(lambda: deque(maxlen=clip_len))  # tid -> deque of frames (RGB)
    prob_buffers = defaultdict(lambda: deque(maxlen=smooth_k))   # tid -> recent fight probabilities
    last_alert_time = {}                                         # tid -> last alert timestamp

    # Prepare outputs
    os.makedirs("alerts", exist_ok=True)
    log_csv = os.path.join("alerts", "fight_events.csv")
    file_new = not os.path.exists(log_csv)
    log_f = open(log_csv, "a", newline="", encoding="utf-8")
    logger = csv.writer(log_f)
    if file_new:
        logger.writerow(["timestamp", "track_id", "prob_fight", "x1", "y1", "x2", "y2", "source"])

    # Setup writer lazily
    vw = None

    # Tracking stream
    stream = yolo.track(
        source=source,
        stream=True,
        tracker=tracker_yaml,
        classes=[0],   # person
        conf=conf,
        iou=iou,
        verbose=False
    )

    frame_idx = 0
    for result in stream:
        frame = result.orig_img
        if frame is None:
            continue

        # Create writer once
        if save_video and vw is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(out_video_path, fourcc, 25, (frame.shape[1], frame.shape[0]))

        # Extract tracked boxes
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            cv2.imshow("Fight Detection + Tracking", frame)
            if save_video: vw.write(frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
            continue

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()  # (N,4)

        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            # build per-track clip buffers
            person = safe_crop(frame, x1, y1, x2, y2, pad=10)
            if person is None:
                continue

            if frame_idx % stride == 0:
                resized = cv2.resize(person, (input_size, input_size))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frame_buffers[tid].append(rgb)

            # classify when buffer is full
            prob_fight = None
            if len(frame_buffers[tid]) == clip_len:
                clip = np.stack(list(frame_buffers[tid]), axis=0)                 # T,H,W,C
                clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32) / 255.0  # C,T,H,W
                tens = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = c3d(tens)
                    probs = softmax(logits).cpu().numpy()[0]  # [NonFight, Fight]
                    prob_fight = float(probs[1])

                prob_buffers[tid].append(prob_fight)
                # smoothed probability
                prob_fight = float(np.mean(prob_buffers[tid]))

            # draw
            label = f"ID {tid}"
            color = (0, 200, 0)
            if prob_fight is not None:
                label = f"ID {tid} | FIGHT {prob_fight:.2f}"
                red = int(255 * min(1.0, prob_fight))
                green = int(255 * (1.0 - min(1.0, prob_fight)))
                color = (0, green, red)

                # trigger alert with cooldown
                now = time.time()
                last_t = last_alert_time.get(tid, 0.0)
                if prob_fight >= prob_threshold and (now - last_t) >= cooldown_sec:
                    last_alert_time[tid] = now
                    # snapshot & log
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    if save_alert_snaps:
                        snap_path = os.path.join("alerts", f"fight_tid{tid}_{ts}.jpg")
                        cv2.imwrite(snap_path, frame)
                    logger.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), tid, f"{prob_fight:.3f}",
                                     int(x1), int(y1), int(x2), int(y2), str(source)])
                    log_f.flush()
                    # banner
                    cv2.putText(frame, "ALERT: FIGHT DETECTED!", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            draw_label(frame, label, int(x1), max(0, int(y1) - 8), color)

        # render
        cv2.imshow("Fight Detection + Tracking", frame)
        if save_video: vw.write(frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    if vw is not None:
        vw.release()
    log_f.close()
    cv2.destroyAllWindows()

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0", help="0 for webcam, path to video.mp4, or RTSP URL")
    p.add_argument("--yolo-weights", type=str, default="yolov8n.pt")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml")
    p.add_argument("--c3d-weights", type=str, default="checkpoints/best_c3d.pth")
    p.add_argument("--clip-len", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--size", type=int, default=112)
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--smooth-k", type=int, default=3)
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument("--save-snaps", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--out", type=str, default="tracked_output.mp4")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    args = p.parse_args()

    # Parse source (int "0" -> 0)
    src = 0 if args.source.strip() == "0" else args.source

    run(
        source=src,
        yolo_weights=args.yolo_weights,
        tracker_yaml=args.tracker,
        c3d_weights=args.c3d_weights,
        clip_len=args.clip_len,
        stride=args.stride,
        input_size=args.size,
        prob_threshold=args.threshold,
        smooth_k=args.smooth_k,
        cooldown_sec=args.cooldown,
        save_alert_snaps=args.save_snaps,
        save_video=args.save_video,
        out_video_path=args.out,
        conf=args.conf,
        iou=args.iou,
    )
