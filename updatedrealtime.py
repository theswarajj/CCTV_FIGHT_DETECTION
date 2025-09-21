import os
import time
import csv
from collections import defaultdict, deque
import argparse
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# Optional (silence some logs)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
            nn.Linear(512 * 7 * 7 * 1, 4096),  # matches training (112x112, T≈16)
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
# Face DB utilities
# ----------------------------
def build_face_database(dataset_dir="faces_dataset", out_file="faces_encodings.pkl"):
    """
    Build a simple face DB from faces_dataset/<Name>/*.jpg using face_recognition.
    """
    try:
        import face_recognition
    except Exception as e:
        print("[Face] face_recognition not installed or failed to import:", e)
        return None

    if not os.path.isdir(dataset_dir):
        print(f"[Face] Dataset folder '{dataset_dir}' not found. Skipping face DB build.")
        return None

    encodings_db = {}
    total_added = 0
    for name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, name)
        if not os.path.isdir(person_dir):
            continue
        person_encs = []
        for img_file in os.listdir(person_dir):
            path = os.path.join(person_dir, img_file)
            ext = os.path.splitext(path)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if len(encs) > 0:
                    person_encs.append(encs[0])
            except Exception as e:
                print(f"[Face] Error reading {path}: {e}")
        if person_encs:
            encodings_db[name] = person_encs
            total_added += 1
            print(f"[Face] Added {name} ({len(person_encs)} encodings)")
    if total_added == 0:
        print("[Face] No encodings built. Check your 'faces_dataset' images.")
        return None

    with open(out_file, "wb") as f:
        pickle.dump(encodings_db, f)
    print(f"[Face] Saved database to {out_file}")
    return encodings_db

def load_face_database(db_file="faces_encodings.pkl", dataset_dir="faces_dataset"):
    """
    Load encodings from pkl; if missing, auto-build from faces_dataset.
    """
    if os.path.exists(db_file):
        try:
            with open(db_file, "rb") as f:
                face_db = pickle.load(f)
            if isinstance(face_db, dict) and len(face_db) > 0:
                print(f"[Face] Loaded DB with {len(face_db)} identities from {db_file}")
                return face_db
        except Exception as e:
            print(f"[Face] Failed loading {db_file}: {e}")

    print("[Face] Building DB...")
    return build_face_database(dataset_dir, db_file)

def recognize_face_bgr(crop_bgr, face_db, tolerance=0.45):
    """
    Detect + encode face(s) in crop; return best matching name or 'Unknown'.
    """
    try:
        import face_recognition
    except Exception as e:
        # If face_recognition isn't available, just skip
        return "Unknown"

    if crop_bgr is None or crop_bgr.size == 0:
        return "Unknown"
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    if len(locs) == 0:
        return "Unknown"
    encs = face_recognition.face_encodings(rgb, locs)
    if len(encs) == 0:
        return "Unknown"
    enc = encs[0]

    best_name = "Unknown"
    # Quick compare: stop at first matching identity
    for name, known_list in face_db.items():
        matches = face_recognition.compare_faces(known_list, enc, tolerance=tolerance)
        if True in matches:
            best_name = name
            break
    return best_name

# ----------------------------
# Misc utilities
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
# Main loop with tracking + classification + face ID
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
    smooth_k=3,                # moving avg over last k probs per track
    cooldown_sec=5.0,          # per-track alert cooldown
    save_alert_snaps=True,
    save_video=False,
    out_video_path="tracked_output.mp4",
    conf=0.25,
    iou=0.45,
    face_db_file="faces_encodings.pkl",
    face_dataset_dir="faces_dataset",
    face_tolerance=0.45
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # YOLO with built-in ByteTrack
    yolo = YOLO(yolo_weights)

    # Load C3D
    c3d = C3D(num_classes=2).to(device)
    try:
        state = torch.load(c3d_weights, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(c3d_weights, map_location=device)
    c3d.load_state_dict(state)
    c3d.eval()
    softmax = nn.Softmax(dim=1)

    # Face DB (auto-build if missing)
    face_db = load_face_database(face_db_file, face_dataset_dir) or {}

    # Per-track buffers and stats
    frame_buffers = defaultdict(lambda: deque(maxlen=clip_len))  # tid -> deque of RGB frames
    prob_buffers  = defaultdict(lambda: deque(maxlen=smooth_k))  # tid -> recent fight probs
    last_alert_time = {}                                         # tid -> last alert timestamp
    track_names = {}                                             # tid -> recognized name

    # Prepare outputs
    os.makedirs("alerts", exist_ok=True)
    log_csv = os.path.join("alerts", "fight_events.csv")
    file_new = not os.path.exists(log_csv)
    log_f = open(log_csv, "a", newline="", encoding="utf-8")
    logger = csv.writer(log_f)
    if file_new:
        logger.writerow(["timestamp", "track_id", "name", "prob_fight", "x1", "y1", "x2", "y2", "source"])

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

    # Parse source display text
    src_text = str(source) if isinstance(source, str) else f"webcam{source}"

    frame_idx = 0
    for result in stream:
        frame = result.orig_img
        if frame is None:
            continue

        # Create writer once
        if save_video and vw is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(out_video_path, fourcc, 25, (frame.shape[1], frame.shape[0]))

        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            # show the frame anyway
            cv2.imshow("Fight Detection + Tracking + FaceID", frame)
            if save_video: vw.write(frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()  # (N,4)

        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            # ---- Per-track clip building ----
            person = safe_crop(frame, x1, y1, x2, y2, pad=10)
            if person is None:
                continue

            if frame_idx % stride == 0:
                resized = cv2.resize(person, (input_size, input_size))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frame_buffers[tid].append(rgb)

            prob_fight = None
            if len(frame_buffers[tid]) == clip_len:
                clip = np.stack(list(frame_buffers[tid]), axis=0)                   # T,H,W,C
                clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32) / 255.0  # C,T,H,W
                tens = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = c3d(tens)
                    probs = softmax(logits).cpu().numpy()[0]  # [NonFight, Fight]
                    prob_fight = float(probs[1])

                prob_buffers[tid].append(prob_fight)
                prob_fight = float(np.mean(prob_buffers[tid]))  # smoothed

            # ---- Face ID (do once per track or occasionally) ----
            name = track_names.get(tid, None)
            if name is None and person is not None and len(face_db) > 0:
                # Try recognize from full person crop (contains the face)
                name = recognize_face_bgr(person, face_db, tolerance=face_tolerance)
                track_names[tid] = name

            # ---- Drawing & alerts ----
            label = f"ID {tid}"
            color = (0, 200, 0)

            if name is not None:
                label = f"{name} | {label}"

            if prob_fight is not None:
                label = f"{label} | FIGHT {prob_fight:.2f}"
                red = int(255 * min(1.0, prob_fight))
                green = int(255 * (1.0 - min(1.0, prob_fight)))
                color = (0, green, red)

                # per-track cooldown to reduce spam
                now = time.time()
                last_t = last_alert_time.get(tid, 0.0)
                if prob_fight >= prob_threshold and (now - last_t) >= cooldown_sec:
                    last_alert_time[tid] = now

                    # snapshot & log
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    if save_alert_snaps:
                        snap_path = os.path.join("alerts", f"fight_tid{tid}_{ts}.jpg")
                        cv2.imwrite(snap_path, frame)

                    logger.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        tid,
                        name or "Unknown",
                        f"{prob_fight:.3f}",
                        int(x1), int(y1), int(x2), int(y2),
                        src_text
                    ])
                    log_f.flush()

                    cv2.putText(frame, "ALERT: FIGHT DETECTED!", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            draw_label(frame, label, int(x1), max(0, int(y1) - 8), color)

        cv2.imshow("Fight Detection + Tracking + FaceID", frame)
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
    p.add_argument("--face-db", type=str, default="faces_encodings.pkl")
    p.add_argument("--face-dataset", type=str, default="faces_dataset")
    p.add_argument("--face-tolerance", type=float, default=0.45)
    args = p.parse_args()

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
        face_db_file=args.face_db,
        face_dataset_dir=args.face_dataset,
        face_tolerance=args.face_tolerance,
    )
