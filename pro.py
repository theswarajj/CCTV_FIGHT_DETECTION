import os
import cv2
import glob
import numpy as np

def preprocess_videos(input_dir, output_dir, clip_len=16, stride=2, size=112, label=None):
    os.makedirs(output_dir, exist_ok=True)

    videos = glob.glob(os.path.join(input_dir, "*.avi"))
    print(f"Found {len(videos)} videos in {os.path.basename(input_dir)}")

    for v in videos:
        cap = cv2.VideoCapture(v)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (size, size))
            frames.append(frame)
        cap.release()

        total = len(frames)
        idx = 0
        for i in range(0, total - clip_len * stride, clip_len * stride):
            clip = []
            for j in range(i, i + clip_len * stride, stride):
                clip.append(frames[j])

            clip = np.array(clip)  # T,H,W,C
            clip = clip[:, :, :, ::-1]  # BGR->RGB
            clip = np.transpose(clip, (3, 0, 1, 2))  # C,T,H,W

            out_path = os.path.join(output_dir, f"{os.path.basename(v).split('.')[0]}_{idx}.npy")
            np.save(out_path, clip)
            idx += 1

    print(f"Processed {os.path.basename(input_dir)} -> saved clips to {output_dir}")


if __name__ == "__main__":
    # Train folders
    preprocess_videos("RWF-2000/train/Train_Fight", "processed_dataset/Fight")
    preprocess_videos("RWF-2000/train/Train_NonFight", "processed_dataset/NonFight")

    # Validation folders
    preprocess_videos("RWF-2000/val/Val_Fight", "processed_dataset/Fight")
    preprocess_videos("RWF-2000/val/Val_NonFight", "processed_dataset/NonFight")

    preprocess_videos("RWF-2000/val/Val_NonFight", "processed_dataset/NonFight")
