import os
import pickle
import face_recognition

def build_face_database(dataset_dir="faces_dataset", out_file="faces_encodings.pkl"):
    encodings_db = {}
    for name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, name)
        if not os.path.isdir(person_dir):
            continue
        person_encs = []
        for img_file in os.listdir(person_dir):
            path = os.path.join(person_dir, img_file)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if len(encs) > 0:
                    person_encs.append(encs[0])
            except Exception as e:
                print(f"[Error] {path}: {e}")
        if person_encs:
            encodings_db[name] = person_encs
            print(f"[DB] Added {name} with {len(person_encs)} encodings")

    with open(out_file, "wb") as f:
        pickle.dump(encodings_db, f)
    print(f"[DB] Saved database to {out_file}")

if __name__ == "__main__":
    build_face_database("faces_dataset")
