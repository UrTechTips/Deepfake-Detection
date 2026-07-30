# preprocess.py
import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from retinaface import RetinaFace
from mlwatcher import Logger
import concurrent.futures

# --- Configuration ---
INPUT_DIRS = ["extracted_frames", "extracted_frames_faceshifter"]  # from your frame extraction
OUTPUT_ROOT = "processed_frames"
IMG_SIZE = 224
MAX_WORKERS = 4  # Number of threads (tune: 3–4)

# Initialize logger
logger = Logger("./logs/preprocess.log", verbose=False, poll_interval=5)
logger.start()


def detect_and_crop_face(image_path, save_path, img_size=IMG_SIZE):
    """
    Detects the largest face using RetinaFace, crops, resizes, and saves.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.log(f"[ERROR] Could not read {image_path}")
            return False

        # Detect faces
        faces = RetinaFace.detect_faces(img)
        if isinstance(faces, dict) and len(faces) > 0:
            # Take the largest face
            biggest_face = None
            max_area = 0
            for _, face in faces.items():
                x1, y1, x2, y2 = face["facial_area"]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    biggest_face = (x1, y1, x2, y2)

            if biggest_face is None:
                logger.log(f"[WARN] No valid face found in {image_path}")
                return False

            x1, y1, x2, y2 = biggest_face
            face_crop = img[y1:y2, x1:x2]

            # Resize to IMG_SIZE x IMG_SIZE
            face_resized = cv2.resize(face_crop, (img_size, img_size))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, face_resized)
            return True
        else:
            logger.log(f"[WARN] No faces detected in {image_path}")
            return False
    except Exception as e:
        logger.log(f"[ERROR] Exception while processing {image_path}: {e}")
        return False


def preprocess_frames(input_dir, output_dir):
    """
    Loops through extracted frames and preprocesses them with RetinaFace (multi-threaded).
    """
    logger.log(f"Starting preprocessing for {input_dir}")
    all_images = list(Path(input_dir).rglob("*.jpg"))
    results = {"success": 0, "failed": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for img_path in all_images:
            relative_path = img_path.relative_to(input_dir)
            save_path = Path(output_dir) / relative_path
            futures[executor.submit(detect_and_crop_face, str(img_path), str(save_path))] = img_path

        for f in tqdm(concurrent.futures.as_completed(futures), total=len(all_images),
                      desc=f"Preprocessing {input_dir}"):
            ok = f.result()
            if ok:
                results["success"] += 1
            else:
                results["failed"] += 1

    logger.log(f"Finished preprocessing {input_dir}. Summary: {results}")
    return results


def main():
    summary = {"success": 0, "failed": 0}
    for input_dir in INPUT_DIRS:
        output_dir = os.path.join(OUTPUT_ROOT, Path(input_dir).name)
        results = preprocess_frames(input_dir, output_dir)
        summary["success"] += results["success"]
        summary["failed"] += results["failed"]

    logger.log("=" * 30)
    logger.log(f" FINAL PREPROCESS SUMMARY: {summary}")
    logger.log("=" * 30)

    logger.stop()


if __name__ == "__main__":
    main()
