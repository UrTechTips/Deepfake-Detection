import cv2
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import threading
import random

# --- Configuration ---
ROOT_DIR = "./archive/FaceForensics++_C23/"

# --- Global Shared Resources ---
# A thread-safe way to store metadata from concurrent operations
training_metadata = []
metadata_lock = threading.Lock()

def extract_frames(video_path, output_folder):
    """
    Extracts frames from a single video file.
    - If video has >= 64 frames, it extracts 4 random chunks of 16 consecutive frames.
    - If video has < 64 frames, it extracts all available frames.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return "failed", 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"[ERROR] No frames in video: {video_path}")
        cap.release()
        return "failed", 0

    # --- MODIFIED LOGIC STARTS HERE ---

    # 1. Define constants
    NUM_CHUNKS = 4
    CHUNK_SIZE = 16
    TOTAL_TARGET_FRAMES = NUM_CHUNKS * CHUNK_SIZE  # 64

    # 2. Check if frames are already extracted
    # Adjust expected frames based on video length
    expected_frames = min(total_frames, TOTAL_TARGET_FRAMES)
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) >= expected_frames:
        cap.release()
        return "skipped", len(os.listdir(output_folder))

    os.makedirs(output_folder, exist_ok=True)
    saved_frames_count = 0

    # 3. Handle short and long videos differently
    if total_frames < TOTAL_TARGET_FRAMES:
        # --- SHORT VIDEO LOGIC: Extract all frames ---
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_file = f"frame_{saved_frames_count:04d}.jpg"
            output_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_path, frame)
            saved_frames_count += 1
    
    else:
        # --- LONG VIDEO LOGIC: Extract random chunks ---
        try:
            possible_starts = range(total_frames - CHUNK_SIZE)
            start_frames = sorted(random.sample(possible_starts, NUM_CHUNKS))
        except ValueError:
            print(f"[WARN] Cannot sample {NUM_CHUNKS} chunks from video: {video_path}")
            cap.release()
            return "failed", 0

        for start in start_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(CHUNK_SIZE):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_file = f"frame_{saved_frames_count:04d}.jpg"
                output_path = os.path.join(output_folder, frame_file)
                cv2.imwrite(output_path, frame)
                saved_frames_count += 1
            if not ret:
                break # Stop processing video if a frame read fails

    # --- MODIFIED LOGIC ENDS HERE ---
    
    cap.release()
    
    if saved_frames_count > 0:
        return "success", saved_frames_count
    else:
        print(f"[WARN] No frames were extracted for video: {video_path}")
        return "failed", 0

def process_dataset_file(dataset, source_csv, output_dir):
    """
    Processes a DataFrame of videos using a thread pool and displays a progress bar.
    """
    print(f"Starting processing for {source_csv} with {len(dataset)} entries.")
    results = {"success": 0, "skipped": 0, "failed": 0}
    
    with ThreadPoolExecutor(max_workers=8) as executor: # Increased workers for I/O bound tasks
        futures = []
        for _, row in dataset.iterrows():
            video_path = os.path.join(ROOT_DIR, row['File Path'])
            label = str(row['Label'])
            output_folder = os.path.join(
                output_dir, label, Path(video_path).parent.name,
                os.path.splitext(os.path.basename(video_path))[0]
            )
            futures.append(executor.submit(extract_frames, video_path, output_folder))

        # Use tqdm for a live progress bar
        progress_bar = tqdm(as_completed(futures), total=len(dataset), desc=f"Extracting from {source_csv}")
        for future in progress_bar:
            try:
                status, num_frames = future.result()
                results[status] += 1
                progress_bar.set_postfix({
                    'Success': results['success'],
                    'Skipped': results['skipped'],
                    'Failed': results['failed']
                })
            except Exception as e:
                print(f"[CRITICAL] A thread failed with an exception: {e}")
                results["failed"] += 1

    print(f"Finished processing {source_csv}. Summary: {results}")
    return results


def process_csv_file(csv_file, output_dir, num_rows=100):
    """
    Reads a CSV file, samples it, and initiates the processing.
    """
    print(f"--- Loading CSV file: {csv_file} ---")
    try:
        df = pd.read_csv(os.path.join(ROOT_DIR, "csv", csv_file))
        # Ensure we don't sample more rows than available
        sample_size = min(num_rows, len(df))
        print(f"Sampling {sample_size} random rows from the dataset.")
        random_rows = df.sample(n=sample_size, random_state=42)
        return process_dataset_file(random_rows, csv_file, output_dir)
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found: {csv_file}")
        return {"success": 0, "skipped": 0, "failed": 0}


def main(csv_files, output_dir, output_csv):
    """
    Main function to orchestrate the frame extraction process.
    """

    all_results = {"success": 0, "skipped": 0, "failed": 0}
    for csv_file in csv_files:
        results = process_csv_file(csv_file, output_dir, num_rows=1000)
        for key in all_results:
            all_results[key] += results[key]

    print("="*30)
    print(f" FINAL SUMMARY: {all_results}")
    print("="*30)
    
if __name__ == "__main__":
    
    # CSV_FILES = ['DeepFakeDetection.csv', 'original.csv']
    # OUTPUT_DIR = "extracted_frames"
    # main(CSV_FILES, OUTPUT_DIR, output_csv="training_data.csv")

    CSV_FILES_2 = ['FaceShifter.csv']
    OUTPUT_DIR_2 = "extracted_frames_faceshifter_continous"
    main(CSV_FILES_2, OUTPUT_DIR_2, output_csv="training_data_faceshifter.csv")
    CSV_FILES_2 = ['Face2Face.csv']
    OUTPUT_DIR_2 = "extracted_frames_face2face_continous"
    main(CSV_FILES_2, OUTPUT_DIR_2, output_csv="training_data_face2face.csv")