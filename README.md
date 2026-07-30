# Deepfake Detection

>A collection of preprocessing scripts and PyTorch notebooks for detecting deepfakes using Vision Transformers (ViT) + temporal models.

## What this project does
- Extracts frames from FaceForensics++ videos and samples chunks for training ([src/extract_frames.py](src/extract_frames.py)).
- Detects and crops faces from frames using RetinaFace and saves processed images ([src/preprocessing.py](src/preprocessing.py)).
- Implements and experiments with ViT-based per-frame feature extraction plus temporal pooling / BiLSTM heads in notebooks:
  - [src/Proposed Model.ipynb](src/Proposed Model.ipynb)
  - [src/VitB16-BiLSTM.ipynb](src/VitB16-BiLSTM.ipynb)
  - [src/VitB16-BiLSTM-TemporalStem.ipynb](src/VitB16-BiLSTM-TemporalStem.ipynb)
  - [src/VitL16-BiLSTM.ipynb](src/VitL16-BiLSTM.ipynb)
  - [src/cnn-pooling.ipynb](src/cnn-pooling.ipynb)

## Why this is useful
- Reproducible preprocessing pipeline for popular deepfake datasets (FaceForensics++).
- Modular notebooks for trying different ViT backbones and temporal aggregation strategies.
- Quick utilities to produce per-frame or chunked training data ready for model training.

## Quick start
Prerequisites: Python 3.8+ and Git.

1. Clone the repo

```bash
git clone <your-repo-url>
cd "Deepfake Detection"
```

2. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

There is no `requirements.txt` in the repository. Install core packages and then follow the official PyTorch install guide for your CUDA/CPU configuration.

```bash
pip install numpy pandas opencv-python tqdm retinaface mlwatcher matplotlib seaborn scikit-learn jupyterlab
# Install PyTorch and torchvision per https://pytorch.org (select appropriate CUDA)
```

Optional: pin versions and create `requirements.txt` with `pip freeze > requirements.txt`.

4. Run preprocessing scripts

- Extract frames (example):

```bash
python src/extract_frames.py
```

- Preprocess frames (face detection & crop):

```bash
python src/preprocessing.py
```

5. Open the notebooks

Start Jupyter and open the notebooks to run experiments and training loops:

```bash
jupyter lab
```

## Project layout
- `src/extract_frames.py` — frame extraction and sampling logic from the FaceForensics++ archive.
- `src/preprocessing.py` — face detection (RetinaFace) + crop, resizing and multi-threaded preprocessing.
- `src/*.ipynb` — model definition experiments (ViT backbones, BiLSTM temporal heads, pooling variants).

## Dependencies (summary)
- Python 3.8+
- numpy, pandas, opencv-python (cv2), tqdm
- retinaface (face detector)
- mlwatcher (lightweight run logger used in preprocessing)
- matplotlib, seaborn, scikit-learn
- PyTorch, torchvision (used in notebooks; ViT backbones come from `torchvision.models`)

## Results
The experiments in the notebooks use the FaceForensics++ dataset (FaceSwap subset) and report the following video-level detection metrics for the proposed ViT + temporal model:

- **Proposed Model (FaceSwap):** Accuracy 94.67%, AUC 98.31%, F1-score 95.00%

Comparison of architectures evaluated on the FaceSwap subset:

- **CNN + Mean Pooling:** Accuracy 86.67%, AUC 94.68%, F1-score 87.00%
- **ViT B/16 + BiLSTM:** Accuracy 88.67%, AUC 95.14%, F1-score 88.00%
- **ViT L/16 + BiLSTM:** Accuracy 92.00%, AUC 97.01%, F1-score 91.00%
- **ViT B/16 + Temporal Stem + BiLSTM:** Accuracy 92.33%, AUC 97.79%, F1-score 92.00%
- **Proposed Model:** Accuracy 94.67%, AUC 98.31%, F1-score 95.00%

Notes:
- Reported numbers are from the project notebooks and accompanying paper; they were obtained using the FaceForensics++ FaceSwap videos with held-out validation and test splits (15% each as used in experiments).
- See the notebooks in `src/` for training details, evaluation scripts, and plots (ROC, confusion matrix, loss curves).

## Where to get help
- Open an issue in the repository with a reproducible example.
- For third-party package issues, consult each project's docs (PyTorch, RetinaFace, OpenCV).

## Contributing
If you'd like to contribute, please open a GitHub issue or a pull request.