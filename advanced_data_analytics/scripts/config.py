from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS_DIR = PROJECT_ROOT / "datasets"
PEN_RAW = DATASETS_DIR / "PenRaw_test"

RAW_IMAGES = PEN_RAW / "images"
RAW_MASKS = PEN_RAW / "masks"

NNUNET_RAW = PROJECT_ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED = PROJECT_ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS = PROJECT_ROOT / "nnUNet_results"

PREDICTIONS_INPUT = PROJECT_ROOT / "predictions" / "input"
PREDICTIONS_OUTPUT = PROJECT_ROOT / "predictions" / "output"

DATASET_ID = 1
DATASET_NAME = "PenSeg_test"

NNUNET_DATASET = NNUNET_RAW / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"