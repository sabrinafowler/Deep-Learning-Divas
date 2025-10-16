import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import zipfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi

"""
This script:
- Downloads the Kaggle NUS-WIDE dataset incrementally to avoid memory issues.
- Builds the NUS-WIDE-10k subset (10 categories x 1000 images each, single-label only).
- Merges in the 81 concept tags for each image.
- Performs stratified train/val/test splits (80/10/10).
- Generates CSVs with:
    image_path, category, category_idx, and tag_* columns.
- Creates label_map.json for model loading.
"""

# Config
KAGGLE_DATASET = "xinleili/nuswide"
DATA_ROOT = Path("/work/cse479/sfowler14/rawData")
OUT_ROOT = Path("nuswide10k_ready")
SAMPLES_PER_CLASS = 1000
RANDOM_SEED = 42
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

np.random.seed(RANDOM_SEED)

# Download
def download_dataset():
    """Download Kaggle dataset safely (no unzip until needed)."""
    if (DATA_ROOT / "Flickr").exists():
        print("Dataset already exists, skipping download.")
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} ...")
    zip_path = api.dataset_download_files(KAGGLE_DATASET, path=DATA_ROOT, unzip=False)
    zip_file = Path(zip_path)

    # Incremental unzip to avoid memory overflow
    print("Extracting dataset incrementally...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Unzipping"):
            zip_ref.extract(member, DATA_ROOT)
    zip_file.unlink()
    print("Dataset downloaded and extracted.")


# ---------------- Preprocess ---------------- #
def preprocess_nuswide10k():
    """Builds the NUS-WIDE-10k subset and saves CSVs ready for training."""
    concepts_path = DATA_ROOT / "ConceptsList/Concepts81.txt"
    label_dir = DATA_ROOT / "Groundtruth/AllLabels"
    image_dir = DATA_ROOT / "Flickr"
    tags_file = DATA_ROOT / "NUS_WID_Tags/AllTags81.txt"

    # Load concept names
    concepts = [line.strip() for line in open(concepts_path, encoding="utf-8").readlines()]
    print(f"Loaded {len(concepts)} concepts.")

    # Load labels and tags
    label_files = sorted(label_dir.glob("Labels_*.txt"))
    all_labels = [np.loadtxt(f, dtype=int) for f in tqdm(label_files, desc="Loading labels")]
    labels_matrix = np.vstack(all_labels).T
    tags_matrix = np.loadtxt(tags_file, dtype=int)
    assert tags_matrix.shape[0] == labels_matrix.shape[0], "Tag count mismatch."

    # Associate images
    image_files = sorted(image_dir.glob("*.jpg"))
    assert len(image_files) == labels_matrix.shape[0], "Image count and labels mismatch."

    # Build DataFrame
    df = pd.DataFrame(labels_matrix, columns=concepts)
    df["image_path"] = [str(p) for p in image_files]
    df["num_labels"] = df[concepts].sum(axis=1)

    # Add tag columns
    tag_df = pd.DataFrame(tags_matrix, columns=[f"tag_{c}" for c in concepts])
    df = pd.concat([df, tag_df], axis=1)

    # Filter single-label images
    df_single = df[df["num_labels"] == 1].copy()

    # Select top 10 most common classes
    label_counts = df_single[concepts].sum().sort_values(ascending=False)
    top10 = label_counts.head(10).index.tolist()
    print("Top 10 categories:", top10)

    # Sample 1000 per class
    samples = []
    for c in top10:
        subset = df_single[df_single[c] == 1]
        samples.append(subset.sample(min(len(subset), SAMPLES_PER_CLASS), random_state=RANDOM_SEED))
    df_subset = pd.concat(samples).reset_index(drop=True)
    df_subset["category"] = df_subset[top10].idxmax(axis=1)

    # Encode category indices for model training
    label_map = {label: i for i, label in enumerate(top10)}
    df_subset["category_idx"] = df_subset["category"].map(label_map)

    # Stratified splits
    train_df, temp_df = train_test_split(
        df_subset, test_size=(1 - SPLITS["train"]),
        stratify=df_subset["category"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=SPLITS["test"] / (SPLITS["val"] + SPLITS["test"]),
        stratify=temp_df["category"], random_state=RANDOM_SEED
    )

    # Save images and CSVs
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for split, df_split in splits.items():
        split_dir = OUT_ROOT / split / "images"
        split_dir.mkdir(parents=True, exist_ok=True)
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Copying {split}"):
            dst = split_dir / Path(row["image_path"]).name
            if not dst.exists():
                shutil.copy(row["image_path"], dst)
        df_split["image_path"] = df_split["image_path"].apply(lambda p: str(Path("images") / Path(p).name))
        df_split.to_csv(OUT_ROOT / split / f"{split}.csv", index=False)

    # Save label map for easy loading
    with open(OUT_ROOT / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("Finished building NUS-WIDE-10k (PyTorch/TensorFlow ready).")


# ---------------- Run ---------------- #
if __name__ == "__main__":
    download_dataset()
    preprocess_nuswide10k()
