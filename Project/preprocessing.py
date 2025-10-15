import os
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
'''
This script creates the NUS-WIDE-10k dataset subset from the full NUS-WIDE
dataset hosted on Hugging Face (https://huggingface.co/datasets/Lxyhaha/NUS-WIDE).

It automatically downloads the dataset if it is not already cached locally.

Process overview:
1. Loads the NUS-WIDE dataset from Hugging Face using the `datasets` library.
2. Identifies the 10 most frequent categories (labels).
3. Randomly samples 1000 image–text pairs from each of those categories,
   ensuring each instance belongs to only one category.
4. Performs a stratified split:
      - 8000 samples for training
      - 1000 samples for validation
      - 1000 samples for testing
   maintaining equal category proportions in each split.
5. Saves all images and their metadata into:
      nuswide10k/
        ├── train/
        │    ├── images + train.csv
        ├── val/
        │    ├── images + val.csv
        └── test/
             ├── images + test.csv

Usage:
1. Make sure you have Python 3.8+ and install the required packages:
       pip install datasets pandas scikit-learn tqdm
2. Run this script:
       python preprocess_nuswide10k.py
3. The resulting dataset will appear in the `nuswide10k/` directory.
'''

# 1. Load dataset
dataset = load_dataset("Lxyhaha/NUS-WIDE", split="train")
df = pd.DataFrame(dataset)

# 2. Expand multi-label rows and count category frequencies
df_exploded = df.explode('labels')
label_counts = df_exploded['labels'].value_counts()

# 3. Select top 10 categories
top_10_labels = label_counts.head(10).index.tolist()
print("Top 10 categories:", top_10_labels)

# 4. Filter and ensure unique image–text pairs
df_filtered = df_exploded[df_exploded['labels'].isin(top_10_labels)]
df_filtered = df_filtered.drop_duplicates(subset=['image', 'text'])

# 5. Randomly sample 1000 per category (reproducibly)
dfs = []
for label in top_10_labels:
    subset = df_filtered[df_filtered['labels'] == label]
    subset_sampled = subset.sample(n=min(1000, len(subset)), random_state=42)
    dfs.append(subset_sampled)

df_10k = pd.concat(dfs, ignore_index=True)
print("Final subset size:", len(df_10k))

# 6. Stratified split into train/val/test: 80% / 10% / 10%
train_df, temp_df = train_test_split(
    df_10k,
    test_size=0.2,
    stratify=df_10k['labels'],
    random_state=42
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['labels'],
    random_state=42
)

splits = {"train": train_df, "val": val_df, "test": test_df}
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 7. Make output folders
base_dir = "nuswide10k"
os.makedirs(base_dir, exist_ok=True)
for split in splits:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)

# 8. Save images and metadata
for split, df_split in splits.items():
    print(f"Saving {split} images...")
    image_dir = os.path.join(base_dir, split)
    rows = []
    for i, row in tqdm(df_split.iterrows(), total=len(df_split)):
        image = row['image']  # PIL image
        label = row['labels']
        text = row['text']
        filename = f"{split}_{i:05d}.jpg"
        image.save(os.path.join(image_dir, filename))
        rows.append({"filename": filename, "text": text, "label": label})

    pd.DataFrame(rows).to_csv(os.path.join(base_dir, f"{split}.csv"), index=False)

print("Stratified dataset saved under:", base_dir)
