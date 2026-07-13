import os
import random
import shutil
import argparse
import numpy as np

def create_dataset_splits(source_dir, split_dir, meta_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Splits a flat image dataset into train, val, and test directories.
    """
    # Set random seeds for deterministic splitting
    random.seed(seed)
    np.random.seed(seed)

    splits = ["train", "val", "test"]
    ratios = {"train": train_ratio, "val": val_ratio, "test": 1.0 - train_ratio - val_ratio}
    
    # Create the base directories
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    classes = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
    print(f"Found classes: {classes}")

    # Create split subfolders for each class
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(split_dir, split, cls), exist_ok=True)

    # Dictionary to store split info (metadata)
    split_record = {"train": {}, "val": {}, "test": {}}

    for cls in classes:
        print(f"Processing class: {cls}...")
        cls_dir = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        images.sort()        # Ensure consistent order before shuffling
        random.shuffle(images)

        n_total = len(images)
        n_train = int(ratios["train"] * n_total)
        n_val   = int(ratios["val"] * n_total)

        train_imgs = images[:n_train]
        val_imgs   = images[n_train:n_train + n_val]
        test_imgs  = images[n_train + n_val:]

        # Record metadata
        split_record["train"][cls] = train_imgs
        split_record["val"][cls]   = val_imgs
        split_record["test"][cls]  = test_imgs

        # Helper function to copy files
        def copy_files(img_list, split_name):
            for img in img_list:
                src_path = os.path.join(cls_dir, img)
                dst_path = os.path.join(split_dir, split_name, cls, img)
                shutil.copy(src_path, dst_path)

        # Copy the images to their respective folders
        copy_files(train_imgs, "train")
        copy_files(val_imgs, "val")
        copy_files(test_imgs, "test")

        print(f"  {cls}: {len(train_imgs)} train | {len(val_imgs)} val | {len(test_imgs)} test")

    # Save split metadata ONCE
    meta_path = os.path.join(meta_dir, "split_record.npy")
    np.save(meta_path, split_record)
    
    print(f"\nDataset successfully split into {train_ratio*100:.0f}/{val_ratio*100:.0f}/{(1-train_ratio-val_ratio)*100:.0f}!")
    print(f"Images saved to: {split_dir}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split image dataset into train/val/test sets.")
    parser.add_argument("--source_dir", type=str, default="data/raw_datasets", help="Path to the original dataset with class subfolders.")
    parser.add_argument("--split_dir", type=str, default="data/dataset_split", help="Path where the split dataset will be saved.")
    parser.add_argument("--meta_dir", type=str, default="data/dataset_split_metadata", help="Path to save the split_record.npy metadata.")
    
    args = parser.parse_args()
    
    create_dataset_splits(args.source_dir, args.split_dir, args.meta_dir)