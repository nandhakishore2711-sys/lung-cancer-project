"""
Split a flat "one folder per class" dataset into train/val/test folders.

Before:
    data_raw/
      Benign/    *.png
      Malignant/ *.png
      Normal/    *.png

After (default 70/15/15 split):
    data/
      train/Benign/... train/Malignant/... train/Normal/...
      val/Benign/...   val/Malignant/...   val/Normal/...
      test/Benign/...  test/Malignant/...  test/Normal/...

Usage:
    python scripts/split_dataset.py --input_dir data_raw --output_dir data \
        --train 0.7 --val 0.15 --test 0.15
"""

import argparse
import os
import random
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing one subfolder per class")
    parser.add_argument("--output_dir", required=True, help="Where to write train/val/test folders")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them (default: copy)")
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "train+val+test must sum to 1.0"

    random.seed(args.seed)
    class_names = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    print("Found classes:", class_names)

    for split in ["train", "val", "test"]:
        for cname in class_names:
            os.makedirs(os.path.join(args.output_dir, split, cname), exist_ok=True)

    for cname in class_names:
        src_dir = os.path.join(args.input_dir, cname)
        files = [f for f in os.listdir(src_dir) if not f.startswith(".")]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * args.train)
        n_val = int(n * args.val)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split, split_files in splits.items():
            dst_dir = os.path.join(args.output_dir, split, cname)
            for fname in split_files:
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, fname)
                shutil.copy2(src_path, dst_path)

        print(f"{cname}: {n} total -> train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}")

    print(f"\nDone. Dataset written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
