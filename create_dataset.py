from glob import glob
from pathlib import Path as p
from tqdm import tqdm
import shutil


def create_dataset():
    # Take emations list from source dataset
    emotions_list = [e.split("/")[1] for e in glob("source_dataset/*")]

    # Create splited dataset for future training and evaluation
    if p("dataset").exists():
        shutil.rmtree("dataset")
    subsets = "train", "test", "val"
    for subset in subsets:
        (p("dataset") / p(subset)).mkdir(exist_ok=True, parents=True)

    # Split dataset in proportions 65% / 20% / 15%
    split_ratios = (0, 0.65), (0.65, 0.85), (0.85, 1)

    # Create dataset for future training
    for subset, split_ratio in zip(subsets, split_ratios):
        for emotion in emotions_list:
            src_files = glob(f"source_dataset/{emotion}/*")
            first, last = [round(i * len(src_files)) for i in split_ratio]
            src_files = src_files[first:last]
            for src in src_files:
                p(f"dataset/{subset}/{emotion}").mkdir(exist_ok=True)
                dst = f"dataset/{subset}/{emotion}/{p(src).name}"
                shutil.copy(src=src, dst=dst)


if __name__ == "__main__":
    create_dataset()
