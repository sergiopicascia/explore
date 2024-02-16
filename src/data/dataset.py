"""
Dataset object.
"""

from abc import ABC, abstractmethod
from typing import Dict
import os
import json
from collections import defaultdict
import cv2


class Dataset(ABC):
    """
    Abstract class for dataset.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __getitem__(self, index) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class PartImageNetDataset(Dataset):
    """
    Object for loading the PartImageNet dataset.
    """

    def __init__(self, root, split) -> None:
        self.root = root
        self.split = split

        if split in ["train", "test", "val"]:
            with open(
                os.path.join(root, f"annotations/{split}/{split}.json"),
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)
        else:
            raise ValueError

        self.img_files = {img["id"]: img["file_name"] for img in data["images"]}

        self.annotations = defaultdict(list)
        for annotation in data["annotations"]:
            self.annotations[annotation["image_id"]].append(annotation)

        self.categories = data["categories"]

    def __getitem__(self, index) -> Dict:
        img = cv2.cvtColor(
            cv2.imread(
                os.path.join(self.root, f"images/{self.split}/{self.img_files[index]}")
            ),
            cv2.COLOR_BGR2RGB,
        )
        return {"id": index, "image": img, "annotations": self.annotations[index]}

    def __len__(self) -> int:
        return len(self.img_files)
