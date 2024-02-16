"""
Dataset object.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
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

    def __init__(self, root: str, split: str = "train") -> None:
        """Load PartImageNet dataset.

        Args:
            root (str): the folder in which the dataset is stored.
            split (str): the split of the dataset to load.
                Defaults to "train"; accepts also "test" and "val".

        Raises:
            ValueError: if the value of split is other than "train", "test" or "val".
        """
        # Load json file with data
        if split in ["train", "test", "val"]:
            with open(
                os.path.join(root, f"annotations/{split}/{split}.json"),
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)
        else:
            raise ValueError(f"{split} is not a valid value for split.")

        self.root = root
        self.split = split

        # Dictionary of the form image_id: image_filename
        self.img_files = {img["id"]: img["file_name"] for img in data["images"]}

        # Dictionary of the form image_id: [image_annotations]
        self.annotations = defaultdict(list)
        for annotation in data["annotations"]:
            self.annotations[annotation["image_id"]].append(annotation)

        self.categories = data["categories"]

    def __getitem__(self, index: int | slice) -> List[Dict]:
        """Returns the indexed images.

        Args:
            index (int or slice): indexes of the images to return.

        Returns:
            List: list of images and their annotations.
        """
        if isinstance(index, slice):
            return [self._get_img_data(i) for i in range(*index.indices(len(self)))]
        if isinstance(index, int):
            return [self._get_img_data(index)]

        raise TypeError(
            f"{type(self).__name__} indices must be integers or slices, not {type(index).__name__}."
        )

    def _get_img_data(self, index) -> Dict:
        """Retrieve data about the image.

        Args:
            index (int): index of the image in the collection.

        Returns:
            Dict: dictionary containing the "id", "image" pixel values and its "annotations".
        """
        img = cv2.cvtColor(
            cv2.imread(
                os.path.join(self.root, f"images/{self.split}/{self.img_files[index]}")
            ),
            cv2.COLOR_BGR2RGB,
        )
        return {"id": index, "image": img, "annotations": self.annotations[index]}

    def __len__(self) -> int:
        """Return the number of images in the dataset.

        Returns:
            int: number of images in dataset.
        """
        return len(self.img_files)
