"""
BDD100K Detection Dataset Definition.

This module defines a custom dataset class for BDD100K detection
tasks using MMDetection. It extends CocoDataset and provides
a custom result formatting method to export predictions into
BDD100K evaluation format.

Author: Your Name
Project: Bosch Assignment
"""

import os
import os.path as osp
from typing import List

import numpy as np
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset
from scalabel.label.io import save
from scalabel.label.transforms import bbox_to_box2d
from scalabel.label.typing import Frame, Label


@DATASETS.register_module()
class BDD100KDetDataset(CocoDataset):  # type: ignore
    """
    BDD100K Dataset class for object detection.

    This class extends MMDetection's CocoDataset and:
    - Defines BDD100K detection class labels
    - Converts model predictions into BDD100K JSON format
    """

    CLASSES = [
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    ]

    def convert_format(
        self, results: List[List[np.ndarray]], out_dir: str  # type: ignore
    ) -> None:
        """
        Convert detection results to BDD100K prediction format.

        Args:
            results (List[List[np.ndarray]]):
                Detection results for each image.
                Each element corresponds to one image and contains
                a list of bounding boxes per category.

            out_dir (str):
                Directory where the formatted JSON file will be saved.

        Raises:
            AssertionError:
                If results is not a list or length mismatch occurs.

        Output:
            Saves a JSON file named 'det.json' in the specified directory.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), f"Length of res and dset not equal: {len(results)} != {len(self)}"

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        frames = []
        ann_id = 0

        for img_idx in range(len(self)):
            img_name = self.data_infos[img_idx]["file_name"]
            frame = Frame(name=img_name, labels=[])
            frames.append(frame)

            result = results[img_idx]
            for cat_idx, bboxes in enumerate(result):
                for bbox in bboxes:
                    ann_id += 1
                    label = Label(
                        id=ann_id,
                        score=bbox[-1],
                        box2d=bbox_to_box2d(self.xyxy2xywh(bbox)),
                        category=self.CLASSES[cat_idx],
                    )
                    frame.labels.append(label)  # type: ignore

        out_path = osp.join(out_dir, "det.json")
        save(out_path, frames)
