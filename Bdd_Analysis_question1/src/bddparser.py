"""
BDD100K Dataset Parser and Analyzer.

This module provides:
- Full dataset analysis
- Per-class distribution
- Image-wise analysis
- Folder-wise analysis

Usage (CLI):
------------
Analyze full dataset:
    python bdd_parser.py --json path/to/val.json

Analyze single image:
    python bdd_parser.py --json path/to/val.json --image path/to/img.jpg

Analyze folder:
    python bdd_parser.py --json path/to/val.json --folder path/to/folder
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict


class BDDParser:
    """
    Parser for BDD100K Scalabel JSON format.

    Parameters
    ----------
    json_path : str
        Path to validation JSON file.
    """

    def __init__(self, json_path: str) -> None:
        self.json_path = Path(json_path)
        self.images: Dict = {}
        self._load_json()

    def _load_json(self) -> None:
        """
        Load JSON file and convert into structured dictionary.
        """

        with open(self.json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for item in data:
            image_name = item["name"]

            self.images[image_name] = {
                "attributes": item.get("attributes", {}),
                "annotations": [],
            }

            for label in item.get("labels", []):
                if "box2d" not in label:
                    continue

                box = label["box2d"]

                x1, y1 = box["x1"], box["y1"]
                x2, y2 = box["x2"], box["y2"]

                width = x2 - x1
                height = y2 - y1
                area = width * height

                self.images[image_name]["annotations"].append(
                    {
                        "category": label["category"],
                        "bbox": [x1, y1, width, height],
                        "area": area,
                        "occluded": label["attributes"].get("occluded", False),
                        "truncated": label["attributes"].get("truncated", False),
                        "traffic_light_color": label["attributes"].get(
                            "trafficLightColor", "none"
                        ),
                    }
                )

        print(f"Loaded {len(self.images)} images successfully.")

    def class_distribution(self) -> Counter:
        """
        Compute object count per class.

        Returns
        -------
        Counter
            Object counts per category.
        """
        counter = Counter()

        for img_data in self.images.values():
            for ann in img_data["annotations"]:
                counter[ann["category"]] += 1

        return counter

    def images_per_class(self) -> Counter:
        """
        Count number of images per class.

        Returns
        -------
        Counter
            Image counts per category.
        """
        counter = Counter()

        for img_data in self.images.values():
            unique_classes = {ann["category"] for ann in img_data["annotations"]}

            for cls in unique_classes:
                counter[cls] += 1

        return counter

    def total_images(self) -> int:
        """Return total number of images."""
        return len(self.images)


class ImageSubsetAnalyzer:
    """
    Analyze specific image or folder from BDD JSON.
    """

    def __init__(self, json_path: str) -> None:
        self.json_path = json_path

        with open(self.json_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def analyze_single_image(self, image_path: str) -> Counter:
        """
        Analyze one image and return object count per class.

        Parameters
        ----------
        image_path : str
            Full image path or image name.

        Returns
        -------
        Counter
            Object counts per category.
        """

        image_name = Path(image_path).name

        for item in self.data:
            if item["name"] == image_name:
                counter = Counter()

                for label in item.get("labels", []):
                    if "box2d" not in label:
                        continue
                    counter[label["category"]] += 1

                return counter

        return Counter()

    def analyze_folder(self, folder_path: str) -> Counter:
        """
        Analyze all images inside a folder.

        Parameters
        ----------
        folder_path : str

        Returns
        -------
        Counter
            Combined object counts.
        """

        counter = Counter()

        image_files = [
            file
            for file in os.listdir(folder_path)
            if file.lower().endswith((".jpg", ".png"))
        ]

        for file in image_files:
            image_path = os.path.join(folder_path, file)
            counter.update(self.analyze_single_image(image_path))

        return counter


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="BDD100K Dataset Analyzer")

    parser.add_argument(
        "--json",
        required=True,
        help="Path to BDD validation JSON file",
    )

    parser.add_argument(
        "--image",
        help="Path to a single image for analysis",
    )

    parser.add_argument(
        "--folder",
        help="Path to folder containing images",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main execution entry.
    """

    args = parse_arguments()

    parser = BDDParser(args.json)

    print("\nClass Distribution:")
    print(parser.class_distribution())

    print("\nImages Per Class:")
    print(parser.images_per_class())

    if args.image:
        subset = ImageSubsetAnalyzer(args.json)
        print("\nSingle Image Analysis:")
        print(subset.analyze_single_image(args.image))

    if args.folder:
        subset = ImageSubsetAnalyzer(args.json)
        print("\nFolder Analysis:")
        print(subset.analyze_folder(args.folder))


if __name__ == "__main__":
    main()
