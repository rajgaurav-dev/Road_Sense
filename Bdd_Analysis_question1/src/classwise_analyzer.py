"""
Class-wise deep analysis for BDD100K detection dataset.

This module performs detailed analysis for a specific object class,
including:
- Number of images containing the class
- Total object count
- Average objects per image
- Truncated and occluded percentages
- Weather distribution
- Time-of-day distribution
- Traffic light color distribution

Usage
-----
python classwise_analysis.py --json path/to/train.json --class_name car

Save results:
python classwise_analysis.py --json train.json --class_name car --output result.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict

from bddparser import BDDParser


class ClassWiseAnalyzer(BDDParser):
    """
    Performs deep analysis for a specific class.

    Inherits dataset parsing functionality from BDDParser.
    """

    def __init__(self, json_path: str) -> None:
        super().__init__(json_path)

    def analyze(self, class_name: str) -> Dict:
        """
        Perform detailed analysis for a given class.

        Parameters
        ----------
        class_name : str
            Target class name.

        Returns
        -------
        Dict
            Dictionary containing class statistics.
        """

        image_count = 0
        object_count = 0
        truncated_count = 0
        occluded_count = 0

        weather_counter = Counter()
        timeofday_counter = Counter()
        traffic_color_counter = Counter()

        for img_data in self.images.values():

            annotations = img_data["annotations"]
            attributes = img_data["attributes"]

            class_objects = [
                ann
                for ann in annotations
                if ann["category"] == class_name
            ]

            if not class_objects:
                continue

            image_count += 1
            object_count += len(class_objects)

            weather = attributes.get("weather", "unknown")
            timeofday = attributes.get("timeofday", "unknown")

            weather_counter[weather] += 1
            timeofday_counter[timeofday] += 1

            for ann in class_objects:
                if ann.get("truncated", False):
                    truncated_count += 1

                if ann.get("occluded", False):
                    occluded_count += 1

                color = ann.get("traffic_light_color", "none")
                traffic_color_counter[color] += 1

        avg_objects_per_image = (
            object_count / image_count if image_count > 0 else 0
        )

        truncated_pct = (
            (truncated_count / object_count) * 100
            if object_count > 0
            else 0
        )

        occluded_pct = (
            (occluded_count / object_count) * 100
            if object_count > 0
            else 0
        )

        return {
            "class_name": class_name,
            "num_images": image_count,
            "num_objects": object_count,
            "avg_objects_per_image": round(
                avg_objects_per_image, 2
            ),
            "truncated_pct": round(truncated_pct, 2),
            "occluded_pct": round(occluded_pct, 2),
            "weather_distribution": dict(weather_counter),
            "timeofday_distribution": dict(timeofday_counter),
            "traffic_light_color_distribution": dict(
                traffic_color_counter
            ),
        }


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """

    parser = argparse.ArgumentParser(
        description="Class-wise analysis for BDD100K"
    )

    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to BDD JSON file",
    )

    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        help="Class name to analyze",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main execution function.
    """

    args = parse_arguments()

    json_path = Path(args.json)

    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {json_path}"
        )

    analyzer = ClassWiseAnalyzer(str(json_path))

    result = analyzer.analyze(args.class_name)

    print("\nClass-wise Statistics:")
    print(json.dumps(result, indent=4))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4)

        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
