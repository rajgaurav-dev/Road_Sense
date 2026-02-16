"""
Plotting utilities for BDD100K statistics.

This module:
- Imports BDDParser
- Computes selected statistics
- Plots results using a generic plotting function

Usage
-----
Plot object distribution:
    python plot_statistics.py --json path/to/train.json --stat object

Plot images per class:
    python plot_statistics.py --json path/to/train.json --stat image

Plot with percentage:
    python plot_statistics.py --json path/to/train.json --stat object --percentage

Save output:
    python plot_statistics.py --json path/to/train.json --stat object --output plot.png
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt

from bddparser import BDDParser


def plot_statistics(
    data: Dict,
    title: str = "Distribution",
    xlabel: str = "Category",
    ylabel: str = "Count",
    save_path: Optional[str] = None,
    sort: bool = True,
    percentage: bool = False,
    log_scale: bool = False,
) -> None:
    """
    Generic plotting function for dictionary-based statistics.

    Parameters
    ----------
    data : Dict
        Dictionary containing labels and values.
    title : str
        Plot title.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    save_path : Optional[str]
        Path to save the plot.
    sort : bool
        Whether to sort values in descending order.
    percentage : bool
        Convert counts to percentage.
    log_scale : bool
        Use logarithmic scale for y-axis.
    """

    if not data:
        print("No data available for plotting.")
        return

    data_dict = dict(data)

    if percentage:
        total = sum(data_dict.values())
        data_dict = {key: (value / total) * 100 for key, value in data_dict.items()}
        ylabel = "Percentage (%)"

    if sort:
        data_dict = dict(
            sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
        )

    labels = list(data_dict.keys())
    values = list(data_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    if log_scale:
        plt.yscale("log")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{round(height, 2)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """

    parser = argparse.ArgumentParser(description="BDD100K Plotting Utility")

    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to BDD JSON file",
    )

    parser.add_argument(
        "--stat",
        type=str,
        required=True,
        choices=["object", "image"],
        help="Statistic type to plot",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save plot",
    )

    parser.add_argument(
        "--percentage",
        action="store_true",
        help="Plot as percentage",
    )

    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Use log scale",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main execution function.
    """

    args = parse_arguments()

    json_path = Path(args.json)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    dataset = BDDParser(json_path)

    if args.stat == "object":
        data = dataset.class_distribution()
        title = "Object Count Per Class"

    elif args.stat == "image":
        data = dataset.images_per_class()
        title = "Images Per Class"

    else:
        raise ValueError("Invalid statistic type.")

    print("\nStatistics:")
    print(data)

    plot_statistics(
        data=data,
        title=title,
        save_path=args.output,
        percentage=args.percentage,
        log_scale=args.log_scale,
    )


if __name__ == "__main__":
    main()
