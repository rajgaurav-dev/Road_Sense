"""
Gradio dashboard for interactive BDD100K detection dataset analysis.

This dashboard supports:
- Object-level distribution with filters
- Image-level distribution with filters
- Filtered image visualization with bounding boxes

Usage
-----
python dashboard.py
"""

import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from bddparser import BDDParser


def create_bar_plot(data: Dict, title: str) -> plt.Figure:
    """
    Create a bar plot from dictionary data.

    Parameters
    ----------
    data : Dict
        Dictionary containing label-count pairs.
    title : str
        Title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
    """

    plt.figure(figsize=(8, 5))

    if not data:
        plt.title("No data available for selected filters")
        return plt.gcf()

    labels = list(data.keys())
    values = list(data.values())

    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()

    return plt.gcf()


def get_object_stats(
    json_path: str,
    selected_classes: List[str],
    weather_filter: str,
    time_filter: str,
    truncated_filter: str,
    occluded_filter: str,
) -> plt.Figure:
    """
    Compute filtered object-level statistics.
    """

    dataset = BDDParser(json_path)
    counter = Counter()

    for img_data in dataset.images.values():

        weather = img_data["attributes"].get("weather", "unknown")
        timeofday = img_data["attributes"].get("timeofday", "unknown")

        if weather_filter and weather != weather_filter:
            continue

        if time_filter and timeofday != time_filter:
            continue

        for ann in img_data["annotations"]:

            cls = ann["category"]

            if selected_classes and cls not in selected_classes:
                continue

            is_truncated = ann.get("truncated", False)
            is_occluded = ann.get("occluded", False)

            if truncated_filter == "Only Truncated" and not is_truncated:
                continue
            if truncated_filter == "Only Non-Truncated" and is_truncated:
                continue

            if occluded_filter == "Only Occluded" and not is_occluded:
                continue
            if occluded_filter == "Only Non-Occluded" and is_occluded:
                continue

            counter[cls] += 1

    return create_bar_plot(counter, "Filtered Object-Level Distribution")


def get_image_stats(
    json_path: str,
    selected_classes: List[str],
    weather_filter: str,
    time_filter: str,
) -> plt.Figure:
    """
    Compute filtered image-level statistics.
    """

    dataset = BDDParser(json_path)
    counter = Counter()

    for img_data in dataset.images.values():

        weather = img_data["attributes"].get("weather", "unknown")
        timeofday = img_data["attributes"].get("timeofday", "unknown")

        if weather_filter and weather != weather_filter:
            continue

        if time_filter and timeofday != time_filter:
            continue

        classes_in_image = {ann["category"] for ann in img_data["annotations"]}

        for cls in classes_in_image:
            if selected_classes and cls not in selected_classes:
                continue
            counter[cls] += 1

    return create_bar_plot(counter, "Filtered Image-Level Distribution")


def get_filtered_images(
    json_path: str,
    image_root: str,
    selected_classes: List[str],
    weather_filter: str,
    time_filter: str,
    truncated_filter: str,
    occluded_filter: str,
) -> Tuple[List[np.ndarray], str]:
    """
    Retrieve filtered images with bounding boxes drawn.

    Returns
    -------
    Tuple[List[np.ndarray], str]
        List of annotated images and status message.
    """

    dataset = BDDParser(json_path)
    matching_images = []

    for img_name, img_data in dataset.images.items():

        weather = img_data["attributes"].get("weather", "unknown")
        timeofday = img_data["attributes"].get("timeofday", "unknown")

        if weather_filter and weather != weather_filter:
            continue

        if time_filter and timeofday != time_filter:
            continue

        filtered_annotations = []

        for ann in img_data["annotations"]:

            cls = ann["category"]

            if selected_classes and cls not in selected_classes:
                continue

            is_truncated = ann.get("truncated", False)
            is_occluded = ann.get("occluded", False)

            if truncated_filter == "Only Truncated" and not is_truncated:
                continue
            if truncated_filter == "Only Non-Truncated" and is_truncated:
                continue

            if occluded_filter == "Only Occluded" and not is_occluded:
                continue
            if occluded_filter == "Only Non-Occluded" and is_occluded:
                continue

            filtered_annotations.append(ann)

        if filtered_annotations:
            matching_images.append((img_name, filtered_annotations))

    if not matching_images:
        return [], "No images matched selected filters."

    sampled = random.sample(matching_images, min(5, len(matching_images)))

    output_images = []

    for img_name, annotations in sampled:

        image_path = os.path.join(image_root, img_name)

        if not Path(image_path).exists():
            continue

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for ann in annotations:
            x, y, w, h = ann["bbox"]

            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                img,
                ann["category"],
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        output_images.append(img)

    message = (
        f"Showing {len(output_images)} sampled images "
        f"(matched total: {len(matching_images)})"
    )

    return output_images, message


def build_dashboard() -> gr.Blocks:
    """
    Build and return Gradio dashboard.
    """

    with gr.Blocks() as demo:

        gr.Markdown("## BDD100K Detection Dataset Interactive Dashboard")

        json_path = gr.Textbox(label="BDD JSON Path")
        image_root = gr.Textbox(label="Image Folder Path")

        selected_classes = gr.CheckboxGroup(
            choices=[
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
            ],
            label="Filter by Classes",
        )

        weather_filter = gr.Dropdown(
            choices=["", "clear", "rainy", "overcast", "snowy", "foggy"],
            label="Weather Filter",
            value="",
        )

        time_filter = gr.Dropdown(
            choices=["", "daytime", "night", "dawn/dusk"],
            label="Time Filter",
            value="",
        )

        truncated_filter = gr.Dropdown(
            choices=["All", "Only Truncated", "Only Non-Truncated"],
            label="Truncated Filter",
            value="All",
        )

        occluded_filter = gr.Dropdown(
            choices=["All", "Only Occluded", "Only Non-Occluded"],
            label="Occluded Filter",
            value="All",
        )

        with gr.Row():
            object_button = gr.Button("Plot Object-Level Stats")
            image_button = gr.Button("Plot Image-Level Stats")
            sample_button = gr.Button("Show Filtered Image Samples")

        object_plot = gr.Plot(label="Object-Level Statistics")
        image_plot = gr.Plot(label="Image-Level Statistics")
        image_gallery = gr.Gallery(label="Filtered Image Samples")
        status_message = gr.Markdown()

        object_button.click(
            get_object_stats,
            inputs=[
                json_path,
                selected_classes,
                weather_filter,
                time_filter,
                truncated_filter,
                occluded_filter,
            ],
            outputs=object_plot,
        )

        image_button.click(
            get_image_stats,
            inputs=[
                json_path,
                selected_classes,
                weather_filter,
                time_filter,
            ],
            outputs=image_plot,
        )

        sample_button.click(
            get_filtered_images,
            inputs=[
                json_path,
                image_root,
                selected_classes,
                weather_filter,
                time_filter,
                truncated_filter,
                occluded_filter,
            ],
            outputs=[image_gallery, status_message],
        )

    return demo


def main() -> None:
    """
    Launch Gradio dashboard.
    """

    dashboard = build_dashboard()
    dashboard.launch()


if __name__ == "__main__":
    main()
