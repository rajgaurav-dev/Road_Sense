"""
Inference script for Faster R-CNN model using MMDetection 3.x.

This script supports:
- Single image inference
- Folder inference
- Full dataset inference

Features:
- Custom model loading
- Custom test pipeline loading
- Bounding box visualization
- Configurable score threshold

Author: Your Name
Project: Bosch Assignment
"""

import argparse
import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmengine.config import Config
from mmengine.dataset import Compose, default_collate
from mmengine.registry import build_from_cfg
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS, DATASETS
from mmdet.utils import register_all_modules


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing input path,
        dataset flag, model weights path, and score threshold.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="Path to an image file or a folder containing images",
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Run inference on full validation dataset",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshold for visualization",
    )
    return parser.parse_args()


def build_model(device, weights_path=None):
    """
    Build Faster R-CNN model from configuration.

    Args:
        device (str): Device type ('cuda' or 'cpu').
        weights_path (str, optional): Path to trained weights.

    Returns:
        torch.nn.Module: Initialized model in evaluation mode.
    """
    model_cfg_path = Path(
        r"C:/Bosch_assignment/Road_Sense/Eval_and_Train_question2/"
        r"configs/models/faster_rcnn_r50_fpn.py"
    )

    model_cfg = Config.fromfile(model_cfg_path)
    model_cfg.model.roi_head.bbox_head.num_classes = 10

    model = MODELS.build(model_cfg.model)

    if weights_path and os.path.exists(weights_path):
        load_checkpoint(model, weights_path, map_location="cpu")
        print("Loaded trained weights")
    else:
        print("Using random weights")

    model.to(device)
    model.eval()

    return model


def build_pipeline():
    """
    Build test pipeline from dataset configuration.

    Returns:
        tuple:
            Compose: Composed test pipeline.
            Config: Dataset configuration.
    """
    dataset_cfg_path = Path(
        r"C:/Bosch_assignment/Road_Sense/Eval_and_Train_question2/"
        r"configs/datasets/bdd10k.py"
    )

    data_cfg = Config.fromfile(dataset_cfg_path)
    test_pipeline_cfg = data_cfg.val_dataloader.dataset.pipeline
    pipeline = Compose(test_pipeline_cfg)

    return pipeline, data_cfg


def visualize(img, result, score_thr=0.3, save_path=None):
    """
    Visualize detection results on an image.

    Args:
        img (np.ndarray): Input image.
        result: Model prediction result.
        score_thr (float): Score threshold for filtering boxes.
        save_path (str, optional): Path to save visualization.

    Returns:
        np.ndarray: Image with drawn bounding boxes.
    """
    boxes = result.pred_instances.bboxes.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()

    classes = [
        "person",
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

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thr:
            continue

        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_name = classes[label] if label < len(classes) else str(label)
        text = f"{label_name}: {score:.2f}"

        cv2.putText(
            img,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    if save_path:
        cv2.imwrite(save_path, img)

    return img


def infer_single_image(model, pipeline, image_path, score_thr):
    """
    Run inference on a single image.

    Args:
        model: Detection model.
        pipeline: Test pipeline.
        image_path (str or Path): Image file path.
        score_thr (float): Score threshold.
    """
    img = cv2.imread(str(image_path))

    data = {"img_path": str(image_path)}
    data = pipeline(data)

    batch = {
        "inputs": [data["inputs"]],
        "data_samples": [data["data_samples"]],
    }

    batch = model.data_preprocessor(batch, training=False)

    with torch.no_grad():
        outputs = model.forward(
            batch["inputs"],
            batch["data_samples"],
            mode="predict",
        )

    result = outputs[0]
    img_vis = visualize(img, result, score_thr)

    cv2.imshow("Detection", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Boxes:", result.pred_instances.bboxes.shape)


def infer_folder(model, pipeline, folder_path, score_thr):
    """
    Run inference on all images inside a folder.

    Args:
        model: Detection model.
        pipeline: Test pipeline.
        folder_path (str): Folder containing images.
        score_thr (float): Score threshold.
    """
    images = list(Path(folder_path).glob("*.*"))
    print(f"Found {len(images)} images")

    for img_path in images:
        print(f"Processing: {img_path}")
        infer_single_image(model, pipeline, img_path, score_thr)


def infer_dataset(model, data_cfg):
    """
    Run inference on entire validation dataset.

    Args:
        model: Detection model.
        data_cfg: Dataset configuration.
    """
    test_dataset = build_from_cfg(
        data_cfg.val_dataloader.dataset,
        DATASETS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=default_collate,
    )

    print(f"Running dataset inference on {len(test_dataset)} samples")

    with torch.no_grad():
        for data_batch in tqdm(test_loader):
            data_batch = model.data_preprocessor(
                data_batch,
                training=False,
            )

            outputs = model.forward(
                data_batch["inputs"],
                data_batch["data_samples"],
                mode="predict",
            )

            result = outputs[0]
            print("Boxes:", result.pred_instances.bboxes.shape)
            break


def main():
    """
    Main execution entry point.
    """
    args = parse_args()
    register_all_modules()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(device, args.weights)
    pipeline, data_cfg = build_pipeline()

    if args.input:
        if os.path.isfile(args.input):
            infer_single_image(
                model,
                pipeline,
                args.input,
                args.score_thr,
            )
        elif os.path.isdir(args.input):
            infer_folder(
                model,
                pipeline,
                args.input,
                args.score_thr,
            )
        else:
            print("Invalid input path")

    elif args.dataset:
        infer_dataset(model, data_cfg)

    else:
        print("Please provide --input or --dataset")


if __name__ == "__main__":
    main()
