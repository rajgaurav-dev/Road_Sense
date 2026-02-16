"""
Manual training script for Faster R-CNN using MMDetection 3.x.

This script performs:
- Model building from configuration
- Dataset loading using MMDetection registry
- Manual training loop with SGD optimizer
- Loss computation and backpropagation
- Model checkpoint saving

Author: Your Name
Project: Bosch Assignment
"""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmengine.config import Config
from mmengine.dataset import default_collate
from mmengine.registry import build_from_cfg

from mmdet.registry import MODELS, DATASETS
from mmdet.utils import register_all_modules


def main():
    """
    Main training execution function.

    This function:
    - Loads model configuration
    - Builds detection model
    - Loads training dataset
    - Executes manual training loop
    - Saves trained model weights
    """
    register_all_modules()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg_path = Path(
        r"C:/Bosch_assignment/Road_Sense/Eval_and_Train_question2/"
        r"configs/models/faster_rcnn_r50_fpn.py"
    )

    model_cfg = Config.fromfile(model_cfg_path)
    model_cfg.model.roi_head.bbox_head.num_classes = 10

    model = MODELS.build(model_cfg.model)
    model.to(device)
    model.train()

    dataset_cfg_path = Path(
        r"C:/Bosch_assignment/Road_Sense/Eval_and_Train_question2/"
        r"configs/datasets/bdd10k.py"
    )

    data_cfg = Config.fromfile(dataset_cfg_path)

    train_dataset = build_from_cfg(
        data_cfg.train_dataloader.dataset,
        DATASETS,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.train_dataloader.batch_size,
        shuffle=True,
        num_workers=data_cfg.train_dataloader.num_workers,
        collate_fn=default_collate,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0025,
        momentum=0.9,
        weight_decay=0.0001,
    )

    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for data_batch in pbar:
            data_batch = model.data_preprocessor(
                data_batch,
                training=True,
            )

            losses = model.forward(
                data_batch["inputs"],
                data_batch["data_samples"],
                mode="loss",
            )

            total_loss = sum(
                sum(v) if isinstance(v, list) else v for v in losses.values()
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

    torch.save(model.state_dict(), "faster_rcnn_bdd_manual.pth")
    print("Training completed successfully")


if __name__ == "__main__":
    main()
