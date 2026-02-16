import json
import os
import cv2
import argparse

BDD_CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign"
]

category_mapping = {name: i + 1 for i, name in enumerate(BDD_CLASSES)}


def convert_bdd_to_coco(bdd_json_path, image_dir, output_json):

    with open(bdd_json_path, 'r') as f:
        bdd_data = json.load(f)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": name}
            for i, name in enumerate(BDD_CLASSES)
        ]
    }

    ann_id = 1

    for img_id, item in enumerate(bdd_data, start=1):

        file_name = item["name"]
        img_path = os.path.join(image_dir, file_name)

        if not os.path.exists(img_path):
            print(f"⚠ Warning: Image not found {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ Warning: Unable to read image {img_path}")
            continue

        h, w = img.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": w,
            "height": h
        })

        for label in item.get("labels", []):

            if "box2d" not in label:
                continue

            x1 = label["box2d"]["x1"]
            y1 = label["box2d"]["y1"]
            x2 = label["box2d"]["x2"]
            y2 = label["box2d"]["y2"]

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_mapping.get(label["category"], 0),
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0
            })

            ann_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f)

    print(f"✅ Conversion completed. Saved to {output_json}")


# --------------------------------------------------
# Main function with argument parsing
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert BDD100K format to COCO format"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to BDD JSON file"
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output COCO JSON file path"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    convert_bdd_to_coco(
        bdd_json_path=args.input,
        image_dir=args.image_dir,
        output_json=args.output
    )


if __name__ == "__main__":
    main()

