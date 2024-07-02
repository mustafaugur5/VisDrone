# Yolo to Coco script
import os
import json
import glob
from PIL import Image
from tqdm import tqdm

def convert_yolo_to_coco(images_folder, labels_folder, output_json_path):
    # Initialize COCO structure
    coco = {
        "info": {
            "description": "Converted YOLO Dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "2024-07-01"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create a mapping for category ids
    category_mapping = {}
    category_id = 1

    # Process each label file
    annotation_id = 1
    image_id = 1
    for label_path in tqdm(glob.glob(os.path.join(labels_folder, "*.txt"))):
        with open(label_path, "r") as file:
            label_data = file.readlines()

        image_filename = os.path.basename(label_path).replace(".txt", ".jpg")
        image_path = os.path.join(images_folder, image_filename)

        # Load the image to get its dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        # Add image info to COCO
        coco["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        # Process each line in the label file
        for line in label_data:
            parts = line.strip().split()
            category = parts[0]
            bbox_yolo = list(map(float, parts[1:]))

            # Convert YOLO format to COCO format
            x_center, y_center, bbox_width, bbox_height = bbox_yolo
            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            bbox_width *= width
            bbox_height *= height
            bbox_coco = [x_min, y_min, bbox_width, bbox_height]

            # Ensure the category is in the categories list
            if category not in category_mapping:
                category_mapping[category] = category_id
                coco["categories"].append({
                    "id": category_id,
                    "name": category,
                    "supercategory": "none"
                })
                category_id += 1

            # Add annotation info to COCO
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_mapping[category],
                "bbox": bbox_coco,
                "area": bbox_width * bbox_height,
                "segmentation": [],
                "iscrowd": 0
            })

            annotation_id += 1

        image_id += 1

    # Save COCO JSON file
    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=4)

