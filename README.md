# 🚗 License Plate Recognition (LPR) using YOLO

This project implements a License Plate Recognition system using the **YOLO (You Only Look Once)** object detection framework. It focuses on processing the **CCPD (Chinese City Parking Dataset)** and training a model to accurately detect license plate bounding boxes.

## 🌟 Project Overview

The core of this project involves converting the raw CCPD dataset—where annotations are embedded in filenames—into a standard **YOLO format** (normalized coordinates in `.txt` files) and training a detection model.

### Key Features
* **Data Parsing:** Extracts bounding box coordinates directly from CCPD filenames.
* **Automated Conversion:** Converts raw data into YOLO-compliant format:
    * **Class ID:** `0` (License Plate)
    * **Coordinates:** Normalized `x_center`, `y_center`, `width`, `height`.
* **Dataset Splitting:** Automatically splits data into training and validation sets.
* **Model Configuration:** Uses a custom `ccpd.yaml` file for YOLO training.

## 📂 Dataset Structure

The project transforms the raw CCPD data into the following directory structure required for YOLO:

cpd_yolo_nested/ ├── images/ │ ├── train/ # Training images │ └── val/ # Validation images ├── labels/ │ ├── train/ # YOLO format .txt annotations for training │ └── val/ # YOLO format .txt annotations for validation └── ccpd.yaml # Configuration file defining paths and classes

Training the Model
To train the model using the prepared dataset, run the standard YOLO training command pointing to the configuration file:
yolo task=detect mode=train model=yolov8n.pt data=ccpd.yaml epochs=50 imgsz=640

Results
Input: Raw images from CCPD dataset.

Output: Trained YOLO model capable of detecting license plates in varying conditions (tilt, brightness, weather).
