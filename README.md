# DartsDetector: Automated Darts Detection and Scoring System

DartsDetector is an advanced video analysis tool designed to detect darts, their components (tip, barrel, shaft, flight), and the dartboard in real-time. Leveraging YOLO models, this system can analyze darts games with minimal human intervention, providing automatic dart detection, score assignment, and bullseye detection, all integrated into a seamless workflow.

🎥 **Demo Video:** [Watch here](https://drive.google.com/file/d/1PPVUS4mfUHC9CDhEEpUWlUTZnlBnmQxh/view?usp=drive_link)

## 📁 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training the Models](#training-the-models)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

## ✨ Features

- **Dart Detection:** Detects the full dart (tip, barrel, shaft, flight) using the `tracker_darts` model.
- **Bullseye Detection:** Specialized `bull_tracker` model detects the bullseye and interpolates its position across frames.
- **Handling Partial Dart Detection:** Even when parts of a dart (like the shaft or barrel) are detected but the full dart is not, the system builds a virtual dart to estimate its position.
- **Score Assignment:** Automatically calculates the score of the last dart thrown based on its position and the bullseye.
- **Interpolation of Darts and Bullseye:** Positions of darts and the bullseye are interpolated between their first and last detection, ensuring accurate tracking even with partial detection.

## 🔧 Prerequisites

- Python 3.8+
- Required Python libraries: Listed in `requirements.txt`

## ⚙️ Installation

1. **Set up your Python environment:**
   - Create a virtual environment (e.g., `venv` or `conda`).
   - Install the necessary dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```

2. **Download the Pretrained Models:**
   - `tracker_darts` model (for detecting darts and their parts)
   - `bull_tracker` model (for detecting the bullseye)
   - Place these model files into the `models/` folder.

## 🎓 Training the Models

Training the models is straightforward using the provided `darts_training.ipynb` Jupyter notebook. The notebook guides you through training the two models:

- **Dart Detection Model (`tracker_darts`)**: Trains a model to detect darts and their components (tip, barrel, shaft, flight) and the dartboard.
- **Bullseye Detection Model (`bull_tracker`)**: Trains a model to detect the bullseye on the dartboard.

Both models are based on YOLO architecture, ensuring fast and efficient detection.
Once trained, place the resulting `.pt` files in the `models/` folder for use.

## 🚀 Usage

You can run the analysis pipeline using Python:

1. Run the main script with your chosen video:
   ```bash
   python main.py path_to_input_video.mp4 --output_video output_videos/output_result.avi
The system will automatically detect darts and the bullseye, assign scores based on dart positions, and handle partial dart detections by building virtual darts where necessary.

2. The system will automatically detect darts and the bullseye, assign scores based on dart positions, and handle partial dart detections by building virtual darts where necessary.

## 🏰 Project Structure

- `main.py`: Orchestrates the entire pipeline, handling video input, detection/tracking, score assignment, and output generation.
  
- `scores_assigner/`: Manages score assignment for each detected dart and overlays the score on the video frame.
  
- `tracker/tracker_bull/`: Contains all functions that utilize the `bull_tracker` model, including bullseye detection, drawing, interpolation, and other utility functions.
  
- `tracker/tracker_darts_/`: Contains all functions that utilize the `tracker_darts` model to detect darts and their components, handle partial detections, interpolate their positions, and draw bounding boxes around them.
  
- `training/`: Contains the `darts_training.ipynb` Jupyter notebook where the models are trained.
  
- `utils/`: Contains utility functions used throughout the project, such as video preprocessing, data augmentation, and other helper functions that assist with model training and video analysis.

## 🔮 Future Work

- **Handling Perspective and Camera Movement**: Currently, due to camera zooms and varying perspectives, stable points for angle calculations and sector/score assignment are difficult to detect. An improved method to account for camera movement and perspective changes is needed to enhance accuracy.

- **Refined Dart Detection**: We plan to fine-tune the system to handle more challenging dart trajectories and further improve the accuracy of dart detection, especially in low-light or high-speed scenarios.


