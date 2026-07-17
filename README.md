# 🚗 Car Colour Detection System
# Project Overview

- The Car Colour Detection System is a computer vision project developed using Python, YOLOv8, OpenCV, and Streamlit.

- The system detects:

Cars
People

- It also identifies whether a detected car is Blue or Other using HSV colour detection.

- The application provides a simple GUI where users can upload an image and view the detection results.

# Objective

The objective of this project is to:

- Detect cars in a traffic image.
- Detect people in the same image.
- Count the total number of cars.
- Count the total number of people.
- Identify whether each detected car is Blue or Other.
- Display the processed image with bounding boxes.
- Allow users to download the processed image.

# Software Requirements
Python 3.10+
Streamlit
Ultralytics YOLOv8
OpenCV
NumPy

# Install the required libraries:

pip install ultralytics
pip install streamlit
pip install opencv-python
pip install numpy

# Dataset Used

- This project uses the BDD100K (Berkeley DeepDrive 100K) dataset.

- BDD100K contains traffic images with object annotations for:

Person
Car
Bus
Truck
Bike
Motor
Traffic Light
Traffic Sign

- For this project:

500 training images were selected.
100 validation images were selected.

- The dataset was prepared in YOLO format before training.

# Model Training

The pre-trained YOLOv8 Nano (yolov8n.pt) model was used.

Training parameters:

Model : YOLOv8 Nano
Epochs : 10
Image Size : 416
Batch Size : 4
Device : CPU

- After training, the best model was saved as:

best.pt

- The Streamlit application uses this trained model for prediction.

Technologies Used
Python
YOLOv8
OpenCV
NumPy
Streamlit

# Project Workflow
# Step 1

Collect the BDD100K dataset.

# Step 2

Select training and validation images.

# Step 3

Convert the dataset into YOLO format.

# Step 4

Create the dataset configuration file (small_data.yaml).

# Step 5

Train the YOLOv8 model.

# Step 6

Save the trained model (best.pt).

# Step 7

Develop a Streamlit GUI.

# Step 8

Upload a traffic image.

# Step 9

The trained YOLO model detects cars and people.

# Step 10

Each detected car is cropped.

# Step 11

The cropped image is converted into HSV colour space.

# Step 12

HSV colour thresholding is applied.

If blue pixels are detected:

Blue Car

Otherwise:

Other Car

# Step 13

Bounding boxes are drawn.

🟩 Green → Person
🟥 Red → Blue Car
🟦 Blue → Other Car

# Step 14

The application displays:

Number of Cars
Number of People
Processed Image

# Step 15

Users can download the processed image.

# Folder Structure
Car_Colour_Detection_Task
- traffic_signal_app.py        # Streamlit application
- car_colour_train.py          # YOLO training script
- best.pt                      # Trained YOLO model
- Yolov8n.pt                   # Pre-trained YOLOv8 model
- test.jpg                     # Sample input image
- README.md                    # Project documentation
- runs                        # Training output (optional)
    - detect/
        - car_colour_detection/
             - weights/
                   - best.pt
                   - last.pt

- output.jpg                   # Generated output image

Note: If you keep best.pt in the runs/.../weights/ folder instead of the project root, update the path in traffic_signal_app.py accordingly.

# Conclusion

The Car Colour Detection System combines YOLOv8, OpenCV, and Streamlit to detect cars and people in traffic images. After detecting cars, HSV colour analysis is used to classify each car as Blue or Other. The application provides a simple interface for uploading images, viewing detection results, and downloading the processed output, making it suitable for traffic monitoring and educational computer vision projects.
