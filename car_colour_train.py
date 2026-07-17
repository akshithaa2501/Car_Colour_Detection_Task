from ultralytics import YOLO
import os

def main():
    model = YOLO("yolov8n.pt")

    yaml_path = os.path.abspath("small_data.yaml")
    print("Using YAML:", yaml_path)

    model.train(
        data=yaml_path,
        epochs=10,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=2,
        project="runs",
        name="car_colour_detection",
        exist_ok=True
    )

if __name__ == "__main__":
    main()