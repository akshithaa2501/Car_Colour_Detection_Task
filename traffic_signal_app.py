import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

# Load trained model
model = YOLO(
    r"C:\Projects\AGE_GENDER_DETECTION-main\runs\detect\runs\car_colour_detection\weights\best.pt"
)

st.set_page_config(page_title="Car Colour Detection", layout="wide")

st.title("🚗 Car Colour Detection System")

st.write("Upload a traffic image to detect vehicles and people.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    temp = tempfile.NamedTemporaryFile(delete=False)

    temp.write(uploaded_file.read())

    temp.close()

    image = cv2.imread(temp.name)

    results = model.predict(
        image,
        conf=0.15,
        iou=0.45,
        imgsz=640,
        augment=True,
        verbose=False,
        classes=[0, 2]
    )

    vehicle_count = 0
    person_count = 0

    vehicle_classes = [
        "car"
    ]
    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]
            print(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            # Person detection
            if label == "person":

                person_count += 1

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Vehicle detection
            elif label in vehicle_classes:

                vehicle_count += 1

                roi = image[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                lower_blue = np.array([80, 40, 40])
                upper_blue = np.array([140, 255, 255])

                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                blue_pixels = cv2.countNonZero(mask)

                total_pixels = roi.shape[0] * roi.shape[1]

                if total_pixels == 0:
                    continue

                blue_ratio = blue_pixels / total_pixels

                if blue_ratio > 0.08:
                    colour_name = "Blue"
                    rectangle_colour = (0, 0, 255)      # Red rectangle
                else:
                    colour_name = "Other"
                    rectangle_colour = (255, 0, 0)      # Blue rectangle

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    rectangle_colour,
                    2
                )

                cv2.putText(
                    image,
                    f"{label} : {colour_name}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    rectangle_colour,
                    2
                )
    st.subheader("Detection Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🚗 Cars Detected", vehicle_count)

    with col2:
        st.metric("👤 People Detected", person_count)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Processed Image")

    st.image(
        image_rgb,
        channels="RGB",
        use_container_width=True
    )

    output_path = "output.jpg"

    cv2.imwrite(output_path, image)

    with open(output_path, "rb") as file:

        st.download_button(
            label="📥 Download Processed Image",
            data=file,
            file_name="output.jpg",
            mime="image/jpeg"
        )
