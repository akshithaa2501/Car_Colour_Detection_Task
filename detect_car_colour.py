from ultralytics import YOLO
import cv2
import numpy as np

# Load trained YOLO model
model = YOLO(
    r"C:\Projects\AGE_GENDER_DETECTION-main\runs\detect\runs\car_colour_detection\weights\best.pt"
)

# Load image
image_path = "test.jpg"
image = cv2.imread(image_path)

# Run prediction
results = model.predict(image, conf=0.4)

car_count = 0
person_count = 0

# Classes considered as vehicles
vehicle_classes = ["car", "bus", "truck", "bike", "motor"]

for result in results:

    boxes = result.boxes

    for box in boxes:

        cls = int(box.cls[0])
        label = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])

        # Count people
        if label == "person":
            person_count += 1

        # Count vehicles
        if label in vehicle_classes:

            car_count += 1

            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                continue
                        # Convert ROI to HSV colour space
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # HSV range for blue colour
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])

            # Create blue mask
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Calculate percentage of blue pixels
            blue_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]

            if total_pixels == 0:
                continue

            blue_ratio = blue_pixels / total_pixels

            # Blue vehicle
            if blue_ratio > 0.20:
                colour = "Blue"
                box_colour = (0, 0, 255)      # Red rectangle

            # Other vehicle
            else:
                colour = "Other"
                box_colour = (255, 0, 0)      # Blue rectangle

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), box_colour, 2)

            # Display label
            cv2.putText(
                image,
                f"{label} : {colour}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_colour,
                2
            )
            # Display total counts
cv2.putText(
    image,
    f"Vehicles: {car_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.putText(
    image,
    f"People: {person_count}",
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

# Save output image
cv2.imwrite("output.jpg", image)

# Show result
cv2.imshow("Car Colour Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("=" * 40)
print(f"Total Vehicles Detected : {car_count}")
print(f"Total People Detected   : {person_count}")
print("Output image saved as output.jpg")
print("=" * 40)