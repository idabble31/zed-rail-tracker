import cv2
import numpy as np
import pyzed.sl as sl
import torch  # For YOLOv5

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/your/yolov5_model.pt')  # Update with your model path

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

err = zed.open(init_params)

cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve the image from the ZED camera
        img = sl.Mat()
        zed.retrieve_image(img, sl.VIEW.LEFT)
        img_cv = img.get_data()

        # Convert the image to RGB (YOLOv5 expects RGB format)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # Run YOLOv5 inference
        results = model(img_rgb)

        # Parse YOLOv5 results
        for result in results.xyxy[0]:  # Iterate through detections
            x1, y1, x2, y2, conf, cls = result[:6]
            label = f"{model.names[int(cls)]} ({conf:.2f})"

            # Draw bounding box
            cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw label
            cv2.putText(img_cv, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the image
        cv2.imshow("ZED", img_cv)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
zed.close()
cv2.destroyAllWindows()