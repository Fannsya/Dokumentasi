
import cv2
import numpy as np

# Load pre-trained COCO model and configuration
model_config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
model_weights = 'frozen_inference_graph.pb'

# Load class labels for COCO dataset
class_labels = []
with open('coco_labels.txt', 'r') as file:
    class_labels = file.read().strip().split('\n')

# Initialize the DNN model
net = cv2.dnn_DetectionModel(model_weights, model_config)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start video capture from the Raspberry Pi camera
camera = cv2.VideoCapture(0)  # Change this if using an external camera

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Threshold for confidence
CONFIDENCE_THRESHOLD = 0.5

print("Press 'q' to quit.")
while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read from the camera.")
        break

    # Detect objects in the frame
    class_ids, confidences, boxes = net.detect(frame, confThreshold=CONFIDENCE_THRESHOLD)

    for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
        # Get class name and draw bounding box
        label = f"{class_labels[class_id]}: {confidence:.2f}"
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
 