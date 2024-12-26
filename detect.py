import cv2
import numpy as np
from detection_utils import load_yolo_model, draw_bounding_boxes

# Load YOLO model
config_path = "D:\\Mini project\\Dehazing-and-Human-Detection-Project\\Human_Detection\\yolov3.cfg"
weights_path = "D:\\Mini project\\Dehazing-and-Human-Detection-Project\\Human_Detection\\yolov3.weights"
labels = ["person", "cat", "dog"]  

net, output_layers = load_yolo_model(config_path, weights_path)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the webcam, or specify video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Modify here: Check if class_id is within the bounds of labels
            if confidence > 0.5 and class_id < len(labels) and labels[class_id] in ["person", "cat", "dog"]:
                center_x, center_y = int(detection[0] * frame.shape[1]), int(detection[1] * frame.shape[0])
                w, h = int(detection[2] * frame.shape[1]), int(detection[3] * frame.shape[0])
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indices) > 0:
        # Flatten indices and extract valid boxes, confidences, and class_ids
        indices = indices.flatten()
        boxes = [boxes[i] for i in indices]
        confidences = [confidences[i] for i in indices]
        class_ids = [class_ids[i] for i in indices]

    # Count humans and animals
    human_count = sum(1 for id in class_ids if labels[id] == "person")
    animal_count = sum(1 for id in class_ids if labels[id] in ["cat", "dog"])

    # Draw bounding boxes
    frame = draw_bounding_boxes(frame, boxes, confidences, class_ids, labels)

    # Display counts on the frame
    cv2.putText(frame, f"Humans: {human_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Animals: {animal_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Human and Animal Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
