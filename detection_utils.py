import cv2
import numpy as np

def load_yolo_model(config_path, weights_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    return net, output_layers

def draw_bounding_boxes(frame, boxes, confidences, class_ids, labels):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        label = labels[class_ids[i]]

        # Set color based on the label
        if label == 'person':
            color = (0, 255, 0)  # Green for humans
        elif label == 'cat':
            color = (255, 0, 0)  # Blue for cats
        elif label == 'dog':
            color = (0, 0, 255)  # Red for dogs
        else:
            color = (255, 255, 255)  # White for any other labels

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame