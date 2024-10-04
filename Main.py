import cv2
import numpy as np

# Function to apply CLAHE for dehazing
def dehaze(frame):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel with a and b channels
    limg = cv2.merge((cl, a, b))
    
    # Convert the LAB image back to BGR
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final

# Accessing the mobile camera via IP webcam or other streaming service
# You can use an app like "IP Webcam" (available on Android)
# In IP Webcam, you can stream the video feed to an IP address, which you can access like this:
# cap = cv2.VideoCapture("http://<your_mobile_ip>:<port>/video")

# For now, we'll use the default webcam as an example
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply dehazing to each frame
    dehazed_frame = dehaze(frame)

    # Display the original and dehazed frames side by side
    combined_frame = np.hstack((frame, dehazed_frame))

    cv2.imshow('Original vs Dehazed', combined_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()