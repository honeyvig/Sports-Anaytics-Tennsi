# Sports-Anaytics-Tennsi
For a sports analytics project focused on analyzing tennis footage, the task will involve processing video frames, detecting players, tracking them, and extracting relevant metrics (such as player movements, shots, and performance analysis). You would likely use YOLO (You Only Look Once) for object detection and OpenCV for handling video data and image processing.

Here's a Python code example that leverages YOLOv5 for object detection, OpenCV for video processing, and some basic machine learning techniques for extracting insights from tennis match footage.
Steps for Implementation:

    Load and Preprocess Video Data: Use OpenCV to load the video and extract frames.
    Apply YOLO Object Detection: Use YOLOv5 for detecting players, balls, and other relevant objects (such as the court).
    Track Player and Ball Movements: After detection, track the players and balls over time to extract movement and performance data.
    Metrics Extraction: Compute player positions, movement speed, shot detection, etc., using the tracking data.
    Machine Learning: Analyze the extracted data to gain insights into player performance, such as shot accuracy, reaction time, and movement efficiency.

Requirements:

    Install YOLOv5 (or use a pretrained model from torch).
    Install OpenCV, NumPy, and other required libraries.

pip install opencv-python torch torchvision numpy matplotlib

Python Code Example:

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Load YOLOv5 pre-trained model (can also load a custom model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use yolov5s for faster inference, or yolov5m/l for better accuracy

# Function to process video and track objects
def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize variables
    player_positions = []  # List to store player positions
    ball_positions = []    # List to store ball positions
    tracking_window = deque(maxlen=20)  # For tracking the last N frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to speed up processing
        frame_resized = cv2.resize(frame, (640, 480))

        # Run the YOLO model on the frame to detect objects
        results = model(frame_resized)

        # Parse detections (YOLOv5 model returns results in format: [x1, y1, x2, y2, confidence, class_id])
        detections = results.xyxy[0].cpu().numpy()

        # Loop through detections and classify them
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            label = model.names[int(class_id)]  # Get the label of the detected object

            # Classify and store player and ball positions
            if label == 'person':  # Assuming the 'person' class corresponds to players
                player_positions.append(((x1 + x2) / 2, (y1 + y2) / 2))  # Store the center position of the player
            elif label == 'sports ball':  # Assuming 'sports ball' corresponds to the tennis ball
                ball_positions.append(((x1 + x2) / 2, (y1 + y2) / 2))  # Store the center position of the ball

            # Draw bounding boxes and labels on the frame
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame_resized, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Update the tracking window with the most recent positions
        tracking_window.append((player_positions[-1] if player_positions else None, ball_positions[-1] if ball_positions else None))

        # Optionally display the frame for debugging
        cv2.imshow('Tennis Match Analysis', frame_resized)

        # Exit condition (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    return player_positions, ball_positions

# Function to analyze and extract performance metrics
def analyze_performance(player_positions, ball_positions):
    # Example analysis: Player speed
    speeds = []
    for i in range(1, len(player_positions)):
        x1, y1 = player_positions[i-1]
        x2, y2 = player_positions[i]
        speed = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Euclidean distance between two points
        speeds.append(speed)
    
    # Plotting the player positions and speeds over time
    plt.figure(figsize=(10, 5))
    
    # Plot player positions on the court
    plt.subplot(1, 2, 1)
    plt.scatter([pos[0] for pos in player_positions], [pos[1] for pos in player_positions], label='Player Positions')
    plt.title('Player Positions on Court')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    
    # Plot player speed over time
    plt.subplot(1, 2, 2)
    plt.plot(speeds, label='Player Speed (Pixels per Frame)')
    plt.title('Player Speed Analysis')
    plt.xlabel('Time (Frames)')
    plt.ylabel('Speed (Pixels/frame)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main execution flow
if __name__ == "__main__":
    video_path = 'path_to_video.mp4'  # Replace with your video path
    player_positions, ball_positions = process_video(video_path)
    analyze_performance(player_positions, ball_positions)

Key Concepts:

    YOLO Object Detection:
        The torch.hub.load('ultralytics/yolov5', 'yolov5s') function loads a pre-trained YOLOv5 model that is capable of detecting various objects, including "person" (for players) and "sports ball" (for the tennis ball).
        Each frame from the video is passed to YOLO, which detects and classifies objects. The bounding box coordinates (x1, y1, x2, y2) are used to locate the player and ball in the frame.

    Tracking Movement:
        We calculate the center of the bounding boxes ((x1 + x2) / 2 and (y1 + y2) / 2) to track player and ball positions over time.
        We store the player and ball positions in lists for further analysis.

    Performance Metrics Extraction:
        The analyze_performance function computes basic metrics like player speed by calculating the Euclidean distance between consecutive player positions in each frame.
        The script also generates plots to visualize the player’s movements on the court and speed over time.

    Video Processing:
        OpenCV is used to read video frames and display them with bounding boxes and labels. The video is processed frame by frame, and object detections are visualized.
        The loop continues until the entire video has been processed.

Further Enhancements:

    Advanced Tracking: Use Kalman Filters or SORT (Simple Online and Realtime Tracking) to improve the tracking of players and the ball over time.
    Shot Detection: Add functionality to detect the specific moment when a player hits the ball (this could be done using motion analysis or detecting changes in the ball’s trajectory).
    AI-Driven Metrics: Implement machine learning algorithms to predict player performance or strategy based on the extracted data.
    Court Analysis: Use computer vision techniques to track the ball’s trajectory and determine if the ball is in or out of bounds.

This code provides a foundation to build on, but as the project scales, consider adding more sophisticated tracking and machine learning algorithms to analyze player performance more comprehensively.
