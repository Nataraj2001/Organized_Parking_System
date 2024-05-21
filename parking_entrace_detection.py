import cv2
import numpy as np
from ultralytics import YOLO
from models.models import ParkingSpace, db
from pathlib import Path
from flask import Flask

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.secret_key = 'POOJAVVCE'  # Set your secret key here
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///parking.db'

db.init_app(app)

# Load parking spaces from the database
def load_parking_spaces():
    with app.app_context():
        parking_spaces = []
        for space in ParkingSpace.query.all():
            coords = list(map(int, space.coordinates.split()))
            parking_spaces.append([(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)])
        return parking_spaces

# Read entrance line coordinates from file
def read_entrance_line_from_file(video_filename):
    entrance_line_filename = f"{video_filename}_entrancemap.txt"
    try:
        with open(entrance_line_filename, 'r') as file:
            line = file.readline().strip()
            coords = list(map(int, line.split()))
            entrance_line = [(coords[0], coords[1]), (coords[2], coords[3])]
            return entrance_line
    except FileNotFoundError:
        print(f"Entrance line file '{entrance_line_filename}' not found.")
        return None

# Check database for available parking spaces
def check_available_spaces():
    with app.app_context():
        available_spaces = [space.id for space in ParkingSpace.query.filter_by(status='empty').all()]
    return available_spaces



def main(input_video_path):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video '{input_video_path}'")
        return

    video_filename = Path(input_video_path).stem
    parking_spaces = load_parking_spaces()
    entrance_line = read_entrance_line_from_file(video_filename)

    if entrance_line is None:
        return

    # Initialize variables for tracking
    previous_midpoints = {}  # Dictionary to store previous midpoints of tracked vehicles
    line_position = entrance_line
    crossed_vehicles = set()  # Set to track vehicles that have already crossed the line

    # Process the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))

        # Draw the crossing line on the frame
        cv2.line(frame, line_position[0], line_position[1], (0, 255, 0), 2)

        # Run tracking on the current frame
        results = model.track(frame, persist=True)

        # Check if tracking IDs are available
        if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            # Check for line crossing by vehicle midpoints
            for box, id in zip(boxes, ids):
                midpoint_x = (box[0] + box[2]) // 2
                midpoint_y = (box[1] + box[3]) // 2

                if id in previous_midpoints:
                    previous_midpoint_x = previous_midpoints[id][0]
                    previous_midpoint_y = previous_midpoints[id][1]
                else:
                    previous_midpoint_x = midpoint_x
                    previous_midpoint_y = midpoint_y

                # Check if the midpoint crosses the line and the vehicle has not crossed before
                if (midpoint_y > line_position[0][1] and previous_midpoint_y <= line_position[0][1]) or \
                (midpoint_y < line_position[0][1] and previous_midpoint_y >= line_position[0][1]):
                    if id not in crossed_vehicles:
                        crossed_vehicles.add(id)
                        print(f"Car {id} is entering the parking lot. Checking parking space status...")
                        available_spaces = check_available_spaces()
                        print(f"Parking Spaces Available: {available_spaces}")

                        # Draw a bounding box around the vehicle that crossed the line
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                        cv2.putText(frame, f"Car {id} - Crossed", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                        # Pause the video until a key press
                        cv2.waitKey(0)

                # Update previous midpoint for the current vehicle
                previous_midpoints[id] = (midpoint_x, midpoint_y)

            # Draw boxes and IDs on the frame for all tracked vehicles
            for box, id in zip(boxes, ids):
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Car {id}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            print("Tracking IDs not available.")

        # Display the frame
        cv2.imshow("frame", frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = 'parking1_entrance.mp4'
    main(input_video_path)