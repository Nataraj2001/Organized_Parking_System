import cv2
import numpy as np
from ultralytics import YOLO
# from yolov5 import YOLOv5
import time
from pathlib import Path

from flask import Flask, request, redirect, url_for, flash, send_from_directory, render_template, render_template_string
from flask_sqlalchemy import SQLAlchemy
import logging

from models.models import ParkingSpace, db

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.secret_key = 'POOJAVVCE'  # Set your secret key here
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///parking.db'

db.init_app(app)




#this function loads from the database ;

def load_parking_areasDatabase():
    parking_areas = []
    for space in ParkingSpace.query.all():
        coords = list(map(int, space.coordinates.split()))
        parking_areas.append([(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)])
    return parking_areas, len(parking_areas)


#this will update parking space
def update_parking_space_status(available_spaces):
    for i, is_available in enumerate(available_spaces):
        space = ParkingSpace.query.get(i + 1)
        if is_available:
            space.status = 'empty'
        else:
            space.status = 'occupied'
    db.session.commit()


def print_parking_space_status():
    for space in ParkingSpace.query.all():
        print(f"Parking Space {space.id}: {space.status}")


#end of new functions

def load_parking_areas(video_filename):
    map_filename = f"{video_filename}_map.txt"
    parking_areas = []

    try:
        with open(map_filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                coords = list(map(int, line.strip().split()))
                parking_areas.append([(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)])
    except FileNotFoundError:
        print(f"Map file '{map_filename}' not found.")

    return parking_areas, len(parking_areas)


#old model , changing to new parkingDetections
def detect_cars(frame):
    model = YOLO('yolov8s.pt')
    results = model.predict(frame, conf=0.35)
    return results[0].boxes.data


# def detect_cars(frame):
#     # Initialize the YOLOv5 model
#     model = YOLOv5('yolov5s.pt', device='cuda' if torch.cuda.is_available() else 'cpu')

#     # Convert the frame to RGB (YOLOv5 expects RGB images)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Perform inference
#     # Note: The YOLOv5 model's predict method expects a list of images, so we wrap frame_rgb in a list
#     results = model.predict([frame_rgb])

#     # Extract bounding boxes from the results
#     # The results object contains several attributes, including 'xyxy' for bounding boxes
#     car_boxes = results.xyxy[0].cpu().numpy()

#     # Filter out detections with low confidence
#     car_boxes = car_boxes[car_boxes[:, 4] > 0.35]

#     return car_boxes


def calculate_overlap(area, car):
    x1, y1, x2, y2 = map(int, car[:4])  # Convert coordinates to integers
    rect = np.array(area, np.int32)
    rect = rect.reshape((-1, 1, 2))
    intersection = cv2.pointPolygonTest(rect, ((x1 + x2) // 2, (y1 + y2) // 2), False)
    if intersection >= 0:
        return cv2.contourArea(rect)
    else:
        return 0


def count_available_spaces(parking_areas, car_boxes, car_label=2):
    NUM_SPACES = len(parking_areas)
    available_spaces = [True] * NUM_SPACES
    total_cars_detected = 0
    total_occupied_spaces = 0
    cars_in_parking = []  # List to store bounding boxes of cars in parking spaces
    car_boxes_filtered = []

    if car_boxes is not None and len(car_boxes) > 0:
        car_boxes_filtered = [box for box in car_boxes if int(box[5]) == car_label]
        occupied_spaces = set()  # Set to store indices of parking spaces occupied by cars

        # Identify occupied parking spaces
        for car in car_boxes_filtered:
            total_cars_detected += 1
            max_overlap = -1
            max_index = -1
            for i, area in enumerate(parking_areas):
                overlap = calculate_overlap(area, car)
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_index = i
            if max_overlap > 0:  # Only consider cars overlapping with parking spaces
                occupied_spaces.add(max_index)
                total_occupied_spaces += 1
                cars_in_parking.append(car)  # Store bounding box of car in parking space

        # Mark occupied parking spaces as unavailable
        for space_index in occupied_spaces:
            available_spaces[space_index] = False

    total_free_spaces = NUM_SPACES - total_occupied_spaces
    return available_spaces, total_cars_detected, total_occupied_spaces, total_free_spaces, car_boxes_filtered, cars_in_parking


# def main():
#     input_video_path = 'parkone.mp4'
#     input_video_name = Path(input_video_path).stem
#     cap = cv2.VideoCapture(input_video_path)
#
#     # Defining the video properties for the output Videofile
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     out = cv2.VideoWriter('processed_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1020, 500))
#
#     PARKING_AREAS, NUM_SPACES = load_parking_areas(input_video_name)
#
#     frame_count = 0
#     skip_frames = 5 # Adjust this value to skip more or fewer frames
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = cv2.resize(frame, (1020, 500))
#
#         # Skip processing for some frames to speed up the process
#         if frame_count % skip_frames == 0:
#             car_boxes = detect_cars(frame)
#             available_spaces, total_cars_detected, total_occupied_spaces, total_free_spaces, car_boxes_filtered, cars_in_parking = count_available_spaces(PARKING_AREAS, car_boxes)
#
#             for i, area in enumerate(PARKING_AREAS):
#                 color = (0, 255, 0) if available_spaces[i] else (0, 0, 255)
#                 cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
#                 overlay = frame.copy()
#                 cv2.fillPoly(overlay, [np.array(area, np.int32)], color)
#                 alpha = 0.3
#                 frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
#                 cv2.putText(frame, str(i+1), ((area[0][0] + area[1][0] + area[2][0] + area[3][0]) // 4, (area[0][1] + area[1][1] + area[2][1] + area[3][1]) // 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#
#             cv2.putText(frame, f'Total cars detected: {total_cars_detected}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             cv2.putText(frame, f'Total occupied parking spaces: {total_occupied_spaces}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             cv2.putText(frame, f'Total free parking spaces: {total_free_spaces}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#
#             cv2.imshow('Processed Video', frame)
#             cv2.waitKey(1)
#
#             out.write(frame)
#
#         frame_count += 1
#
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


#new modified main function to retrive from the database


def main():
    # Create an application context
    with app.app_context():
        input_video_path = 'occupiedParkingSpace.mp4'
        input_video_name = Path(input_video_path).stem
        cap = cv2.VideoCapture(input_video_path)

        # Now you can safely call load_parking_areasDatabase() within the application context
        PARKING_AREAS, NUM_SPACES = load_parking_areasDatabase()

        frame_count = 0
        skip_frames = 5 # Adjust this value to skip more or fewer frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1020, 500))

            # Skip processing for some frames to speed up the process
            if frame_count % skip_frames == 0:
                car_boxes = detect_cars(frame)
                available_spaces, total_cars_detected, total_occupied_spaces, total_free_spaces, car_boxes_filtered, cars_in_parking = count_available_spaces(
                    PARKING_AREAS, car_boxes)

                # Update parking space status in the database
                update_parking_space_status(available_spaces)

                # Print parking space status
                print_parking_space_status()

                # Display the result
                for i, area in enumerate(PARKING_AREAS):
                    color = (0, 255, 0) if available_spaces[i] else (0, 0, 255)
                    cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [np.array(area, np.int32)], color)
                    alpha = 0.3
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    cv2.putText(frame, str(i + 1), ((area[0][0] + area[1][0] + area[2][0] + area[3][0]) // 4,
                                                    (area[0][1] + area[1][1] + area[2][1] + area[3][1]) // 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.putText(frame, f'Total cars detected: {total_cars_detected}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
                cv2.putText(frame, f'Total occupied parking spaces: {total_occupied_spaces}', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Total free parking spaces: {total_free_spaces}', (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow('Processed Video', frame)
                cv2.waitKey(1)

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()