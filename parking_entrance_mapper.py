import cv2

def mark_entrance_line(image):
    clone = image.copy()
    marker_locations = []
    entrance_line = None

    def draw_entrance_line(clone, marker_locations):
        if len(marker_locations) == 2:
            cv2.line(clone, marker_locations[0], marker_locations[1], (0, 128, 0), 2)

    def mouse_callback(event, x, y, flags, param):
        nonlocal entrance_line
        if event == cv2.EVENT_LBUTTONDOWN:
            marker_locations.append((x, y))
            if len(marker_locations) == 2:
                entrance_line = marker_locations[:]  # Copy the coordinates for the line
                draw_entrance_line(clone, marker_locations)

    cv2.namedWindow('Mark Entrance Line')
    cv2.setMouseCallback('Mark Entrance Line', mouse_callback)

    while True:
        cv2.imshow('Mark Entrance Line', clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):  # Press 'r' to reset markers
            clone = image.copy()
            marker_locations.clear()
            entrance_line = None
        elif key == ord('c') and entrance_line is not None:  # Press 'c' to capture entrance line
            break

    cv2.destroyAllWindows()
    return entrance_line

def save_entrance_line_to_text(entrance_line, video_filename):
    if entrance_line:
        map_filename = f"{video_filename.split('.')[0]}_entrancemap.txt"
        with open(map_filename, 'w') as file:
            file.write(f"{entrance_line[0][0]} {entrance_line[0][1]} {entrance_line[1][0]} {entrance_line[1][1]}")

        print(f"Entrance line saved to '{map_filename}'")
    else:
        print("No entrance line marked.")

def main():
    video_filename = 'parking1_entrance.mp4'  # Change this to your video filename
    cap = cv2.VideoCapture(video_filename)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from video.")
        return
    
    frame = cv2.resize(frame, (1020, 500))
    entrance_line = mark_entrance_line(frame.copy())
    save_entrance_line_to_text(entrance_line, video_filename)

    cap.release()

if __name__ == "__main__":
    main()
