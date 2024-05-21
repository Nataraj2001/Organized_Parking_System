from flask import Flask, request, redirect, url_for, flash, send_from_directory, render_template, render_template_string
import os
from werkzeug.utils import secure_filename
import cv2

from parkingDetections.models.models import ParkingSpace, db
from parking_space_mapper import mark_parking_lots, save_to_text, save_to_text_uploadFolder
from flask_sqlalchemy import SQLAlchemy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.secret_key = 'POOJAVVCE'  # Set your secret key here
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///parking.db'
db.init_app(app)

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Call the marking function
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if not ret:
                flash('Error reading video frame.')
                return redirect(request.url)
            frame = cv2.resize(frame, (1020, 500))
            marker_locations = mark_parking_lots(frame.copy())

            # Log marker_locations
            logging.info(f"Marker Locations: {marker_locations}")
            # save for the file
            save_to_text_uploadFolder(marker_locations, filename, app.config['UPLOAD_FOLDER'])

            cap.release()

            # Create database entries for parking spaces
            db.create_all()
            # Delete all records from the ParkingSpace table
            ParkingSpace.query.delete()
            # Assuming marker_locations is a list of tuples of coordinates
            for i in range(0, len(marker_locations), 4):
                try:
                    # Extract coordinates for each parking space
                    corner1, corner2, corner3, corner4 = marker_locations[i:i + 4]
                    x1, y1 = corner1
                    x2, y2 = corner2
                    x3, y3 = corner3
                    x4, y4 = corner4

                    # Concatenate the coordinates into a single string
                    coordinates = f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"
                    parking_space = ParkingSpace(coordinates=coordinates)
                    db.session.add(parking_space)
                except ValueError:
                    flash('Error parsing coordinates.')
                    return redirect(request.url)

            db.session.commit()

            # Retrieve parking spaces from the database
            spaces = ParkingSpace.query.all()

            # Render the marked spaces on a webpage
            return render_template('parking_spaces.html', spaces=spaces)

    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Video</title>
    </head>
    <body>
        <form action="/" method="POST" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    '''



@app.route('/parking-spaces')
def parking_spaces():
    spaces = ParkingSpace.query.all()
    return render_template('parking_spaces.html', spaces=spaces)


if __name__ == '__main__':
    app.run(debug=True)
