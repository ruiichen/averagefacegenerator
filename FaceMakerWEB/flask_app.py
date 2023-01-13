#################################################################
# Rui Chen
# Python website that generates the average of faces 
# December 21, 2022
#################################################################



#################################################################
# Important things to import!
#################################################################

from flask import Flask, flash, request, redirect, url_for, render_template, after_this_request
import urllib.request
import os
from facemaker import makeAverageImage
from werkzeug.utils import secure_filename
import cv2
import dlib
import numpy as np
import math
import sys
from PIL import Image
import io
import base64
#################################################################

# location of the model (path of the model).
Model_PATH = "static/shape_predictor_68_face_landmarks.dat"

# Extracts important methods from Dlib
frontalFaceDetector = dlib.get_frontal_face_detector()
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)
 
app = Flask(__name__)
app.secret_key = "guh6"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
  
ALLOWED_EXTENSIONS = set(['jpg','webp','png','jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
@app.route('/')
def upload_form():
    return render_template('index.html',img_data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
 
@app.route('/', methods=['POST'])
def upload_image():
    # Create an array of array of points.
    pointsArray = []
    # Create array of images.
    imagesArray = []

    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    
    for file in files:
        filename = secure_filename(file.filename)
        if file and allowed_file(file.filename):
            # Create an array of points
            points = []

            # Reads the image found and converts it to openCV image
            file=Image.open(file)
            opencv_img = np.asarray(file)
            opencv_img = opencv_img[:,:,::-1].copy()
            
            #################################################################
            # Adds the image to the array of images
            img = np.float32(opencv_img)/255.0
            imagesArray.append(img)
            #################################################################
            
            # Detects all the faces in the image
            imageRGB = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
            allFaces = frontalFaceDetector(imageRGB, 0)

            if len(allFaces) == 0:
                flash('Please ensure every picture has a face!')
                return redirect(request.url)

            # Makes a rectangle focused on the first face
            faceRectangleDlib = dlib.rectangle(int(allFaces[0].left()),int(allFaces[0].top()),
            int(allFaces[0].right()),int(allFaces[0].bottom()))

            # Detects the landmarks on the face and pairs each into a point,
            # hen appends them to the list of points
            detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)
            for p in detectedLandmarks.parts():
                points.append((int(p.x),int(p.y))) 

            # Stores the points in that image into the array of points
            pointsArray.append(points)
        else:
            flash('Please submit a .jpg, .png, .webp, or a .jpeg!')
            return redirect(request.url)

    # Creates the blended image and converts it from a numpy array
    # to a base64 image to pass to HTML for display
    imgToShow= makeAverageImage(pointsArray,imagesArray)
    imgToShow= cv2.cvtColor(imgToShow,cv2.COLOR_BGR2RGB)
    imgToShow = Image.fromarray(imgToShow)
    data = io.BytesIO()
    imgToShow.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template('index.html', img_data=encoded_img_data.decode('utf8'))

@app.route('/display/<filename>')
def display_image(filename):
   return redirect(request.url)
    
if __name__ == "__main__":
    app.run()
