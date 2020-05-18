# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:52:46 2020

@author: vijaynchakole

"""

from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



import urllib3.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify
import urllib3.request
from face_processing import FaceProcessing


import cv2
import numpy as np



fc = FaceProcessing()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
	return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['file']
    filename=file.filename
    file.save("/home/vijayfacedetection/mysite/static/uploads/"+filename)
    IMAGE_PATH = ("/home/vijayfacedetection/mysite/static/uploads/"+filename)

    # resize image
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)

    WIDTH = 640
    HEIGHT = 640
    DIM = (WIDTH, HEIGHT)

    image = cv2.resize(image,DIM, interpolation = cv2.INTER_AREA)

    status = cv2.imwrite(IMAGE_PATH, image)

    image=open(IMAGE_PATH,"rb")
    result = fc.face_detection(image.read())
    img_opencv = cv2.imread(IMAGE_PATH)

    for face in result:
        left, top, right, bottom = face['box']
        print(left, top, right, bottom)
        # To draw a rectangle, you need top-left corner and bottom-right corner of rectangle:
        cv2.rectangle(img_opencv, (left,top), (right,bottom), (0, 255, 255), 2)
        # Draw top-left corner and bottom-right corner (checking):
        cv2.circle(img_opencv, (left, top), 5, (0, 0, 255), -1)
        cv2.circle(img_opencv, (right, bottom), 5, (255, 0, 0), -1)

    
    seperate = filename.split(".")
    img_path = "/home/vijayfacedetection/mysite/static/uploads/output_" + seperate[0] + "." + seperate[1]
    status = cv2.imwrite(img_path, img_opencv)
    print(status)
    output = "output_" + seperate[0] + "." + seperate[1]
    return render_template('upload.html', filename=[filename,output])



@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()