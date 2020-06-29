
# Data Science Projects:

## Computer Vision Projects :

### 1) Flask Face Detection Application and its Deployment to cloud :

upload your image and submit it. then it will detect your face in that image

Application URL : http://vijayfacedetection.pythonanywhere.com/

Dependencies : cv2, numpy, urllib3, Flask (python web framework)

short overview :

with the help of haarcascade classifier I have detected face in given image 

then I have drawn rectangle around detected faces in image

and finally deploy my model to cloud (pythonanywhere.com)


### 2) Snapchat-based Augmented Reality :

Dependencies : cv2 (OpenCV)

short overview :
I have used haarcascade classifier for face detection, nose detection and eyespair detection which detects faces, nose and eyes respectively in given image and return its 
vertices then

I drawn rectangle around those faces with the help of return vertices
By doing some mathematical calculation on pixel region(x-axis, y-axis) 

I found ROI (region of interest) for placing moustache and glasses


### 3) OCR : Optical Character Recognition 

Dependencies : cv2, pytesseract

short overview : with the help of pytesseract library I have recognised character in given image

steps :

read image

preprocessing image (convert to gray scale)

perform threshold technique on gray image

applying dilation of the threshold

Specify structure shape and kernel size. A smaller value like (10, 10) will detect each word instead of a sentence.

finding contours (sequence of point defining object in an image)

Contours are typically used to find a white object from a black background.

create new text file

extracts texts from contours and written in text file

### 4) QR Code Scanner 

Dependencies : cv2 (OpenCV), numpy, re, matplotlib, webrowser

short overview :

with the help of cv2 I have created QR Code detector which returns data, the array of vertices of the found QR code quadrangle.
QR Code Generator

Dependencies : qrcode, cv2 (OpenCV), numpy, re

with help of qrcode library I have created QR code generator, we have to just provide text then it will create QR code image


### 5) Face Recognition

link : https://github.com/vijaynchakole/Face_Recognition

Dependencies : face_recognition, cv2, sklearn, matplotlib, selenium (for images download from google)

short overview:

I have done this project from scratch 

I have selected 5 famous people for the Face Recognition Model.

Names of those people : Modi, Trump, Putin, Xi Jin Ping, Kim Jong Un 

and downloaded their images from google using selenium for creating training and testing dataset.

I have created a Face Recognition Model. which recognizes the faces of above mentioned people.


code explanation in short :

with the help of selenium library I have downloaded all images for my project

1
with the help of face_recognition library I have created face encodings once I get face encodings then 

I have used them for Model build.

2

I have built an SVC (Support Vector Classifier) model by using face encodings

3

Once model is built then it is used for face recognition in given image



## Classificatoin Problem Statement Projects :

### 6) Titanic Survival Predictions : 

Dependencies :

numpy, pandas, sklearn,matplotlib,seaborn,pandas_profiling, joblib, pickle

Agorithms Used :

Logistic Regression

KNeighborsClassifier

LinearSVC

Decision Tree

RandomForestClassifier

### 7)  Naval Mine Detector :

Supervised Machine Learning :

Classification Problem Statement

Technology : Deep Learning with Neural network using Python


## Natural Language Processing :

### 8) Spam Detector

with the help NLP techniques we classify whether sms is spam or Not

Dependencies : numpy, pandas, sklearn, nltk, re(regular expression)


## Web Scraping Projects : 

### 9) Crawl popular website IMDb (Internet Movie Database) and create a database of Indian movie celebrities containing their images and personality traits.


I have selected website https://www.imdb.com/list/ls002913270/ for web scrapping. IMDB website provides top 100 indian celebraties list with their best movie work, images and 

some personal information.

I have scrapped all information ralated celebraties. list of scrapped information as per below

Celebraty Name :

Celebraty image :

profession :

best work movie :

personal information :

Dependencies : os, urllib, bs4 (BeautifulSoup), wget (for image download), mysql (for database connectivity and database operation)


### 10) Web Scraping Wikipedia.

Automation script which fetch URL, title and content headers of wikipedia page using Beautiful Soup.

Dependencies : request, BeautifulSoup


## Python Automation Projects :

### 11) Directory Operation :

Automation script which display all files , find checkSum of files, display duplicate files, remove duplicate files

1). Directory Watcher

2). Directory File CheckSum

3). Directory Duplicate File Detector

4). Directory duplicate File Removal

Dependencies : os, sys, hashlib


### 12) Process Monitor :

Automation script which accepts time interval from user and create log file in that Log Monitoring folder directory which contains information of all running processes. 

After creating the log file send that log file through mail.

Dependencies :  os, psutil, time, urllib, sys, smtplib, schedule, email


### 13) Web Launcher

Topic : Automation script which accept file name. Extract all URL’s from that file and connect to that URL’s through Webbrowser.

Dependencies : urllib, re, webbrowser, os


### small size projects are in another repository :

https://github.com/vijaynchakole/dataScienceProject


## work in progress :

### currently, I am working on Stocks Screener project which is web application
### once I complete this project then I deploy it to heroku 
### Web Framework : FastAPI
