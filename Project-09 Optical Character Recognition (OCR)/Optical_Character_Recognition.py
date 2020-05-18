# -*- coding: utf-8 -*-
"""
Created on Fri May 15 04:34:31 2020

@author: vijaynchakole
topic : OCR
"""
# import required libraries
import cv2 
import pytesseract

# system path of tesseract.exe
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


# read image from from which text needs to be extracted
img = cv2.imread("sample4.jpg")
#img = cv2.imread("q2.jpg")
#path = "C:\\Users\\hp\\Desktop\\practicals_ml\\OpenCV\\Projects\\Project-09 OCR\\test_ocr.png"
#img = cv2.imread(path)
# preprocessing the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape
# perform OTSU threshold technique
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size.  
# Kernel size increases or decreases the area  
# of the rectangle to be detected. 
# A smaller value like (10, 10) will detect  
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))

""" Dilation makes the groups of text to be detected more accurately since it dilates (expands) a text block."""
# applying dilation of the threshold
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)


"""
Contours : sequence of point defining object in an image

Contours are typically used to find a white object from a black background.

"""
# finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )

# creating copy of an image
img2 = img.copy()

# a text file create and flushed
file = open("recognized.txt", "w+")
file.write("")
file.close()

# Looping through the identified contours 
# Then rectangular part is cropped and passed on 
# to pytesseract for extracting text from it 
# Extracted text is then written into the text file 
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # draw rectangle on copied image
    rect = cv2.rectangle(img2, (x,y), (x+w, y+h), (0,255,0), 2)
    
    # cropping text block for giving input to OCR
    cropped = img2[y:y+h, x:x+w]
    
    # open the file in append mode
    file = open("recognized.txt", "a")
    
    # apply OCR to cropped image
    text = pytesseract.image_to_string(cropped)
    
    # appending the text into the file
    file.write(text)
    file.write("\n")
    
    # close the file 
    file.close()