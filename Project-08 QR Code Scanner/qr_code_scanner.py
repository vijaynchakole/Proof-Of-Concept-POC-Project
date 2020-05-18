# -*- coding: utf-8 -*-
"""
Created on Thu May 14 06:00:20 2020

@author: hp

https://morioh.com/p/29f7e4c5f900
"""

# import required packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import re

def show_img_with_matplotlib(color_img, title, pos):
    """shows an image using matplotlib cababilities"""
    
    # convert BGR image to to RGB
    img_RGB = color_img[:,:,::-1]
    
    ax = plt.subplot(1,2,pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis("off")
    
    
    
def show_qr_detection(img,pts):
    """draw both the lines and corners based on the array of vertices of the found QR code
    """
    pts = np.int32(pts).reshape(-1,2)
    
    for j in range(pts.shape[0]):
        cv2.line(img, tuple(pts[j]), tuple(pts[(j+1) % pts.shape[0]]),(255,0,0), 5)
    
    for j in range(pts.shape[0]):
        cv2.circle(img, tuple(pts[j]), 10, (255,0,0), -1)



# create dimensions of figures and set title
fig = plt.figure(figsize = (14,5))
plt.suptitle("QR Code detection", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')
        
        
# load the input image
# image = cv2.imread("qrcode_rotate_45_image.png")
image = cv2.imread("internshala_qr_code_1.png")



def qr_code_detector(image):
    # Create QR Code detector
    qr_code_detector = cv2.QRCodeDetector()
    
    # detect and decode the qr code by using qr_code_detector.detectAndDecode()
    # this function returns the data, the array of vertices of the found QR code quadrangle and
    # the image containing the rectified binarized QR code
    
    data, vertices, rectified_qr_code_binarized = qr_code_detector.detectAndDecode(image)

    return data, vertices, rectified_qr_code_binarized


data, vertices, rectified_qr_code_binarized = qr_code_detector(image)

image.shape
# display vertices of QR code 
x = (vertices[0][:,0],vertices[1][:,0],vertices[2][:,0],vertices[3][:,0])
#x = np.float32(x)
y = (vertices[0][:,1],vertices[1][:,1],vertices[2][:,1],vertices[3][:,1])
#y = np.float32(y)
plt.plot(x,y)
print(vertices)
#plt.scatter(x,y)

if len(data) > 0 :
    print(f"Decoded data : {data}")
    
    # show detection in the image
    show_qr_detection(image, vertices)
    
    # convert binarized  image to uint8
    rectified_image = np.uint8(rectified_qr_code_binarized)
    
    # plot the images
    show_img_with_matplotlib(cv2.cvtColor(rectified_image, cv2.COLOR_GRAY2BGR), "rectified QR code", 1)
    show_img_with_matplotlib((image), ("decoded data : " + data), 2)
    
    # show the figure
    plt.show()
    
else:
    print("QR code not detected")


def open_url(data):
    ls_url = (re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',data))
       
    if len(ls_url) > 0 :
        webbrowser.open(data, new = 2)
    else:
        print("there is no URL to open")
        print("data is : " + data)
 
# open_url(data)
