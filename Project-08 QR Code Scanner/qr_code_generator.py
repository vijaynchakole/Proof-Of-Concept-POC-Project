# -*- coding: utf-8 -*-
"""
Created on Thu May 14 05:06:00 2020

@author: vijaynchakole

topic: QR code generator

"""
import qrcode
import cv2
import numpy as np
qr = qrcode.QRCode(
     version =1,
     error_correction = qrcode.constants.ERROR_CORRECT_L,
     box_size = 10,
     border = 4
     )

# qr.add_data("https://github.com/vijaynchakole")
qr.add_data("https://internshala.com/student/resume?detail_source=resume_intermediate")
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color = "white")


img.save("internshala_qr_code_1.png")



