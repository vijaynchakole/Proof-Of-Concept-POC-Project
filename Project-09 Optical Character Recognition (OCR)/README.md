### Project Name : OCR : Optical Character Recognition 

##### Dependencies : cv2, pytesseract

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
