### Project Name :  Snapchat-based augmented reality

Project files : snapchat_augmeted_reality_moustache.py , snapchat_augmeted_reality_glasses.py


Dependencies : cv2 (OpenCV)

short overview :
I have used haarcascade classifier for face detection, nose detection and eyespair detection which detects faces, nose and eyes respectively in given image and return its vertices then
I drawn rectangle around those faces with the help of return vertices
By doing some mathematical calculation on pixel region(x-axis, y-axis) 
I found ROI (region of interest) for placing moustache and glasses
