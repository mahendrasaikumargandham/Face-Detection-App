# synatx to import opencv is "pip install opencv-python"
import cv2

# randrange is a fuction which was imported from random package which helps us to pic a number randomly. In this case, randrange() helps us to get random color according to the number.
from random import randrange

#importing haarcascade.xml file from the following link "https://github.com/opencv/opencv/tree/master/data/". There you can find the haarcascade file. Download the file and keep in your project directory.
# haar algorithm has all the trained data (Supervised Learning) so that, it can helps us with detecting faces according to the frames
trained_face = cv2.CascadeClassifier('haarcascade.xml')

# img = cv2.imread('download.jpeg') 
# you can uncomment the above line to see how this AI detects faces in image ('download.jpeg').

# VideoCapture(0) is used to open your webcam. if you want to detect faces in video, then change the value from 0 to the name of the video you want to play
# But make sure that the video also kept in the same directory.
webcam = cv2.VideoCapture(0)

# Looping the webcam screen.
while True:
    # taking the input from the webcam into frames.
    successful_frame_read, frame = webcam.read()
    
    # converting the color frames into grayscale frames so that the AI can detect the faces very easily.
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detecting the multiple faces at a time.
    face_coordinates = trained_face.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        
        # using randrange to input the color randomly. It will display the color which are the shades of RGB(Red, Green, Blue).
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256),
                                                  randrange(256), randrange(256)), 3)
    # here Python Face Detection App is the name of the screen. You can change this sentence according to your intrest.
    cv2.imshow('Python Face Detection App', frame)

    key = cv2.waitKey(1)
    
    if key == 81 or key == 113:
        break

# exits from the webcam if you press 'Q' in your keyboard.
webcam.release()

# prints "code executed successfully" after you pressing q in your keyboard which means the program is terminated.
print("Code executed successfully")
