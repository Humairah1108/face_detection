import cv2

cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# capture video frame by frame from computer's webcam
# Instantiate an object from the video capture class
# Zero means selecting the Video Capture from the default source (Default webcam), can also give video file name
cap = cv2.VideoCapture(0)

# We want to do this continuously, we use a loop
while True:
    # Reads first frame from videoCapture obj
    ret, frame = cap.read()
    # Our operations on the frame come here

    gray = cv2.cvtColor(frame, 0)
    detections = cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Displays the frame using the imshow method
    cv2.imshow('frame', frame)

    # Using waitKey to avoid overwriting of the frames, else nothing will show up
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
