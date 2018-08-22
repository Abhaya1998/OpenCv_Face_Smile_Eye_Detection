import cv2
#Importing Haar Cascade Classifiers
face_detector=cv2.CascadeClassifier("face.xml")
eye_detector=cv2.CascadeClassifier("eye.xml")
smile_detector=cv2.CascadeClassifier("smile.xml")
#Setting up the webcam
cap=cv2.VideoCapture(0)
while(1):
    #Reading Each Frame
    _, frame=cap.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Converting BGR frame to GRAY

    face=face_detector.detectMultiScale(gray,1.3,5) #returns a tuple containg coordinates of faces
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #Creating a boundary for fae
        face_gray=gray[y:y+h,x:x+w]
        face_org=frame[y:y+h,x:x+w]

        eyes=eye_detector.detectMultiScale(face_gray,1.1,22) #detecting Eye
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_org,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        smile=smile_detector.detectMultiScale(face_gray,1.7,22) #Detecting smile
        for (ex,ey,ew,eh) in smile:
            cv2.rectangle(face_org,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    cv2.imshow("Detected",frame)   #Showing the frames
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()  #Destroying all the windows
