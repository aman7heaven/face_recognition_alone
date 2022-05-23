
from flask import Flask,render_template,Response
import cv2

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
           break
        else:
            from flask import Flask,render_template,Response
            from face_recognition.api import face_distance, face_encodings
            import cv2
            import numpy as np
            import face_recognition
            import os
            from datetime import datetime
            from datetime import date
            from datetime import time
            import pymongo
            #import dnspython
            from pymongo import MongoClient
            myClient=pymongo.MongoClient("mongodb+srv://aman7heaven:engage2022@cluster0.buqua.mongodb.net/?retryWrites=true&w=majority")
            #Creating Database in mongoDB
            myDB=myClient["Students_data"]
            myCol=myDB["Attendance_record"]

            # from PIL import ImageGrab

            path = 'ImagesAttendance'
            images = []
            classNames = []
            myList = os.listdir(path)
            print(myList)
            for cl in myList:
                curImg = cv2.imread(f'{path}/{cl}')
                images.append(curImg)
                classNames.append(os.path.splitext(cl)[0])
            print(classNames)
            
            def findEncodings(images):
                encodeList = []
                for img in images:

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encode = face_recognition.face_encodings(img)[0]
                    encodeList.append(encode)
                return encodeList
            
            def markAttendance(name):
                
                  now =datetime.now()
                  today_date=now.strftime("%x")
                  dtstring=now.strftime('%H:%M:%S')
                  student_data={'Name': name ,'Date': today_date , 'Time': dtstring , 'Status': 'Present' }
                  if myCol.find_one({'Name':name}):
                    print("already exists")
                  else:
                   myCol.insert_one(student_data)
                   print('done')
                  
                 #if myCol.find({'Name':name}):
                    #cv2.destroyAllWindows()

                 
                
              


            #### FOR CAPTURING SCREEN RATHER THAN WEBCAM
            # def captureScreen(bbox=(300,300,690+300,530+300)):
            #     capScr = np.array(ImageGrab.grab(bbox))
            #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
            #     return capScr
            
            encodeListKnown = findEncodings(images)
            print('Encoding Complete')
            
            cap = cv2.VideoCapture(0)
            
            while True:
                success, img = cap.read()
            #img = captureScreen()
                imgS = cv2.resize(img,(0,0),None,0.25,0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
                facesCurFrame = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
            
                for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                    #print(faceDis)
                    matchIndex = np.argmin(faceDis)
            
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        #print(name)
                        y1,x2,y2,x1 = faceLoc
                        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                        markAttendance(name)
                        
                
                        

                    #cv2.imshow('Webcam',img)
                    #cv2.waitKey(1)
                #nparr = np.fromstring(frame, np.uint8)
               # frame = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR) 
               # nparr = np.fromstring(frame, np.uint8)
               # frame = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
                ret,buffer=cv2.imencode('.jpeg',img)
                img=buffer.tobytes()
                yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' +img+ b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)