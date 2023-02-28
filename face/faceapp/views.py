from django.shortcuts import render,redirect
from django.views.decorators.csrf import csrf_protect


import cv2
from .models import *
import os
from datetime import datetime
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
from django.http import HttpResponse

# Create your views here.
def index1(request):
    att1=Attendence.objects.all().values()
    a=[]
    b=[]
    c=[]
    for i in att1:
        a.append(i['aid'])
        b.append(i['name'])
        c.append(i['Date'])
    df=pd.DataFrame()
    df['Id']=a
    df['date']=c
    df['Name']=b
    a.clear()
    b.clear()
    c=df['Name'].unique()
    for i in df['Id'].unique():
       a.append(i)
       b.append((len(a)/2)*100)
    df1=pd.DataFrame()
    df1['Student Id']=a
    df1['Name']=c
    df1['Attendence Percentage']=b
    print(a)
    print(b)
    fig = px.line(df1,x='Student Id', y='Attendence Percentage',title='Monthly Student Attendence',markers=True)
    fig.update_layout(xaxis_range=[0,5],yaxis_range=[0,101])
    g = fig.to_html()
    return render(request,'index.html',{'graph':g})
def take(request):
    
    Id = request.POST.get('fname')
    name = request.POST.get('lname')
    
    take1(Id,name)
    TrainImages()
    return render(request,'index.html')
def take1(Id,name):    
    cam = cv2.VideoCapture(0)
    directory = os.getcwd()
    print(directory)
    harcascadePath = os.path.join(directory,"haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                sampleNum = sampleNum + 1

                cv2.imwrite("TrainingImage/ " + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

                cv2.imshow('frame', img)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

            elif sampleNum > 60:
                break
    cam.release()
    cv2.destroyAllWindows()
    res = "Images Saved for ID : " + Id + " Name : " + name
    row = [Id, name]
    student=Student(name=name,rId=Id)
    student.save()
@csrf_protect   
def TrackImages(request):
    student=Student.objects.all().values()
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, im = cam.read()
        
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        
        
            
        
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                aa=Student.objects.filter(rId=Id).values()
                
                a1=aa[0]['name']
                
                tt = str(Id) + "-" + a1

            else:
                Id = 'Unknown'
                tt = str(Id)
                

            if (conf > 75):
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])

            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
           
            
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()   
    if Id!='unknown':
      att=Attendence(aid=Id,name=a1,Date=datetime.now(),Status='present')
      att.save() 
    
    return redirect('/')

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"
    


def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids
def Export(request):
    att1=Attendence.objects.all().values()
    a=[]
    b=[]
    c=[]
    for i in att1:
        a.append(i['aid'])
        b.append(i['name'])
        c.append(i['Date'])
    df=pd.DataFrame()
    df['Id']=a
    df['date']=c
    df['Name']=b
    df.to_csv('Attendence1.csv')
    data = open('Attendence1.csv')
    
# write dataframe to excel
    
# save the excel
    
    data = open('Attendence1.csv')
    
    response = HttpResponse(data, content_type="csv")
    response['Content-Disposition'] = 'attachement; filename=Attendence1' 
    print('t')
    return response