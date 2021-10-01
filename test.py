import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter import *
from tkinter import messagebox
from tkinter import PhotoImage


#Creating root window
root=Tk()
root.geometry("800x400")
# bg=PhotoImage(file = "background/b.png")
# label3 = Label( root, image = bg)
# label3.place(x = 0, y = 0)


root.title("Face Recognition Project")
my_menu=Menu(root)
root.config(menu=my_menu)
root.iconbitmap("face_recognition_icon_155419.ico")

#about me
def details():
    messagebox.showinfo("Project","Made By : \nSreeparno Dhar \nKumar Gaurav \nSourabh Kumar \nSubham Kumar Giri")

#Create menu item
about_menu=Menu(my_menu, tearoff=0)
my_menu.add_cascade(label="More", menu=about_menu)
about_menu.add_command(label="About",command=details)
about_menu.add_separator()
about_menu.add_command(label="Exit", command=root.quit)


#algo
def face_embeding(images):
    encodeList=[]


    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


#attendance
def markAttendance(name):
    with open('Attendance.csv', "r+") as f:
        dataset=f.readlines()
        nameList=[]
        for i in dataset:
            a=i.split(',')
            nameList.append(a[0])
        if name not in nameList:
            now=datetime.now()
            tString=now.strftime('%H:%M:%S')
            dtString=now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tString},{dtString}')


def operation():
    path = 'Images'
    image=[]
    classNames=[]
    my_list=os.listdir(path)
    # print(my_list)
    for i in my_list:
        cur_img=cv2.imread(f'{path}/{i}')
        image.append(cur_img)
        classNames.append(os.path.splitext(i)[0])
    # print(classNames)








    embed_known=face_embeding(image)
    messagebox.showinfo('Face Embeding Complete !')

    cap=cv2.VideoCapture(0)

    while True:
        ret, frame=cap.read()
        if frame is None:
            print('Wrong path:', path)
        else:
            imgS=cv2.resize(frame, dsize=(350,350))
        imgS=cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame=face_recognition.face_locations(imgS)
        encodesCurFrame=face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, (x, y, w, h) in zip(encodesCurFrame, facesCurFrame):
            match=face_recognition.compare_faces(embed_known, encodeFace)
            face_distance=face_recognition.face_distance(embed_known, encodeFace)

            matchIndex=np.argmin(face_distance)

            if match[matchIndex]:
                name=classNames[matchIndex].title()
                print(name)

                # cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x+h, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                markAttendance(name)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1)==13:
            break


    cap.release()
    cv2.destroyAllWindows()


label1=Label(root,text="ATTENDANCE SYSTEM USING FACE RECOGNITION", width=60, fg="blue", font=("bold", 15))
label1.place(x=80, y=80)
Button(root,text="Start !", width=20, height=2, bg="red", fg="white",command=operation).place(x=320,y=200)


root.mainloop()