"""
Created on Fri Nov 23 14:02:40 2018

@author: Rudra
"""
###############################################################################
#Importing packages
import numpy as np
import cv2
from sklearn.externals import joblib #For loading saved model
import feature_extract               #Python code containing code for feature extraction from image
from tkinter import filedialog       #For opening file explorer
from tkinter import font             #For changing font of the words
from tkinter import *
import os
from PIL import Image, ImageTk
import identity_trail                #For training classifier for identity
import classifier_model              #For training classifier for identifying the status of the signature
import pandas as pd
import time
###############################################################################
root=Tk()                                             #For blank window
root.title('GUI FOR SIGNATURE VERIFICATION SYSTEM')   #For giving title to the window

fname = Canvas(root,height=200,width=200,bg='white')  #Canvas for displaying claimed signature
fname.grid(row=9,column=0)

fname1 = Canvas(root,height=200,width=200,bg='white') #Canvas for displaying original signature of claimed identity
fname1.grid(row=9,column=1,sticky=W)

frame1=Frame(root,width=90,height=20,bg='white')      #Frame for displaying the status of the signature
frame1.grid(row=9,column=2,sticky=W)

try:
    c1=joblib.load('classifier2.sav')                 #Loading classifier model for identity
    c2=joblib.load('classifier1.sav')                 #Loading classifier model for status of the signature
except:
    print('Identity classifier file is not present')  #Error message if classifier model is not present
###############################################################################
Database=[]
l=pd.read_csv('directory.csv')                        #Reading current directory name
if os.path.exists(np.array(l).tolist()[0][0]):
    Database.append(np.array(l).tolist()[0][0])
else:
    Database.append(os.getcwd()+'\Dataset\dataset4/real1')
###############################################################################
'''
FUNCTION NAME:verify
INPUT:NONE
OUTPUT:Status of the identity of the claimed signature
LOGIC:Two seperate model is trained independent of each other one for identity
and another for status of the signature and both the model are tuned such that 
model gives max accuracy_score on test case and then both the model is used for 
finding out the status of the claimed signature.
'''      
def verify():    
    try:
        frame2=Frame(frame1,width=90,height=20,bg='white')
        frame2.grid(row=9,column=2,sticky=W)
        root.filename=filedialog.askopenfilename(initialdir="/",title='select file',\
                                                 filetypes=(('PNG','*.png'),('all files','*.*')))
        file=root.filename                                      #Selected file
        img=cv2.imread(file)
        image=[(file,img)]
        feature=feature_extract.extract(image)
        genuine=c2.predict(np.array(feature).reshape(1,-1))
        identity=c1.predict(np.array(feature).reshape(1,-1))
#        print(genuine)
#        print(identity)
        pil_image = Image.open(file)
        pil_image = pil_image.resize((200, 200), Image.ANTIALIAS)       
        image = ImageTk.PhotoImage(pil_image)
        fname.image=image
        fname.create_image((0,0), image=image, anchor=NW)
        claimed_id=entry.get()                                  #Claimed identity
        cudi=Database[0]
        di=os.listdir(cudi)
        im=[]
        for i_d in range(len(di)):
            if int(claimed_id)== int(di[i_d][-6:-4]):
                im.append(di[i_d])
                break
        pil_image1 = Image.open(os.path.join(cudi,im[0]))
        pil_image1 = pil_image1.resize((200, 200), Image.ANTIALIAS)
        image1 = ImageTk.PhotoImage(pil_image1)
        fname1.image=image1
        fname1.create_image((0,0), image=image1, anchor=NW)
        
        if int(claimed_id)!=identity[0]:
            label8=Label(frame2,text='NOT MATCHED')
            label8.grid(row=9,column=2)
         
        else:
            if int(genuine[0])==1:
                label8=Label(frame2,text='MATCHED')
                label8.grid(row=9,column=2)                  
            else:
                label8=Label(frame2,text='NOT MATCHED')
                label8.grid(row=9,column=2)  
    except:
        print('CLAIMED ID NUMBER NOT FOUND IN DATABASE')  
        label8=Label(frame2,text='         CLAIMED ID NUMBER NOT FOUND \n \
        IN DATABASE OR NO FILE IS SELECTED')
        label8.grid(row=9,column=2) 
###############################################################################
        
###############################################################################
'''
FUNCTION NAME:change_directory
INPUT:NONE
OUTPUT:New identityclassifier file and statusclassifier file are trained on the selected database
LOGIC:RandomForestClassifier model is trained on new database to identify the 
identity of the signatures and the status of the signature, and then model is 
tuned by taking accuracy_score as a measure and changing n_estimators and Random_state 
values and best value is found out for which model performs best on test case.
'''      
def change_directory():
    Database.clear()
    try:
        frame4=Frame(frame3,width=90,height=20,bg='white')
        frame4.grid(row=17,column=2)
        label14=Label(frame4,text='PROCESSING....')
        label14.grid(row=17,column=2)
        filename = filedialog.askdirectory()
        list1=[filename]
        df=pd.DataFrame(data={"col1":list1})
        df.to_csv("directory.csv",sep=',',index=False)         #saving current directory into csv file
        filename1 = filedialog.askdirectory()
        num=len(os.listdir(filename))
        #num=len(os.listdir(filename))
        #print(num)
        print('Training identity classifier')
        genuine_image_database=feature_extract.load_images_from_folder(filename)
        genuine_feature_database=feature_extract.extract(genuine_image_database)
        n,r,clas=identity_trail.tuning(genuine_feature_database,num,genuine_image_database)
        print('Training status classifier')
        forge_image_database=feature_extract.load_images_from_folder(filename1)
        forge_feature_database=feature_extract.extract(forge_image_database)
        clas1=classifier_model.training_tuning(genuine_feature_database,forge_feature_database,num,forge_image_database)
        frame5=Frame(frame4,width=90,height=20,bg='white')
        frame5.grid(row=17,column=2)
        label14=Label(frame5,text='......DONE......')
        label14.grid(row=17,column=2)
        time.sleep(3)
        label14=Label(frame5,text='Please run the file once again')
        label14.grid(row=17,column=2)
        filen='classifier2.sav'                  #Status classifier
        joblib.dump(clas,filen)        
        filen1='classifier1.sav'
        joblib.dump(clas1,filen1)                #Identity classifier
        Database.append(filename)
        label15=Label(frame6,text=Database[0])    
        label15.grid(row=15,column=1,sticky=E)
        time.sleep(5)
        print('Please run the file once again')
        root.destroy()
    except:
        print('NO FOLDER OR WRONG FOLDER IS SELECTED')
###############################################################################        
        
###############################################################################    
'''
FUNCTION NAME:leave
INPUT:NONE
OUTPUT:NONE
LOGIC:IT Simply destroy or closes the windows/GUI
'''      
def leave():
    print('THANK YOU')    
    root.destroy()
###############################################################################    

###############################################################################
#BELOW CODE WILL MAKE THE GUI AS PER THE REQUIREMENT 
font=font.Font(family='Helvetica',size=10,weight='bold')       #It assign the font variable with the information of the font.
label1=Label(root,text='GUI FOR SIGNATURE VERIFICATION SYSTEM')#This is the label which gives the idea about what 
                                                               
label1['font']=font                                            #It set the font of the letters displayed in the GUI
label1.grid(row=0,column=0)

frame=Frame(root,width=15,height=15)
frame.grid(row=1,column=0)

label2=Label(root,text='VERIFY SIGNATURES:',fg='red')    
label2.grid(row=2,column=0)

frame=Frame(root,width=15,height=15)
frame.grid(row=3,column=0)

label3=Label(root,text='CLAIMED ID NUMBER:')    
label3.grid(row=4,column=0,sticky=E)

entry=Entry(root)
entry.grid(row=4,column=1,sticky=W)

frame=Frame(root,width=15,height=15)
frame.grid(row=5,column=0)

label4=Label(root,text='VERIFY:')    
label4.grid(row=6,column=0,sticky=E)

button1=Button(root,text='click to choose and verify',command=verify)
button1.grid(row=6,column=1,sticky=W)

frame=Frame(root,width=15,height=15)
frame.grid(row=7,column=0)

label5=Label(root,text='CLAIMED SIGNATURE:')    
label5.grid(row=8,column=0)

label6=Label(root,text='SIGNATURE IN DATABASE:')    
label6.grid(row=8,column=1)

label7=Label(root,text='STATUS:')    
label7.grid(row=8,column=2)

frame=Frame(root,width=15,height=15)
frame.grid(row=10,column=0)

label9=Label(root,text='DO YOU WANT TO CLOSE THE GUI, PRESS THE QUIT BUTTON:')    
label9.grid(row=11,column=0,sticky=E)

button1=Button(root,text='QUIT',command=leave)
button1.grid(row=11,column=1,sticky=W)

frame=Frame(root,width=15,height=15)
frame.grid(row=12,column=0)

label10=Label(root,text='WANNA CHANGE THE DATABASE DIRECTORY:',fg='red')    
label10.grid(row=13,column=0)

frame=Frame(root,width=15,height=15)
frame.grid(row=14,column=0)

label11=Label(root,text='CURRENT DIRECTORY:')    
label11.grid(row=15,column=0,sticky=E)

frame6=Frame(root,width=60,height=15)
frame6.grid(row=15,column=1)

label12=Label(frame6,text=Database[0])    
label12.grid(row=15,column=1,sticky=E)

frame=Frame(root,width=15,height=15)
frame.grid(row=16,column=0)

label13=Label(root,text='CHOOSE DIRECTORY:\n (FIRST GENUINE FOLDER THAN FORGED FOLDER)')    
label13.grid(row=17,column=0,sticky=E)

button2=Button(root,text='CLICK',command=change_directory)
button2.grid(row=17,column=1,sticky=W)

frame3=Frame(root,width=90,height=20,bg='white')
frame3.grid(row=17,column=2)

root.mainloop()

###############################################################################
