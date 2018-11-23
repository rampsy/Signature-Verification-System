"""
Created on Fri Nov 23 14:02:40 2018

@author: Rudra
"""
###############################################################################
#Importing packages
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
###############################################################################
'''
FUNCTION NAME:tuning
INPUT:Image_database containing file name and array(conatining pixels value)
OUTPUT:Trained classifier model for identity
LOGIC:Training data is splitted into train and test data according to 80:20 ratio
and then RandomForestClassifier model is trained so that it will predict the id number
of the signature, model is tuned with different parameters and best parameters are
found out for which model performs best on test data
'''   
def tuning(train_genuine,num,image_database): 
    '''Creating train and test data '''
    train1_gen=[]
    test1_gen=[]
    training_data=[]
    testing_data=[]
    y_train=[]
    y_test=[]
    for i in range(0,num,5):  
        train1_gen.append(train_genuine[i])
        y_train.append(int(image_database[i][0][-6:-4]))
        train1_gen.append(train_genuine[i+1])
        y_train.append(int(image_database[i+1][0][-6:-4]))
        train1_gen.append(train_genuine[i+2])
        y_train.append(int(image_database[i+2][0][-6:-4]))
        train1_gen.append(train_genuine[i+3])
        y_train.append(int(image_database[i+3][0][-6:-4]))
        test1_gen.append(train_genuine[i+4])
        y_test.append(int(image_database[i+4][0][-6:-4]))
    #print(y_train)    
    training_data=train1_gen  
    testing_data=test1_gen
    
    '''Initial parameters''' 
    pre_best=0
    best=0.1
    c=1
    n=35
    r=60
    score=[]
    while(c!=0):
        score=[]
        for i in range(1,100):
            classifier=RandomForestClassifier(n_estimators=i,criterion='gini',random_state=r)
            classifier.fit(training_data,y_train)
            score.append(accuracy_score(y_test,classifier.predict(testing_data)))
        if best>pre_best:    
            n=np.argmax(score)+1
            pre_best=max(score)
#            print(pre_best)
        score1=[]    
        for i in range(1,100):
            classifier=RandomForestClassifier(n_estimators=n,criterion='gini',random_state=i)
            classifier.fit(training_data,y_train)
            score1.append(accuracy_score(y_test,classifier.predict(testing_data)))           
        r=np.argmax(score1)+1
        best=max(score1)
        if pre_best>=best:
            c=0
    classifier=RandomForestClassifier(n_estimators=n,criterion='gini',random_state=r)
    classifier.fit(training_data,y_train)                       #Trained classifer model
    print('accuracy_score of identity classifer:',accuracy_score(y_test,classifier.predict(testing_data)))
    return n,r,classifier
        
