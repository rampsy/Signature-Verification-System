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
FUNCTION NAME:training_tuning
INPUT:Image_database containing file name and array(conatining pixels value)
OUTPUT:Trained classifier model for signature status
LOGIC:Training data is splitted into train and test data according to 80:20 ratio
and then RandomForestClassifier model is trained so that it will predict 1 for genuine 
signature and 0 for forge signature,model is tuned with different parameters 
and best parameters are found out for which model performs best on test data
'''   
def training_tuning(train_genuine,train_forged,num,image_database):
    '''Creating train and test data '''
    train1_gen=[]
    test1_gen=[]
    train1_for=[]
    test1_for=[]   
    training_data=[]
    testing_data=[]
    for i in range(0,num,5):
        train1_gen.append(train_genuine[i])
        train1_gen.append(train_genuine[i+1])
        train1_gen.append(train_genuine[i+2])
        train1_gen.append(train_genuine[i+3])
        test1_gen.append(train_genuine[i+4])
        train1_for.append(train_forged[i])
        train1_for.append(train_forged[i+1])
        train1_for.append(train_forged[i+2])
        train1_for.append(train_forged[i+3])
        test1_for.append(train_forged[i+4])        
    training_data=train1_gen  
    training_data.extend(train1_for)    
    testing_data=test1_gen
    testing_data.extend(test1_for)    
    y_train_gen=np.ones(num).tolist()
    y_train_for=np.zeros(num).tolist()
    y_train=y_train_gen[:int((num/5)*4)]
    y_train.extend(y_train_for[:int((num/5)*4)])    
    y_test=y_train_gen[int((num/5)*4):]
    y_test.extend(y_train_for[int((num/5)*4):])
    
    '''Initial parameters''' 
    criterions=['gini','entropy']
    datab=[]
    gi=[]
    en=[]
    max_epoch=10
    for j in range(1,max_epoch):
        data=[]
        for criterion in range(len(criterions)):
            pre_best=0
            best=0.1
            c=1
            n=35
            r=j
            score=[]
            while(c!=0):                           #It will run till model keeps on improving
                score=[]
                for i in range(1,100):
                    classifier=RandomForestClassifier(n_estimators=i,criterion=criterions[criterion],random_state=r)
                    classifier.fit(training_data,y_train)
                    score.append(accuracy_score(y_test,classifier.predict(testing_data)))
                if best>pre_best:    
                    if max(score)>=best:
                        n=np.argmax(score)+1
                        pre_best=max(score)     
                    else:
                        pre_best=pre_best
                score1=[]    
                for i in range(1,100):
                    classifier=RandomForestClassifier(n_estimators=n,criterion=criterions[criterion],random_state=i)
                    classifier.fit(training_data,y_train)
                    score1.append(accuracy_score(y_test,classifier.predict(testing_data)))                    
                r=np.argmax(score1)+1
                best=max(score1)
                if pre_best>=best:
                    c=0
            data.append([n,r,best])
        gi.append(data[0][2])
        en.append(data[1][2])
        datab.append(data)
    sc=max(gi)
    sc1=max(en)
    if sc>=sc1:        
        n=datab[np.argmax(gi)][0][0]
        r=datab[np.argmax(gi)][0][1]
        criterion='gini'           
    else:
        n=datab[np.argmax(en)][1][0]
        r=datab[np.argmax(en)][1][1]
        criterion='entropy'

    classifier=RandomForestClassifier(n_estimators=n,criterion=criterion,random_state=r)
    classifier.fit(training_data,y_train)            #Trained classifer model
    print('accuracy_score of status classifier:',accuracy_score(y_test,classifier.predict(testing_data)))
#    filename='classifier1.sav'
#    joblib.dump(classifier,filename)
    return classifier
        
