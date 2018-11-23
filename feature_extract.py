"""
Created on Fri Nov 23 14:02:40 2018

@author: Rudra
"""
###############################################################################
#Importing packages
import mahotas
import cv2
import os
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from skimage.filters import threshold_otsu   # For finding the threshold for grayscale to binary conversion
import sys
import warnings
###############################################################################
#Ignoring deprecation warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
###############################################################################  
'''Functions for calculating Important features from image'''
def fd_hu_moments(gray):
    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    return feature
# feature-descriptor-2: Haralick Texture
def fd_haralick(gray):
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def EccentricitySolidity(img):
    r = regionprops(img.astype("int8"))
    return r[0].eccentricity, r[0].solidity

def SkewKurtosis(img):
    h,w = img.shape
    x = range(w)  # cols value
    y = range(h)  # rows value
    #calculate projections along the x and y axes
    xp = np.sum(img,axis=0)
    yp = np.sum(img,axis=1)
    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)
    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2
    sx = np.sqrt(np.sum(x2*xp)/np.sum(img))
    sy = np.sqrt(np.sum(y2*yp)/np.sum(img))    
    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3
    skewx = np.sum(xp*x3)/(np.sum(img) * sx**3)
    skewy = np.sum(yp*y3)/(np.sum(img) * sy**3)
    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp*x4)/(np.sum(img) * sx**4) - 3
    kurty = np.sum(yp*y4)/(np.sum(img) * sy**4) - 3
    return (skewx , skewy), (kurtx, kurty)

def get_contour_features(im):
    rect = cv2.minAreaRect(cv2.findNonZero(im))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])
    aspect_ratio = max(w, h) / min(w, h)
    bounding_rect_area = w * h
    im2, contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = 0
    for cnt in contours:
        contour_area += cv2.contourArea(cnt)
    return aspect_ratio, bounding_rect_area, contour_area

def centroid(im):
    M1 = cv2.moments(im) 
# calculate x,y coordinate of center
    cX1 = int(M1["m10"] / M1["m00"])
    cY1 = int(M1["m01"] / M1["m00"])
    return cX1,cY1

def angle_between(p1,p2):
    ang1=np.arctan2(*p1[::-1])
    ang2=np.arctan2(*p2[::-1])
    return np.rad2deg((ang1-ang2)%(2*np.pi))

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append((filename,img))
    return images
###############################################################################
    
###############################################################################
'''Function which will extract all important features from images using above functions'''
def extract(file):
    train_list=[]
    for i in range(len(file)):
        img=file[i][1]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           #(y,x) is the coordnation of the gray
        thres = threshold_otsu(gray)
        ret,thresh = cv2.threshold(gray,thres,255,cv2.THRESH_BINARY)   
        c1,c2=centroid(gray)
        r, c = np.where(thresh==0)
        # Now we will make a bounding box with the boundary as the position of pixels on extreme.
        # Thus we will get a cropped image with only the signature part.
        signimg = thresh[r.min(): r.max(), c.min(): c.max()]
        height,width=signimg.shape
        aspect_ratio, bounding_rect_area, contour_area=get_contour_features(signimg)
        
        sign1=signimg[:,:int(width/2)]
        sign2=signimg[:,int(width/2):]
        ce1=ndimage.measurements.center_of_mass(signimg)
        ce2=ndimage.measurements.center_of_mass(sign1)
        ce3=ndimage.measurements.center_of_mass(sign2)
        baseline_shift=ce2[0]-ce3[0]
        
        cr=ndimage.measurements.center_of_mass(signimg)
        
        i1=signimg[0:int(cr[0]),0:int(cr[1])]
        i2=signimg[int(cr[0]):,0:int(cr[1])]
        i3=signimg[0:int(cr[0]),int(cr[1]):]
        i4=signimg[int(cr[0]):,int(cr[1]):]
        
        cr11=ndimage.measurements.center_of_mass(i1)
        i11=signimg[0:int(cr11[0]),0:int(cr11[1])]
        i12=signimg[int(cr11[0]):int(cr[0]),0:int(cr11[1])]
        i13=signimg[0:int(cr11[0]),int(cr11[1]):int(cr[1])]
        i14=signimg[int(cr11[0]):int(cr[0]),int(cr11[1]):int(cr[1])]
        
        cr12=ndimage.measurements.center_of_mass(i3)
        i21=signimg[0:int(cr12[0]),int(cr[1]):int(cr[1])+int(cr12[1])]
        i22=signimg[int(cr12[0]):int(cr[0]),int(cr[1]):int(cr[1])+int(cr12[1])]
        i23=signimg[0:int(cr12[0]),int(cr12[1])+int(cr[1]):]
        i24=signimg[int(cr12[0]):int(cr[0]),int(cr[1])+int(cr12[1]):]
        
        cr13=ndimage.measurements.center_of_mass(i2)
        i31=signimg[int(cr[0]):int(cr[0])+int(cr13[0]),0:int(cr13[1])]
        i32=signimg[int(cr13[0])+int(cr[0]):,0:int(cr13[1])]
        i33=signimg[int(cr[0]):int(cr[0])+int(cr13[0]),int(cr13[1]):int(cr[1])]
        i34=signimg[int(cr13[0])+int(cr[0]):,int(cr13[1]):int(cr[1])]
        
        cr14=ndimage.measurements.center_of_mass(i4)
        i41=signimg[int(cr[0]):int(cr[0])+int(cr14[0]),int(cr[1]):int(cr[1])+int(cr14[1])]
        i42=signimg[int(cr14[0])+int(cr[0]):,int(cr[1]):int(cr[1])+int(cr14[1])]
        i43=signimg[int(cr[0]):int(cr[0])+int(cr14[0]),int(cr14[1])+int(cr[1]):]
        i44=signimg[int(cr14[0])+int(cr[0]):,int(cr14[1])+int(cr[1]):]
        
        s1=ndimage.measurements.center_of_mass(i11)
        s2=ndimage.measurements.center_of_mass(i12)
        s3=ndimage.measurements.center_of_mass(i13)
        s4=ndimage.measurements.center_of_mass(i14)
        s5=ndimage.measurements.center_of_mass(i21)
        s6=ndimage.measurements.center_of_mass(i22)
        s7=ndimage.measurements.center_of_mass(i23)
        s8=ndimage.measurements.center_of_mass(i24)
        s9=ndimage.measurements.center_of_mass(i31)
        s10=ndimage.measurements.center_of_mass(i32)
        s11=ndimage.measurements.center_of_mass(i33)
        s12=ndimage.measurements.center_of_mass(i34)
        s13=ndimage.measurements.center_of_mass(i41)
        s14=ndimage.measurements.center_of_mass(i42)
        s15=ndimage.measurements.center_of_mass(i43)
        s16=ndimage.measurements.center_of_mass(i44)
        
        t1=s1
        t2=(s2[0]+cr11[0],s2[1])
        t3=(s3[0],cr11[1]+s3[1])
        t4=(cr11[0]+s4[0],cr11[1]+s4[1])
        t5=(s5[0],cr[1]+s5[1])
        t6=(cr12[0]+s6[0],cr[1]+s6[1])
        t7=(s7[0],cr12[1]+s7[1])
        t8=(cr12[0]+s8[0],cr[1]+s8[1]+cr12[1])
        t9=(cr[0]+s9[0],s9[1])
        t10=(cr[0]+cr13[0]+s10[0],s10[1])
        t11=(cr[0]+s11[0],s11[1]+cr13[1])
        t12=(cr[0]+cr13[0]+s12[0],cr13[1]+s12[1])
        t13=(cr[0]+s13[0],cr[1]+s13[1])
        t14=(cr[0]+s14[0],cr14[1]+s14[1]+cr[1])
        t15=(cr[0]+cr14[0]+s15[0],cr[1]+s15[1])
        t16=(cr[0]+cr14[0]+s16[0],cr[1]+s16[1]+cr14[1])

        s=[]
        s.append(t1)
        s.append(t2)
        s.append(t3)
        s.append(t4)
        s.append(t5)
        s.append(t6)
        s.append(t7)
        s.append(t8)
        s.append(t9)
        s.append(t10)
        s.append(t11)
        s.append(t12)
        s.append(t13)
        s.append(t14)
        s.append(t15)
        s.append(t16)
                
        angle=[]
        for i in range(0,16):
            angle.append(angle_between(s[i],s[15]))
 
        feature=[]
        for i in range(0,16):
            feature.append(angle[i])
   
        fe=fd_hu_moments(signimg)
        ha=fd_haralick(signimg)    
        signimg=np.invert(np.logical_not(signimg))
        es1,es2=EccentricitySolidity(signimg)
        sk1,sk2=SkewKurtosis(signimg)

        signimg=signimg*255
        signimg=signimg.astype(np.uint8)

        feature.append(height)
        feature.append(width)
        feature.append(c1)
        feature.append(c2)
        feature.append(aspect_ratio)
        feature.append(bounding_rect_area)
        feature.append(contour_area)
        feature.append(ce1[0])
        feature.append(ce1[1])
        feature.append(baseline_shift)
        feature.append(es1)
        feature.append(es2)
        feature.append(sk1[0])
        feature.append(sk1[1])
        feature.append(sk2[0])
        feature.append(sk2[1])
        
        for i in range(13):
            feature.append(ha[i])
        for i in range(7):
            feature.append(fe[i])
            
        train_list.append(feature)
    return train_list    #Extracted features




