# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:53:21 2020

@author: raymond
"""


# import the necessary packages
import numpy as np
import cv2



def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        pt = tuple(refPt[0])
        cv2.circle(img_4pt,pt, 3, (0,255,255), -1)
        cv2.putText(img_4pt,'{}'.format(len(refPts)+1),pt, cv2.FONT_HERSHEY_SIMPLEX, 2, (200,255,155),5) #---write the text
        refPts.append(refPt[0])
        cv2.imshow("image with irregular quadrilateal", img_4pt)
        print(refPt)
        
        
        
def sample4pt(image):
    
    cv2.namedWindow("image with irregular quadrilateal",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image with irregular quadrilateal", click)
    cv2.putText(image,'Sample 4 points',(0,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (200,255,155),5) #---write the text
    
    cv2.imshow("image with irregular quadrilateal", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.polylines(image, np.array([refPts]),True, (0,0,255),5)
    
    
    
    
    
if __name__ == '__main__':
    refPts = []
    
    raw_img = cv2.imread('img/highway.jpg', cv2.IMREAD_COLOR)
    #raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_4pt = raw_img.copy()
    

    sample4pt(img_4pt)
    pts = np.array(refPts,dtype = "int")
    

    
    cv2.namedWindow("image with irregular quadrilateal",cv2.WINDOW_NORMAL)
    cv2.imshow("image with irregular quadrilateal", img_4pt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
