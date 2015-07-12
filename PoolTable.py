# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:32:26 2015

@author: Stuart Grieve
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import Indexer

def Test():
    
    img = LoadImage('img/pool_crop_2.png')
    
    hsv = ToHSV(img)    
    
    lower_color, upper_color = GetClothColor(hsv)
    
    contours = GetContours(hsv, lower_color, upper_color,7)
    
    TableContour = MaskTableBed(contours)
    
    warp = TransformToOverhead(img,TableContour)
    
    #Now the table is cropped and warped, lets find the balls
    hsv = ToHSV(warp)
    
    lower_color, upper_color = GetClothColor(hsv)    
    
    contours = GetContours(hsv, lower_color, upper_color,17)
        
    BallData = FindTheBalls(warp, contours)
    print lower_color
    CueBall(BallData)    
    
def CueBall(BallData):    
    
    data = BallData[1][2]


    #this mask does not reflect the boundary between data and nodata.
    mask = cv2.inRange(data, (0,0,10), (180,255,255))
       
#    cv2.imshow('result1',mask)
#    cv2.imshow('result',data)
#
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    
    hist = cv2.calcHist([data], [0], mask, [180], [0, 180])
    
    plt.plot(hist)
    plt.show()
    
    hist = cv2.calcHist([data], [1], mask, [256], [0, 256])
    
    plt.plot(hist)
    plt.show()
    
    hist = cv2.calcHist([data], [2], mask, [256], [0, 256])
    
    plt.plot(hist)
    plt.show()


def LoadImage(filename):
    """
    Loads an image file
    """
    #img is loaded in bgr colorspace
    return cv2.imread(filename)

def ToHSV(img):
    """
    Convert an image from BGR to HSV colorspace
    """
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    

def GetClothColor(hsv,search_width=45):
    """
    Find the most common HSV values in the image.
    In a well lit image, this will be the cloth
    """

    hist = cv2.calcHist([hsv], [1], None, [180], [0, 180])
    h_max = Indexer.get_index_of_max(hist)[0]
    
    hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    s_max = Indexer.get_index_of_max(hist)[0]
    
    hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    v_max = Indexer.get_index_of_max(hist)[0]

    # define range of blue color in HSV
    lower_color = np.array([h_max-search_width,s_max-search_width,v_max-search_width])
    upper_color = np.array([h_max+search_width,s_max+search_width,v_max+search_width])

    return lower_color, upper_color


def MaskTableBed(contours):
    """
    Mask out the table bed, assuming that it will be the biggest contour.
    """
            
    #The largest area should be the table bed    
    areas = []    
    for c in contours:
        areas.append(cv2.contourArea(c))
    
    #return the contour that delineates the table bed
    largest_contour = Indexer.get_index_of_max(areas)
    return contours[largest_contour[0]]

def distbetween(x1,y1,x2,y2):
    """
    Compute the distance between points (x1,y1) and (x2,y2)
    """

    return np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

def Get_UL_Coord(contour,pad=10):
    """
    Get the upper left coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(distbetween(c[0][0],c[0][1],0,0))

    return (contour[Indexer.get_index_of_min(dists)[0]][0][0]-pad,contour[Indexer.get_index_of_min(dists)[0]][0][1]-pad)
    
def Get_UR_Coord(contour,imgXmax, pad=10):
    """
    Get the upper right coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(distbetween(c[0][0],c[0][1],imgXmax,0))

    return (contour[Indexer.get_index_of_min(dists)[0]][0][0]+pad,contour[Indexer.get_index_of_min(dists)[0]][0][1]-pad)

def Get_LL_Coord(contour,imgYmax, pad=10):
    """
    Get the lower left coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(distbetween(c[0][0],c[0][1],0,imgYmax))

    return (contour[Indexer.get_index_of_min(dists)[0]][0][0]-pad,contour[Indexer.get_index_of_min(dists)[0]][0][1]+pad)
    
def Get_LR_Coord(contour,imgXmax,imgYmax, pad=10):    
    """
    Get the lower right coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(distbetween(c[0][0],c[0][1],imgXmax,imgYmax))

    return (contour[Indexer.get_index_of_min(dists)[0]][0][0]+pad,contour[Indexer.get_index_of_min(dists)[0]][0][1]+pad)

def TransformToOverhead(img,contour):
    """
    Get the corner coordinates of the table bed by finding the minumum
    distance to the corners of the image for each point in the contour.
    
    Transform code is built upon code from: http://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/ 
    """

    #get dimensions of image
    height, width, channels = img.shape 

    #find the 4 corners of the table bed
    UL = Get_UL_Coord(contour)
    UR = Get_UR_Coord(contour,width)  
    LL = Get_LL_Coord(contour,height)  
    LR = Get_LR_Coord(contour,width,height)  
    
    #store the coordinates in a numpy array    
    rect = np.zeros((4, 2), dtype = "float32")
    rect[0]= [UL[0],UL[1]]
    rect[1]= [UR[0],UR[1]]
    rect[2]= [LR[0],LR[1]]
    rect[3]= [LL[0],LL[1]]
    
    #get the width at the bottom and top of the image
    widthA = distbetween(LL[0],LL[1],LR[0],LR[1])
    widthB = distbetween(UL[0],UL[1],UR[0],UR[1])
    
    #choose the maximum width 
    maxWidth = max(int(widthA), int(widthB))
    maxHeight  = (maxWidth*2) #pool tables are twice as long as they are wide
    
    # construct our destination points which will be used to
    # map the image to a top-down, "birds eye" view
    dst = np.array([
    	[0, 0],
    	[maxWidth - 1, 0],
    	[maxWidth - 1, maxHeight - 1],
    	[0, maxHeight - 1]], dtype = "float32")
     
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))    
    
    return warp    

def GetContours(hsv, lower_color, upper_color,filter_radius):
    """
    Returns the contours generated from the given color range
    """
    
    # Threshold the HSV image to get only cloth colors
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    #use a median filter to get rid of speckle noise
    median = cv2.medianBlur(mask,filter_radius)
    
    #get the contours of the filtered mask
    #this modifies median in place!
    _, contours, _ = cv2.findContours(median,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return contours

def FindTheBalls(img, contours, similarity_threshold=5):
    """
    Find and circle all of the balls on the table.
    
    Currently struggles with balls on the rail. Not yet tested on clusters.
    
    Returns a three-tuple containing a tuple of x,y coords, a radius and the masked
    out area of the image. Needs to be made into a ball object.
    """

    #dimensions of image
    height,width, channels = img.shape

    #compare the difference in area of a min bounding circle and the cotour area
    diffs = []
    indexes = []
    
    for i,contour in enumerate(contours):
        contourArea = cv2.contourArea(contour)
        (x,y),radius = cv2.minEnclosingCircle(contour)
        
        circleArea = 3.141 * (radius**2)
        diffs.append(abs(circleArea-contourArea))
        indexes.append(i)
        
    sorted_data = sorted(zip(diffs,indexes))
    
    diffs = [x[0] for x in sorted_data]
    indexes = [x[1] for x in sorted_data]
    
    #list of center coords as tuples
    centers = []    
    radii = []
    masks = []
    for i,d in zip(indexes,diffs):
        #if the contour is a similar shape to the circle it is likely to be a ball.
        if (d < diffs[0] * similarity_threshold):
            (x,y),radius = cv2.minEnclosingCircle(contours[i])
        
            center = (int(x),int(y))
            radius = int(radius)
            #remove .copy() to display a circle round each ball
            cv2.circle(img.copy(),center,radius,(0,0,255),2)
            centers.append(center)
            radii.append(radius)
            
            circle_img = np.zeros((height,width), np.uint8)
            cv2.circle(circle_img,center,radius,1,thickness=-1)
            masked_data = cv2.bitwise_and(img, img, mask=circle_img)    
            masks.append(masked_data)

    
    return zip(centers,radii,masks)

    
Test()
