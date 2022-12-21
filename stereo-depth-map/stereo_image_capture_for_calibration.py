#!/usr/bin/env python

#Usage - Open checkboard image with atleast 6 by 8 checkerboard image
#connect the camera, point the camera at the cheakerboard image. take a picture bby pressing 'p' when you see the lines.

import cv2
import numpy as np
import os
import glob

import sys,time

# init camera
#camera = cv2.VideoCapture(1) ### <<<=== SET THE CORRECT CAMERA NUMBER
cameraL = cv2.VideoCapture("/dev/video2") ### Left USB Camera
cameraR = cv2.VideoCapture("/dev/video4") ### Left USB Camera


# Defining the dimensions of checkerboard
CHECKERBOARD = (6,8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # Creating vector to store vectors of 3D points for each checkerboard image
# objpoints = []
# # Creating vector to store vectors of 2D points for each checkerboard image
# imgpoints = [] 


# Defining the world coordinates for 3D points
objpL = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objpL[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpR = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objpR[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

prev_img_shape = None

# Extracting path of individual image stored in a given directory
# images = glob.glob('./images/*.jpg')

counter=0
file_save_counter=0

while 1:

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpointsL = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpointsL = [] 


    # Creating vector to store vectors of 3D points for each checkerboard image
    objpointsR = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpointsR = [] 

    # (grabbed_left,fname) = camera_left.read()
    (grabbedL,imgL) = cameraL.read()
    img_cL = imgL.copy()

    (grabbedR,imgR) = cameraR.read()
    img_cR = imgR.copy()


    # for fname in images:
    # img = cv2.imread(grabbed_left)
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if retL == True and retR == True:
        # counter=counter+1

        # if counter > 100:
        #     break


        objpointsL.append(objpL)
        # refining pixel coordinates for given 2d points.
        corners2L = cv2.cornerSubPix(grayL, cornersL, (11,11),(-1,-1), criteria)
        
        imgpointsL.append(corners2L)

        objpointsR.append(objpR)
        # refining pixel coordinates for given 2d points.
        corners2R = cv2.cornerSubPix(grayR, cornersR, (11,11),(-1,-1), criteria)
        
        imgpointsR.append(corners2R)

        # Draw and display the corners
        img_cL = cv2.drawChessboardCorners(img_cL, CHECKERBOARD, corners2L, retL)
        img_cR = cv2.drawChessboardCorners(img_cR, CHECKERBOARD, corners2R, retR)

        hL,wL = img_cL.shape[:2]
        hR,wR = img_cR.shape[:2]

        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpointsL, imgpointsL, grayL.shape[::-1], None, None)

        retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpointsR, imgpointsR, grayR.shape[::-1], None, None)

        # print("Camera matrix : \n")
        # print(mtx)
        # print("dist : \n")
        # print(dist)
        # print("rvecs : \n")
        # print(rvecs)
        # print("tvecs : \n")
        # print(tvecs)

    
    cv2.imshow('imgL',img_cL)
    cv2.imshow('imgR',img_cR)


    # key delay and action
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        print('Taking picture!!!')
        image_fileL = 'imageL_'+ str(file_save_counter)+".png"
        image_fileR = 'imageR_'+ str(file_save_counter)+".png"
        file_save_counter=file_save_counter+1
        cv2.imwrite(image_fileL,imgL)
        cv2.imwrite(image_fileR,imgR)
        # break
    elif key == ord('q'):
        print('q pressed!!!')
        break
    elif key != 255:
        print('key:',[chr(key)])



    # cv2.waitKey(0)

# release camera
cameraL.release()
cameraR.release()

cv2.destroyAllWindows()

# h,w = img.shape[:2]

# """
# Performing camera calibration by 
# passing the value of known 3D points (objpoints)
# and corresponding pixel coordinates of the 
# detected corners (imgpoints)
# """
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# print("Camera matrix : \n")
# print(mtx)
# print("dist : \n")
# print(dist)
# print("rvecs : \n")
# print(rvecs)
# print("tvecs : \n")
# print(tvecs)

