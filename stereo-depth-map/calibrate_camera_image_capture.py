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
camera = cv2.VideoCapture("/dev/video2") ### Left USB Camera


# Defining the dimensions of checkerboard
CHECKERBOARD = (6,8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # Creating vector to store vectors of 3D points for each checkerboard image
# objpoints = []
# # Creating vector to store vectors of 2D points for each checkerboard image
# imgpoints = [] 


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
# images = glob.glob('./images/*.jpg')

counter=0
file_save_counter=0

while 1:

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 

    # (grabbed_left,fname) = camera_left.read()
    (grabbed,img) = camera.read()
    img_c = img.copy()


    # for fname in images:
    # img = cv2.imread(grabbed_left)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        # counter=counter+1

        # if counter > 100:
        #     break


        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img_c = cv2.drawChessboardCorners(img_c, CHECKERBOARD, corners2, ret)

        h,w = img_c.shape[:2]

        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # print("Camera matrix : \n")
        # print(mtx)
        # print("dist : \n")
        # print(dist)
        # print("rvecs : \n")
        # print(rvecs)
        # print("tvecs : \n")
        # print(tvecs)

    
    cv2.imshow('img',img_c)


    # key delay and action
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        print('Taking picture!!!')
        image_file = 'image_'+ str(file_save_counter)+".png"
        file_save_counter=file_save_counter+1
        cv2.imwrite(image_file,img)
        # break
    elif key == ord('q'):
        print('q pressed!!!')
        break
    elif key != 255:
        print('key:',[chr(key)])



    # cv2.waitKey(0)

# release camera
camera.release()

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

