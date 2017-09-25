# coding=gbk

'''
just for fun: 摄像头识别
'''

import cv2.cv as cv
import cv2
import numpy as np

capture = cv.CaptureFromFile('data/airport/input.avi')



nbFrames = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
wait = int(1/fps * 1000/1)
width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))

out_foreground = cv2.VideoWriter('data/airport/testfun2.avi', -1, fps, (height, width))

gray = cv.CreateImage((width,height), cv.IPL_DEPTH_8U, 1)

background = cv.CreateMat(height, width, cv.CV_32F)
backImage = cv.CreateImage((width,height), cv.IPL_DEPTH_8U, 1)
foreground = cv.CreateImage((width,height), cv.IPL_DEPTH_8U, 1)
output = cv.CreateImage((width,height), 8, 1)

begin = True
threshold = 10

for f in xrange( nbFrames ):
    frame = cv.QueryFrame( capture )

    out_foreground.write(np.uint8(frame))

    cv.CvtColor(frame, gray, cv.CV_BGR2GRAY)

    if begin:
        cv.Convert(gray, background) #Convert gray into background format
        begin = False

    cv.Convert(background, backImage) #convert existing background to backImage

    cv.AbsDiff(backImage, gray, foreground) #Absdiff to get differences

    cv.Threshold(foreground, output, threshold, 255, cv.CV_THRESH_BINARY_INV)

    cv.Acc(foreground, background,output) #Accumulate to background

    cv.ShowImage("Output", output)
    cv.ShowImage("Gray", gray)
    c = cv.WaitKey(wait)
    if c==27: #Break if user enters 'Esc'.
        break