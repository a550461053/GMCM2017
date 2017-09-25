# coding=gbk

'''

问题一：
    平均值法：
        通过计算两帧图像之间变化了的像素点占的百分比，来确定图像中是否有动作产生。
        这里主要用到 Absdiff 函数，比较两帧图像之间有差异的点，当然需要将图像进行一些处理，例如平滑处理，灰度化处理，二值化处理，经过处理之后的二值图像上的点将更有效。
'''

import cv2
import cv2.cv as cv
import numpy as np

print(cv2.__version__)

# 定义输入，修改file_num号即可
file_num = 1
input = ['airport', 'hall', 'office', 'pedestrian', 'smoke']
video = 'data/noshake_static/' + input[file_num] + '/input.avi'
capture = cv.CaptureFromFile(video) # 从文件获取图片cap

video_dynamic = 'data/noshake_dynamic/waterSurface/input.avi'
# capture = cv.CaptureFromFile(video_dynamic)

# video_shake = 'data/shake/people2/input.avi'
# capture = cv.CaptureFromFile(video_shake)

video_campus = 'data/Campus.avi'
video_shake = 'data/shake/people2/input.avi'
# capture = cv.CaptureFromFile(video)

frame1 = cv.QueryFrame(capture)

# 获得视频码率及尺寸
nbFrames = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))
codec = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FOURCC)
fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
duration = (nbFrames * fps) / 1000 # 时长计算
print 'Num. Frames = ', nbFrames # frame为帧数，frames为总帧数
print 'Frame Rate = ', fps, 'fps' # fps为文件的帧率
print 'Duration = ', duration, 'sec'
print 'codec = ', codec

# 定义输出
out_list = ['airport', 'hall', 'office', 'pedestrian', 'smoke']
out_f = 'data/noshake_static/' + out_list[file_num] + '/yu_foreground' + '.avi'
out_m = 'data/noshake_static/' + out_list[file_num] + '/yu_mask' + '.avi'
# out_foreground = cv2.VideoWriter(out_f, -1, 30.0, (height, width)) # 此为新版cv2的存储视频api
# out_mask = cv2.VideoWriter(out_m, -1, 30.0, (height, width))
# writer=cv.CreateVideoWriter("output.avi", cv.CV_FOURCC("D", "I", "V", "X"), 5, cv.GetSize(temp), 1)
out_foreground=cv.CreateVideoWriter(out_f, int(codec), int(fps), (width,height), 1) #Create writer with same parameters
out_mask=cv.CreateVideoWriter(out_m, int(codec), int(fps), (width,height), 1) #Create writer with same parameters
# On linux I used to take "M","J","P","G" as fourcc

# print(frame1.height, frame1.width)

# 建立中间变量：frame1gray 和 frame2gray
frame1gray = cv.CreateMat(height, width, cv.CV_8U) # CreateMat(rows, cols, type)
cv.CvtColor(frame1, frame1gray, cv.CV_RGB2GRAY)  # cv.CvtColor(src, dst, code) ，输入图像，输出图像，颜色值
res = cv.CreateMat(height, width, cv.CV_8U)
frame2gray = cv.CreateMat(height, width, cv.CV_8U)
# frame2gray = np.array([height, width, cv2.CV_8U])

# gray = cv.CreateImage((width,height), cv.IPL_DEPTH_8U, 1)

w= width
h= height
nb_pixels = width * height

# 总循环
index = 1 # 记帧数，或者用GetCaptureProperty方法
while True:
    # 1. 当前帧数
    print('frame:', index)
    if index == nbFrames:
        print('finish.')
        break

    # print cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_POS_FRAMES) # frame == index

    # 2. 当前帧frame2赋给frame2gray
    frame2 = cv.QueryFrame(capture)
    cv.CvtColor(frame2, frame2gray, cv.CV_RGB2GRAY) # 原图frame2转换为frame2gray格式

    # 3. 图像做帧差法，结果为res
    cv.AbsDiff(frame1gray, frame2gray, res) # 比较frame1gray 和 frame2gray 的差，结果输出给res
    # res = cv2.absdiff(frame1gray, frame2gray) # 此为cv2新版api
    cv.ShowImage("After AbsDiff", res)

    # 4. 保存res作为前景目标foreground
    # cv.Convert(res, gray)
    # out_foreground.write(np.uint8(res)) # 保存为前景目标
    # out_foreground.write(np.asarray(cv.GetMat(res)))
    cv.WriteFrame(out_foreground, cv.GetImage(res)) # res格式为cvmat格式，转化为iplimage格式

    # 5. 平滑处理
    # cv.Smooth(res, res, cv.CV_BLUR, 5, 5) # 光滑一下res

    # 6. 形态学变换，开闭操作
    element = cv.CreateStructuringElementEx(5*2+1, 5*2+1, 5, 5,  cv.CV_SHAPE_RECT) # CreateStructuringElementEx(cols, rows, anchorX, anchorY, shape, values=None)

    cv.MorphologyEx(res, res, None, None, cv.CV_MOP_OPEN) # cv2.morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]])
    cv.MorphologyEx(res, res, None, None, cv.CV_MOP_CLOSE) # 形态学变换相应的开 闭操作

    # 7. 二值化阈值处理：对得到的前景进行阈值选择，去掉伪前景
    cv.Threshold(res, res, 10, 255, cv.CV_THRESH_BINARY) # 二值化 cv.Threshold(src, dst, threshold, maxValue, thresholdType)


    # 8. blob提取电梯区域 ------------- 未完
    # print(type(res))
    # # Set up the detector with default parameters.
    # detector = cv2.SimpleBlobDetector()
    # # Detect blobs.
    # keypoints = detector.detect(np.array(res))
    # # Draw detected blobs as red circles.
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # im_with_keypoints = cv2.drawKeypoints(np.array(res), keypoints, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # Show keypoints
    # cv2.imshow("Keypoints", im_with_keypoints)


    # cv.ShowImage("Image", frame2) # 当前原始图形


    # 9. 保存处理过的res作为mask
    cv.ShowImage("Res", res)
    # out_mask.write(np.array(res))  # 保存为mask
    cv.WriteFrame(out_mask, cv.GetImage(res))


    # #----------- 单纯的显示作用：表明是否视频变动
    # nb=0
    # for y in range(h):
    #     for x in range(w):
    #         if res[y,x] == 0.0:
    #             nb += 1
    # avg = (nb*100.0)/nb_pixels
    # #print "Average: ",avg, "%\r",
    # if avg >= 5:
    #     print("Something is moving !")
    # #-----------

    # 10. 循环迭代，用于做差
    cv.Copy(frame2gray, frame1gray) # cv.Copy(src, dst, mask=None)
        # if mask != 0: dst = src
        # 相当于循环迭代
    index += 1

    # 11. 视频太短，用于延迟放慢视频
    c = cv.WaitKey(30) # 等待时间

    # 12. 视频太长，提前退出
    if c == 27: #Break if user enters 'Esc'.
        break

# 退出处理
# capture.release()
# out_foreground.release()
# out_mask.release()
cv2.destroyAllWindows()
