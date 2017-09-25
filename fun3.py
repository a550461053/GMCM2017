# coding=gbk
'''
问题二：
    采用 opencv2算法实现，但是参考的是opencv3的api：https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html#background-subtraction
    
        MOG2
        GMG
        
    注意：
        程序代码需要稍作修改，适应opencv2.4.13，比如去掉create

'''
import numpy as np
import cv2
import cv2.cv as cv
print(cv2.__version__)


def MOG_test(input_path, save_path, save_path_mask):
    cap = cv2.VideoCapture(input_path)

    # 1. 获取视频码率、格式：
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    codec = cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

    print(fps, size, codec)

    # 指定写视频的格式, I420-avi, MJPG-mp4
    videoWriter = cv2.VideoWriter(save_path, int(codec), fps, size)
    videoWriter_mask = cv2.VideoWriter(save_path_mask, int(codec), fps, size) # cv2.cv.CV_FOURCC('I', '4', '2', '0')
    fgbg = cv2.BackgroundSubtractorMOG()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:

            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度化
            fgmask = fgbg.apply(frame1, learningRate=0.01) # 调用MOG算法


            # 对原始帧进行膨胀去噪
            th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
            th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
            # 获取所有检测框
            contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                # 获取矩形框边界坐标
                x, y, w, h = cv2.boundingRect(c)
                # 计算矩形框的面积
                area = cv2.contourArea(c)
                if 1500 < area < 8000:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 保存数据
            videoWriter.write(fgmask)

            ret, mask = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)
            videoWriter_mask.write(mask)
            # cv2.imshow("mask", mask)
            cv2.imshow('frame', frame)
            # cv2.imshow('frame_mask',fgmask)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break

    print('finish.')
    cap.release()
    cv2.destroyAllWindows()

def MOG2_test(input_path, save_path):
    cap = cv2.VideoCapture(input_path)

    # 1. 获取视频码率、格式：
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    codec = cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

    print(fps, size, codec)

    # 指定写视频的格式, I420-avi, MJPG-mp4
    videoWriter = cv2.VideoWriter(save_path, cv2.cv.CV_FOURCC('I', '4', '2', '0'), fps, size)

    fgbg = cv2.BackgroundSubtractorMOG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.BackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:

            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            print(type(fgmask))
            videoWriter.write(fgmask)
            cv2.imshow('frame', fgmask)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break
    print('finish.')
    cap.release()
    cv2.destroyAllWindows()



def save(input_path, save_path_mask):

    cap = cv2.VideoCapture(input_path)

    # 1. 获取视频码率、格式：
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    codec = cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

    print(fps, size, codec)

    # 指定写视频的格式, I420-avi, MJPG-mp4
    # videoWriter = cv2.VideoWriter(save_path, cv2.cv.CV_FOURCC('I', '4', '2', '0'), fps, size)
    videoWriter_mask = cv2.VideoWriter(save_path_mask, cv2.cv.CV_FOURCC('I', '4', '2', '0'), fps, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            ret, mask = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)
            videoWriter_mask.write(mask)
            cv2.imshow("mask", mask)
        else:
            break

if __name__ == '__main__':


    noshake_static = 'data/noshake_static/hall/input.avi'
    out_path_static = 'data/noshake_static/hall/fun2out.avi'
    out_path_static_mask = 'data/noshake_static/hall/fun2mask.avi'

    # video_dynamic = 'data/noshake_dynamic/waterSurface/input.avi' # input foreground
    out_path_dynamic = 'data/noshake_dynamic/waterSurface/fun2out.avi'
    out_path_dynamic_mask = 'data/noshake_dynamic/waterSurface/fun2mask.avi'

    video_shake = 'data/shake/people2/people2.avi' # input
    out_path_shake = 'data/shake/people2/fun2out.avi'
    out_path_shake_mask = 'data/shake/people2/fun2mask.avi'


    video = 'data/noshake_static/' + 'hall' + '/input.avi'
    # MOG_test(video, out_path_static, out_path_static_mask)
    # test function:
    # MOG_test(noshake_static, out_path_static, out_path_static_mask)  # 静态效果一般
    # save(out_path_static, out_path_static_mask)
    # MOG2_test(noshake_static, out_path_static)  # 效果需要调试


    video_dynamic = 'data/1.avi'

    MOG_test(video_dynamic, out_path_dynamic, out_path_dynamic_mask) # 动态效果不错
    # MOG2_test(video_dynamic, out_path_dynamic) # 效果很差，。。

    # MOG_test(video_shake, out_path_shake, out_path_shake_mask)
    # MOG2_test(video_shake, out_path_shake)
