# coding=gbk
'''
�������
    ���� opencv2�㷨ʵ�֣����ǲο�����opencv3��api��https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html#background-subtraction
    
        MOG2
        GMG
        
    ע�⣺
        ���������Ҫ�����޸ģ���Ӧopencv2.4.13������ȥ��create

'''
import numpy as np
import cv2
import cv2.cv as cv
print(cv2.__version__)


def MOG_test(input_path, save_path, save_path_mask):
    cap = cv2.VideoCapture(input_path)

    # 1. ��ȡ��Ƶ���ʡ���ʽ��
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    codec = cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

    print(fps, size, codec)

    # ָ��д��Ƶ�ĸ�ʽ, I420-avi, MJPG-mp4
    videoWriter = cv2.VideoWriter(save_path, int(codec), fps, size)
    videoWriter_mask = cv2.VideoWriter(save_path_mask, int(codec), fps, size) # cv2.cv.CV_FOURCC('I', '4', '2', '0')
    fgbg = cv2.BackgroundSubtractorMOG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            # frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # �ҶȻ�
            fgmask = fgbg.apply(frame, learningRate=0.01) # ����MOG�㷨
            cv2.imshow("fgmask", fgmask)
            # ��������
            videoWriter.write(fgmask)

            ret, mask = cv2.threshold(fgmask, 15, 255, cv2.THRESH_BINARY)

            # �����㣬�������������ӱ����Ϊ���С��Ķ���
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # ��ʾ��ʴ���ͼ��
            # cv2.imshow("Close", closed)

            # �����㣬�����������Ƴ���ͼ�������γɵİߵ㡣
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            # ��ʾ��ʴ���ͼ��
            cv2.imshow("Open", opened)

            videoWriter_mask.write(opened)
            # cv2.imshow("mask", mask)

            # cv2.imshow('frame',fgmask)
            k = cv2.waitKey(3000) & 0xff
            if k == 27:
                continue
        else:
            break

    print('finish.')
    cap.release()
    cv2.destroyAllWindows()

def MOG2_test(input_path, save_path):
    cap = cv2.VideoCapture(input_path)

    # 1. ��ȡ��Ƶ���ʡ���ʽ��
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    codec = cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

    print(fps, size, codec)

    # ָ��д��Ƶ�ĸ�ʽ, I420-avi, MJPG-mp4
    videoWriter = cv2.VideoWriter(save_path, cv2.cv.CV_FOURCC('I', '4', '2', '0'), fps, size)

    fgbg = cv2.BackgroundSubtractorMOG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgbg = cv2.BackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:

            fgmask = fgbg.apply(frame)
            # cv2.imshow('frame222', fgmask)
            videoWriter.write(fgmask)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            # print(type(fgmask))
            # fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)





            # ret, fgmask = cv2.threshold(fgmask, 114, 255, cv2.THRESH_BINARY)  # 2. ��ֵ����תΪ0-255��


            # �����㣬�������������ӱ����Ϊ���С��Ķ���
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            # ��ʾ��ʴ���ͼ��
            # cv2.imshow("Close", closed)

            # �����㣬�����������Ƴ���ͼ�������γɵİߵ㡣
            opened = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            # ��ʾ��ʴ���ͼ��
            # cv2.imshow("Open", opened)

            # videoWriter.write(fgmask)
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

    # 1. ��ȡ��Ƶ���ʡ���ʽ��
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    codec = cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

    print(fps, size, codec)

    # ָ��д��Ƶ�ĸ�ʽ, I420-avi, MJPG-mp4
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

    video_dynamic = 'data/noshake_dynamic/waterSurface/input.avi' # input foreground
    out_path_dynamic = 'data/noshake_dynamic/waterSurface/fun2out.avi'
    out_path_dynamic_mask = 'data/noshake_dynamic/waterSurface/fun2mask.avi'

    video_shake = 'data/shake/people2/input.avi'
    out_path_shake = 'data/shake/people2/fun2_out.avi'

    # test function:
    MOG_test(noshake_static, out_path_static, out_path_static_mask)  # ��̬Ч��һ��
    # save(out_path_static, out_path_static_mask)
    # MOG2_test(noshake_static, out_path_static)  # Ч����Ҫ����

    # MOG_test(video_dynamic, out_path_dynamic, out_path_dynamic_mask) # ��̬Ч������
    # MOG2_test(video_dynamic, out_path_dynamic) # Ч���ܲ����

    # MOG_test(video_shake, out_path_shake)
    # MOG2_test(video_shake, out_path_shake)
