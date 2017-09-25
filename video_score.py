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


def diff(mask1, mask2):
    cap1 = cv2.VideoCapture(mask1)

    cap2 = cv2.VideoCapture(mask2)


    index, history = 1, 40 # ���˲�
    while True:
        ret, frame1 = cap1.read()
        ret, frame2 = cap2.read()
        # if i < history:
        #     i += 1
        #     continue
        # else:
        #     break
        if ret == True:

            # img1 = cv2.imread(frame1) # �����
            # img2 = cv2.imread(frame2) # ��׼��
            # cv2.imshow("mask1", frame1)
            # cv2.imshow("mask2", frame2)
            # cv2.waitKey(0)

            img1, img2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) # 1.�ҶȻ���ȥ����ɫ
            # img1, img2 = frame1, frame2
            ret, img1 = cv2.threshold(img1, 10, 255, cv2.THRESH_BINARY) # 2. ��ֵ����תΪ0-255��
            ret, img2 = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
            # cv2.imshow("img1", img1)
            # cv2.imshow("img2", img2)
            # cv2.waitKey(0)
            # print(img1.shape, img2.shape)
            # �����������ص㣬�Ա���ɫ��һ�����ǰ�ɫ����TP+1��
            rows, cols = img1.shape
            # rows, cols, channels = img1.shape
            # print('img1:', img1[1, 2])
            # for i in range(0, rows):
            #     for j in range(0, cols):
            #         if img1[i, j] != 0:
            #             # print(i, j, img1[i, j, 0], img1[i, j, 1], img1[i, j, 2], img2[i,j,0], img2[i, j, 1], img2[i, j, 2])
            #             print(i, j, img1[i, j], img2[i, j])
            # print(rows, cols)
            score_DR, score_ERROR, TP, TN, FN, FP = 0., 0., 0, 0, 0, 0
            for i in range(0, rows):
                for j in range(0, cols):
                    if img1[i, j] == img2[i, j] and img1[i, j] == 255: # ��ɫ��Ŀ����ɫ
                        TP += 1 # TP�Ǽ������������˶�Ŀ���������
                    elif img1[i, j] == img2[i, j] and img1[i, j] == 0: # ��ɫ����Ҫ�ı�����ɫ
                        TN += 1 # TNδ���������Ĳ�����Ŀ���������
                    elif img1[i, j] == 255 and img2[i, j] == 0:
                        FP += 1 # FP�Ǽ������Ĳ������˶�Ŀ���������
                    elif img1[i, j] == 0 and img2[i, j] == 255:
                        FN += 1 # FN��δ���������������˶�Ŀ���������
            print(TP, TN, FP, FN)
            # DR����� DR = TP / (TP + FN)
            score_DR = float(TP*1.0 / (TP + FN))

            # ERROR����� ERROR = FP / (TP + FP)
            score_ERROR = float(FP*1.0 / (TP + FP))

            print(index, score_DR, score_ERROR)
            index += 1

            # ͼ��������Ϊ��ά���飬����������õ�BGR��ע�ⲻ��RGB,һ��������һ����Ԫ��
            #
            # px = img1[100, 100]
            # print(px)
            # # ����img[row , col , index]  ,index=0ʱ��������ɫ������ֵ
            # blue = img1[100, 100, 0]
            # print(blue)
        else:
            break


#��λ���㣬������:AND , OR , NOT , XOR�����ã�ѡ��Ǿ���ROI�����������
def bitOperation():
    img1 = cv2.imread("data_test/roi.png")
    img2 = cv2.imread("data_test/opencv_logo.png")
    #ϣ����logo�������Ͻ�
    rows , cols , channels = img2.shape
    roi = img1[0 : rows , 0 : cols]

    #���ڴ�������logo������:��Դ�������루��Ҫ���ֶ�λΪ1������������õ���������Ľ��

    '''
    cv2.threshold(src, thresh , maxval , type[,dst] )����retval,dst
    src�����������ͼ��dst���ͼ��maxval���ڶ�Ԫ��ֵ�����ֵ,type:��ֵ����
    ���ã�����ֵӦ�õ���ͨ�����飬��Ҫ�õ��Ҷ�ͼ��,��Ҫ�ǹ��˵�̫���̫С��ͼ��
    ��Ҷ��ڴ���75��������ֵ������Ϊ255��������������Ϊ0��
    '''
    img2gray = cv2.cvtColor(img2 , cv2.COLOR_BGR2GRAY)
    ret , mask = cv2.threshold( img2gray , 75 , 255 , cv2.THRESH_BINARY )

    cv2.imshow("mask" , mask)
    '''
    cv2.bitwise_not(src[,dst[,mask]]) ,
    src:��������,dst:������飨��src��ͬ���Ĵ�С�����ͣ�,mask:��ѡ��Ĳ�������
    ���ã���λȡ��
    dst(I) = ȡ��src(I)
    bitwise��ʾ��λ
    '''
    mask_inv = cv2.bitwise_not(mask)

    #������ǽ�ROI������д���ȡroi����mask�в�Ϊ���ֵ��Ӧ�����ص�ֵ������ֵΪ0
    #ע�����������mask=mask ����mask=mask_inv�����е�mask= ���ܺ���
    '''
    cv2.bitwise_and(src1 ,src2[,dst[,mask]])->dst
    src1:��һ������������߱���
    src2:�ڶ�������
    src:��ͨ������������
    value:����ֵ
    dst:�������
    mask:����
    ���㰴λ��
    dst(I) = src1(I) & src2(I) , if mask(I) != 0
    '''
    #�����roi��������Ƭ�����ڱ���,mask��logo�ĻҶ�ͼ��0�Ǻڣ�255�ǰף�Ҳ���ǰѰ�ɫ���ֵ������ó������룬��ʵ���ǰ�����ƫ��ɫ�Ĳ����ó���
    img1_bg = cv2.bitwise_and(roi, roi , mask = mask)
    #ȡroi����mask_inv�в�Ϊ0��ֵ��Ӧ�����ص�ֵ������ֵΪ0����logo�к�ɫ������ȡ����
    img2_fg = cv2.bitwise_and(img2 , img2 , mask = mask_inv)
    cv2.imshow("img1_bg" , img1_bg) # ȥ��Ŀ��ͼƬҪ���һ��
    cv2.imshow("img2_fg" , img2_fg) # ����Ҫ���һ��
    #��ROI�е�logo���޸���Ҫ��ͼ��
    dst = cv2.add(img1_bg , img2_fg)
    #�滻ԭ����ͼ��
    img1[0:rows , 0:cols] = dst
    cv2.imshow("res" , img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def MOG_test(input_path, save_path):
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

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            fgmask = fgbg.apply(frame) # ����MOG�㷨
            # print(type(fgmask))
            # ��������
            videoWriter.write(fgmask)

            cv2.imshow('frame',fgmask)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break
    print('finish.')
    cap.release()
    cv2.destroyAllWindows()


def makeMask(path, path2):
    cap = cv2.VideoCapture(path)
    # 1. ��ȡ��Ƶ���ʡ���ʽ��
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    codec = cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

    print(fps, size, codec)

    # ָ��д��Ƶ�ĸ�ʽ, I420-avi, MJPG-mp4
    # videoWriter = cv2.VideoWriter(save_path, cv2.cv.CV_FOURCC('I', '4', '2', '0'), fps, size)
    videoWriter_mask = cv2.VideoWriter(path2, int(codec), fps, size)
    i, history = 0, 40 # ���˲�
    while True:
        ret, img = cap.read()
        # if i < history:
        #     i += 1
        #     continue
        # else:
        #     break
        if ret == True:
            pass
            # OpenCV����ĽṹԪ��
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

            # 1. ��һ
            # # ��ʴͼ��
            # eroded = cv2.erode(img, kernel)
            # # ��ʾ��ʴ���ͼ��
            # cv2.imshow("Eroded Image", eroded);
            #
            # # ����ͼ��
            # dilated = cv2.dilate(img, kernel)
            # # ��ʾ���ͺ��ͼ��
            # cv2.imshow("Dilated Image", dilated);
            # # ԭͼ��
            # cv2.imshow("Origin", img)

            # ����
            # �����㣬�������������ӱ����Ϊ���С��Ķ���
            closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            # ��ʾ��ʴ���ͼ��
            cv2.imshow("Close", closed)

            # �����㣬�����������Ƴ���ͼ�������γɵİߵ㡣
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # ��ʾ��ʴ���ͼ��
            cv2.imshow("Open", opened)

            videoWriter_mask.write(opened)

            # ����
            # # NumPy����ĽṹԪ��
            # NpKernel = np.uint8(np.ones((3, 3)))
            # Nperoded = cv2.erode(img, NpKernel)
            # # ��ʾ��ʴ���ͼ��
            # cv2.imshow("Eroded by NumPy kernel", Nperoded);


            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            break


def posScore(path):

    # ��ԭʼ֡��������ȥ��
    th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    # ��ȡ���м���
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # ��ȡ���ο�߽�����
        x, y, w, h = cv2.boundingRect(c)
        # ������ο�����
        area = cv2.contourArea(c)
        if 2500 < area < 8000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



if __name__ == '__main__':

    video_dynamic = 'data/noshake_dynamic/waterSurface/input.avi'
    out_path_dynamic = 'data/noshake_dynamic/waterSurface/fun2_out.avi'

    # video_shake = 'data/shake/people2/input.avi'
    # out_path_shake = 'data/shake/people2/fun2_out.avi'

    # test function:
    # MOG_test(video_dynamic, out_path_dynamic) # Ч������
    # MOG2_test(video_dynamic, out_path_dynamic) # Ч���ܲ����

    # MOG_test(video_shake, out_path_shake)
    # MOG2_test(video_shake, out_path_shake)

    # mask1 = "data/noshake_static/airport/fun2mask.avi"
    # mask2 = "data/noshake_static/airport/mask.avi"

    mask1 = "data/noshake_static/hall/fun2mask.avi"
    mask11 = "data/noshake_static/hall/fun2mask11.avi"
    mask2 = "data/noshake_static/hall/mask.avi"

    mask1_dynamic = 'data/noshake_dynamic/waterSurface/fun2mask.avi'
    mask2_dynamic = 'data/noshake_dynamic/waterSurface/mask.avi'
    # bitOperation()
    diff(mask1, mask2)
    # makeMask(mask1, mask11)
