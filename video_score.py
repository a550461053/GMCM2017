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


def diff(mask1, mask2):
    cap1 = cv2.VideoCapture(mask1)

    cap2 = cv2.VideoCapture(mask2)


    index, history = 1, 40 # 简单滤波
    while True:
        ret, frame1 = cap1.read()
        ret, frame2 = cap2.read()
        # if i < history:
        #     i += 1
        #     continue
        # else:
        #     break
        if ret == True:

            # img1 = cv2.imread(frame1) # 检测结果
            # img2 = cv2.imread(frame2) # 标准答案
            # cv2.imshow("mask1", frame1)
            # cv2.imshow("mask2", frame2)
            # cv2.waitKey(0)

            img1, img2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) # 1.灰度化，去掉颜色
            # img1, img2 = frame1, frame2
            ret, img1 = cv2.threshold(img1, 10, 255, cv2.THRESH_BINARY) # 2. 二值化，转为0-255，
            ret, img2 = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
            # cv2.imshow("img1", img1)
            # cv2.imshow("img2", img2)
            # cv2.waitKey(0)
            # print(img1.shape, img2.shape)
            # 遍历所有像素点，对比颜色，一致且是白色，则TP+1，
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
                    if img1[i, j] == img2[i, j] and img1[i, j] == 255: # 白色，目标颜色
                        TP += 1 # TP是检测出来的属于运动目标的像素数
                    elif img1[i, j] == img2[i, j] and img1[i, j] == 0: # 黑色，不要的背景颜色
                        TN += 1 # TN未被检测出来的不属于目标的像素数
                    elif img1[i, j] == 255 and img2[i, j] == 0:
                        FP += 1 # FP是检测出来的不属于运动目标的像素数
                    elif img1[i, j] == 0 and img2[i, j] == 255:
                        FN += 1 # FN是未被检测出来的属于运动目标的像素数
            print(TP, TN, FP, FN)
            # DR检出率 DR = TP / (TP + FN)
            score_DR = float(TP*1.0 / (TP + FN))

            # ERROR误检率 ERROR = FP / (TP + FP)
            score_ERROR = float(FP*1.0 / (TP + FP))

            print(index, score_DR, score_ERROR)
            index += 1

            # 图像可以理解为二维数组，给出行列则得到BGR，注意不是RGB,一个像素是一个三元组
            #
            # px = img1[100, 100]
            # print(px)
            # # 给出img[row , col , index]  ,index=0时，给出蓝色的像素值
            # blue = img1[100, 100, 0]
            # print(blue)
        else:
            break


#按位运算，操作有:AND , OR , NOT , XOR，作用：选择非矩形ROI操作会很有用
def bitOperation():
    img1 = cv2.imread("data_test/roi.png")
    img2 = cv2.imread("data_test/opencv_logo.png")
    #希望把logo放在左上角
    rows , cols , channels = img2.shape
    roi = img1[0 : rows , 0 : cols]

    #现在创建对于logo的掩码:将源码与掩码（需要的字段位为1）经过或运算得到符合需求的结果

    '''
    cv2.threshold(src, thresh , maxval , type[,dst] )返回retval,dst
    src输入数组或者图像，dst输出图像，maxval用于二元阈值的最大值,type:阈值类型
    作用：将阈值应用到单通道数组，需要用到灰度图像,主要是过滤掉太大或太小的图像
    则灰度在大于75的像素其值将设置为255，其它像素设置为0。
    '''
    img2gray = cv2.cvtColor(img2 , cv2.COLOR_BGR2GRAY)
    ret , mask = cv2.threshold( img2gray , 75 , 255 , cv2.THRESH_BINARY )

    cv2.imshow("mask" , mask)
    '''
    cv2.bitwise_not(src[,dst[,mask]]) ,
    src:输入数组,dst:输出数组（与src有同样的大小和类型）,mask:可选择的操作掩码
    作用：按位取反
    dst(I) = 取反src(I)
    bitwise表示按位
    '''
    mask_inv = cv2.bitwise_not(mask)

    #下面就是讲ROI区域进行处理，取roi中与mask中不为零的值对应的像素的值，其他值为0
    #注意这里必须有mask=mask 或者mask=mask_inv，其中的mask= 不能忽略
    '''
    cv2.bitwise_and(src1 ,src2[,dst[,mask]])->dst
    src1:第一个输入数组或者标量
    src2:第二个数组
    src:单通道的输入数组
    value:标量值
    dst:输出数组
    mask:掩码
    计算按位与
    dst(I) = src1(I) & src2(I) , if mask(I) != 0
    '''
    #这里的roi是足球照片，用于背景,mask是logo的灰度图像，0是黑，255是白，也就是把白色部分的像素拿出来求与，其实就是把足球偏白色的部分拿出来
    img1_bg = cv2.bitwise_and(roi, roi , mask = mask)
    #取roi中与mask_inv中不为0的值对应的像素的值，其他值为0，把logo中黑色部分提取出来
    img2_fg = cv2.bitwise_and(img2 , img2 , mask = mask_inv)
    cv2.imshow("img1_bg" , img1_bg) # 去掉目标图片要插的一块
    cv2.imshow("img2_fg" , img2_fg) # 构建要插的一块
    #将ROI中的logo和修改主要的图像
    dst = cv2.add(img1_bg , img2_fg)
    #替换原来的图像
    img1[0:rows , 0:cols] = dst
    cv2.imshow("res" , img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def MOG_test(input_path, save_path):
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

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            fgmask = fgbg.apply(frame) # 调用MOG算法
            # print(type(fgmask))
            # 保存数据
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
    # 1. 获取视频码率、格式：
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    codec = cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

    print(fps, size, codec)

    # 指定写视频的格式, I420-avi, MJPG-mp4
    # videoWriter = cv2.VideoWriter(save_path, cv2.cv.CV_FOURCC('I', '4', '2', '0'), fps, size)
    videoWriter_mask = cv2.VideoWriter(path2, int(codec), fps, size)
    i, history = 0, 40 # 简单滤波
    while True:
        ret, img = cap.read()
        # if i < history:
        #     i += 1
        #     continue
        # else:
        #     break
        if ret == True:
            pass
            # OpenCV定义的结构元素
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

            # 1. 法一
            # # 腐蚀图像
            # eroded = cv2.erode(img, kernel)
            # # 显示腐蚀后的图像
            # cv2.imshow("Eroded Image", eroded);
            #
            # # 膨胀图像
            # dilated = cv2.dilate(img, kernel)
            # # 显示膨胀后的图像
            # cv2.imshow("Dilated Image", dilated);
            # # 原图像
            # cv2.imshow("Origin", img)

            # 法二
            # 闭运算，闭运算用来连接被误分为许多小块的对象，
            closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            # 显示腐蚀后的图像
            cv2.imshow("Close", closed)

            # 开运算，开运算用于移除由图像噪音形成的斑点。
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # 显示腐蚀后的图像
            cv2.imshow("Open", opened)

            videoWriter_mask.write(opened)

            # 法三
            # # NumPy定义的结构元素
            # NpKernel = np.uint8(np.ones((3, 3)))
            # Nperoded = cv2.erode(img, NpKernel)
            # # 显示腐蚀后的图像
            # cv2.imshow("Eroded by NumPy kernel", Nperoded);


            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            break


def posScore(path):

    # 对原始帧进行膨胀去噪
    th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    # 获取所有检测框
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # 获取矩形框边界坐标
        x, y, w, h = cv2.boundingRect(c)
        # 计算矩形框的面积
        area = cv2.contourArea(c)
        if 2500 < area < 8000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



if __name__ == '__main__':

    video_dynamic = 'data/noshake_dynamic/waterSurface/input.avi'
    out_path_dynamic = 'data/noshake_dynamic/waterSurface/fun2_out.avi'

    # video_shake = 'data/shake/people2/input.avi'
    # out_path_shake = 'data/shake/people2/fun2_out.avi'

    # test function:
    # MOG_test(video_dynamic, out_path_dynamic) # 效果不错
    # MOG2_test(video_dynamic, out_path_dynamic) # 效果很差，。。

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
