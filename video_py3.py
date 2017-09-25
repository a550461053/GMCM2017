# coding:utf8
import cv2
import numpy as np

print(cv2.__version__) # 3.3.0

def detect_video(video):
    camera = cv2.VideoCapture(video)
    history = 20    # 训练帧数
    # createBackgroundSubtractorMOG2 MOG2算法，也是高斯混合模型分离算法，是MOG的改进算法
    # createBackgroundSubtractorKNN
    # bgsegm.createBackgroundSubtractorGMG
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
    bs.setHistory(history)

    frames = 0

    while True:
        res, frame = camera.read()

        if not res:
            break

        fg_mask = bs.apply(frame)   # 获取 foreground mask

        if frames < history:
            frames += 1
            continue

        # 对原始帧进行膨胀去噪
        th = cv2.threshold(fg_mask.copy(), 1, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 3)), iterations=1)
        # 获取所有检测框
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # 获取矩形框边界坐标
            x, y, w, h = cv2.boundingRect(c)
            # 计算矩形框的面积
            area = cv2.contourArea(c)
            if 50 < area < 300:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("detection", frame)
        cv2.imshow("back", dilated)
        if cv2.waitKey(10) & 0xff == 27:
            break
    camera.release()

def test_KNN(video, output):
    cap = cv2.VideoCapture(video)
    fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    history = 0
    fgbg.setHistory(history)

    # 1. 获取视频码率、格式：
    fps = cap.get(cv2.CAP_PROP_FPS) # opencv3没有cv.CV_
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    codec = (cap.get(cv2.CAP_PROP_FOURCC))

    print(fps, size, codec)

    # 2. 指定写视频的格式, I420-avi, MJPG-mp4
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoWriter = cv2.VideoWriter(output, int(codec), fps, size)

    # 3. 针对问题四指定 帧号：
    frame_list = []
    frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            # cv2.waitKey(1000 // int(fps))
            fgmask = fgbg.apply(frame) # , learningRate=0.01

            if frames < history: # 初始延迟等待
                frames += 1
                continue

            # 对原始帧进行膨胀去噪
            th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
            th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
            # 获取所有检测框
            image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 当前帧号获取：
            frame_now = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if len(contours) >= 2:
                print('frame_now:', frame_now, len(contours))
                frame_list.append(frame_now)

            # 保存视频
            videoWriter.write(image)

            cv2.imshow('frame', fgmask)

            cv2.imshow("back", image)

            k = cv2.waitKey(3000) & 0xff
            if k == 27:
                continue
        else:
            print('finished.')
            break
    print(frame_list)
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


def test1_diff(video, output):
    cap = cv2.VideoCapture(video)

    # 1. 获取视频码率、格式：
    fps = cap.get(cv2.CAP_PROP_FPS) # opencv3没有cv.CV_
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    codec = cap.get(cv2.CAP_PROP_FOURCC)

    print(fps, size, codec)

    # 2. 指定写视频的格式, I420-avi, MJPG-mp4
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoWriter = cv2.VideoWriter(output, int(codec), fps, size)

    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # kernel = np.ones((5, 5), np.uint8)
    background = None
    while cap.isOpened():
        # 1. 读取每帧图像
        ret, frame = cap.read()

        if ret == True:

            # 2. 对背景帧进行灰度和平滑处理
            if background is None:
                background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                background = cv2.GaussianBlur(background, (21, 21), 0)
                continue

            # 3. 将其他帧进行灰度处理和模糊平滑处理
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            # 4. 计算其他帧与背景之间的差异，得到一个差分图
            diff = cv2.absdiff(background, gray_frame)
            # 5. 应用阈值得到一副黑白图像，并通过dilate膨胀图像，从而对孔和缺陷进行归一处理
            diff = cv2.threshold(diff, 21, 255, cv2.THRESH_BINARY)[1]
            diff = cv2.dilate(diff, es, iterations=2)

            # 6. 显示矩形框，在计算出的差分图中找到所有的白色斑点轮廓，并显示轮廓
            image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < 1500:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # 保存diff
            diff = cv2.flip(diff, 0)
            print(type(diff)) # np.ndarray
            print(type(frame))
            videoWriter.write(diff) # 保存打不开，

            cv2.imshow("contours", frame)
            cv2.imshow("dif", diff)

            # if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            #     break

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break
    print('finish!')
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

def test1_MOG(video): # mog 识别电梯效果还可以
    cap = cv2.VideoCapture(video)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    frame_list = []
    while cap.isOpened():
        ret, src = cap.read()
        if ret == True:
            fgmask = fgbg.apply(src, learningRate=0.01)
            # dst = src.copy()
            # dst = cv2.bitwise_and(src, src, mask=fgmask)
            dst = fgmask
            cv2.imshow('frame', dst)

            # 获取所有检测框
            # 对原始帧进行膨胀去噪
            th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
            th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)

            # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            cv2.imshow('dilated', dilated)
            image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 当前帧号获取：
            frame_now = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # if frame_now <= 250 and frame_now >= 240 or frame_now <433 and frame_now > 420:
            #     print('frame_now:', frame_now, len(contours))

            # 检测前景目标所在帧数
            if len(contours) >= 1:
                print('frame_now:', frame_now, len(contours))
                frame_list.append(frame_now)
            else:
                print('frame_now:', frame_now)


            k = cv2.waitKey(10) & 0xff
            if k == 27:  # ESC key
                break
        else:
            print('finished.')
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame_list

def test1_GMG(video): # 黑屏。。。
    cap = cv2.VideoCapture(video)
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    while True:
        ret, src = cap.read()
        fgmask = fgbg.apply(src, learningRate=0.5)
        dst = src.copy()
        dst = cv2.bitwise_and(src, src, mask=fgmask)

        cv2.imshow('frame', dst)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # ESC key
            break
    cap.release()
    cv2.destroyAllWindows()


# 求两个图像的差距程度
def diff(mask1, mask2):
    cap1 = cv2.VideoCapture(mask1)
    ret, frame1 = cap1.read()
    cap2 = cv2.VideoCapture(mask2)
    ret, frame2 = cap2.read()
    # img1 = cv2.imread(frame1) # 检测结果
    # img2 = cv2.imread(frame2) # 标准答案
    # img1, img2 = cv2.cvtColor(frame1 , cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    img1, img2 = frame1, frame2
    print(img1.shape, img2.shape)
    # 遍历所有像素点，对比颜色，一致且是白色，则TP+1，
    rows, cols, channels = img1.shape
    print('the img1 is:', img1[1, 2, 1], img1[1, 2, 0], img1[1, 2, 2])
    print(img1[100, 22, 0], img1[100, 22, 1],img1[100, 22, 2])

    print(rows, cols)

    for i in range(0, rows):
        for j in range(0, cols):
            if img1[i, j, 0] != 0:
                print(i, j, img1[i, j, 0], img1[i, j, 1], img1[i, j, 2])

    score_DR, score_ERROR, TP, TN, FN, FP = 0., 0., 0, 0, 0, 0
    for i in range(0, rows):
        for j in range(0, cols):
            # print(i, j, img1[i, j, 0])
            if img1[i, j, 0] == img2[i, j, 0] and img1[i, j, 0] == 255: # 白色，目标颜色
                TP += 1
            elif img1[i, j, 0] == img2[i, j, 0] and img1[i, j, 0] == 0: # 黑色，不要的背景颜色
                TN += 1 # TF未被检测出来的不属于目标的像素数
            elif img1[i, j, 0] == 255 and img2[i, j, 0] == 0:
                FP += 1 # FP是检测出来的不属于运动目标的像素数
            elif img1[i, j, 0] == 0 and img2[i, j, 0] == 255:
                FN += 1 # FN是未被检测出来的属于运动目标的像素数
    print(TP, TN, FP, FN)
    # DR检出率 DR = TP / (TP + FN)
    score_DR = float(TP*1.0 / (TP + FN))

    # ERROR误检率 ERROR = FP / (TP + FP)
    score_ERROR = float(TP*1.0 / (TP + FP))

    print(score_DR, score_ERROR)

if __name__ == '__main__':
    input_list_q4 = ['campus','curtain','escalator','fountain','hall','lobby','office','overpass']
    input_list_Q4 = ['Campus','Curtain','Escalator','Fountain','Hall','Lobyy','Office','Overpass']
    num_q4 = 0
    # video = 'data/q4/'+ input_list_q4[num_q4] + '/'+input_list_Q4[num_q4]+'.avi'
    input = ['airport','hall', 'office', 'pedestrian', 'smoke']
    video = 'data/'+ input[0] +'/input.avi'
    output = 'data/testyupy3.avi'
    # video = 'data/people2.avi'
    # video = 'data/car1.avi'

    video = 'data/water.avi'
    detect_video(video)
    # test_KNN(video, output)
    # test1_diff(video, output)
    # test1_MOG(video) # 电梯的还可以
    # test1_GMG(video)

    out_path_static = 'data/fun2mask.avi'
    out_path_static_mask = 'data/airport/mask.avi'
    # diff(out_path_static, out_path_static_mask)

