# coding=gbk

'''

����һ��
    ƽ��ֵ����
        ͨ��������֡ͼ��֮��仯�˵����ص�ռ�İٷֱȣ���ȷ��ͼ�����Ƿ��ж���������
        ������Ҫ�õ� Absdiff �������Ƚ���֡ͼ��֮���в���ĵ㣬��Ȼ��Ҫ��ͼ�����һЩ��������ƽ�������ҶȻ�������ֵ��������������֮��Ķ�ֵͼ���ϵĵ㽫����Ч��
'''

import cv2
import cv2.cv as cv
import numpy as np

print(cv2.__version__)

# �������룬�޸�file_num�ż���
file_num = 1
input = ['airport', 'hall', 'office', 'pedestrian', 'smoke']
video = 'data/noshake_static/' + input[file_num] + '/input.avi'
capture = cv.CaptureFromFile(video) # ���ļ���ȡͼƬcap

video_dynamic = 'data/noshake_dynamic/waterSurface/input.avi'
# capture = cv.CaptureFromFile(video_dynamic)

# video_shake = 'data/shake/people2/input.avi'
# capture = cv.CaptureFromFile(video_shake)

video_campus = 'data/Campus.avi'
video_shake = 'data/shake/people2/input.avi'
# capture = cv.CaptureFromFile(video)

frame1 = cv.QueryFrame(capture)

# �����Ƶ���ʼ��ߴ�
nbFrames = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))
codec = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FOURCC)
fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
duration = (nbFrames * fps) / 1000 # ʱ������
print 'Num. Frames = ', nbFrames # frameΪ֡����framesΪ��֡��
print 'Frame Rate = ', fps, 'fps' # fpsΪ�ļ���֡��
print 'Duration = ', duration, 'sec'
print 'codec = ', codec

# �������
out_list = ['airport', 'hall', 'office', 'pedestrian', 'smoke']
out_f = 'data/noshake_static/' + out_list[file_num] + '/yu_foreground' + '.avi'
out_m = 'data/noshake_static/' + out_list[file_num] + '/yu_mask' + '.avi'
# out_foreground = cv2.VideoWriter(out_f, -1, 30.0, (height, width)) # ��Ϊ�°�cv2�Ĵ洢��Ƶapi
# out_mask = cv2.VideoWriter(out_m, -1, 30.0, (height, width))
# writer=cv.CreateVideoWriter("output.avi", cv.CV_FOURCC("D", "I", "V", "X"), 5, cv.GetSize(temp), 1)
out_foreground=cv.CreateVideoWriter(out_f, int(codec), int(fps), (width,height), 1) #Create writer with same parameters
out_mask=cv.CreateVideoWriter(out_m, int(codec), int(fps), (width,height), 1) #Create writer with same parameters
# On linux I used to take "M","J","P","G" as fourcc

# print(frame1.height, frame1.width)

# �����м������frame1gray �� frame2gray
frame1gray = cv.CreateMat(height, width, cv.CV_8U) # CreateMat(rows, cols, type)
cv.CvtColor(frame1, frame1gray, cv.CV_RGB2GRAY)  # cv.CvtColor(src, dst, code) ������ͼ�����ͼ����ɫֵ
res = cv.CreateMat(height, width, cv.CV_8U)
frame2gray = cv.CreateMat(height, width, cv.CV_8U)
# frame2gray = np.array([height, width, cv2.CV_8U])

# gray = cv.CreateImage((width,height), cv.IPL_DEPTH_8U, 1)

w= width
h= height
nb_pixels = width * height

# ��ѭ��
index = 1 # ��֡����������GetCaptureProperty����
while True:
    # 1. ��ǰ֡��
    print('frame:', index)
    if index == nbFrames:
        print('finish.')
        break

    # print cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_POS_FRAMES) # frame == index

    # 2. ��ǰ֡frame2����frame2gray
    frame2 = cv.QueryFrame(capture)
    cv.CvtColor(frame2, frame2gray, cv.CV_RGB2GRAY) # ԭͼframe2ת��Ϊframe2gray��ʽ

    # 3. ͼ����֡������Ϊres
    cv.AbsDiff(frame1gray, frame2gray, res) # �Ƚ�frame1gray �� frame2gray �Ĳ��������res
    # res = cv2.absdiff(frame1gray, frame2gray) # ��Ϊcv2�°�api
    cv.ShowImage("After AbsDiff", res)

    # 4. ����res��Ϊǰ��Ŀ��foreground
    # cv.Convert(res, gray)
    # out_foreground.write(np.uint8(res)) # ����Ϊǰ��Ŀ��
    # out_foreground.write(np.asarray(cv.GetMat(res)))
    cv.WriteFrame(out_foreground, cv.GetImage(res)) # res��ʽΪcvmat��ʽ��ת��Ϊiplimage��ʽ

    # 5. ƽ������
    # cv.Smooth(res, res, cv.CV_BLUR, 5, 5) # �⻬һ��res

    # 6. ��̬ѧ�任�����ղ���
    element = cv.CreateStructuringElementEx(5*2+1, 5*2+1, 5, 5,  cv.CV_SHAPE_RECT) # CreateStructuringElementEx(cols, rows, anchorX, anchorY, shape, values=None)

    cv.MorphologyEx(res, res, None, None, cv.CV_MOP_OPEN) # cv2.morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]])
    cv.MorphologyEx(res, res, None, None, cv.CV_MOP_CLOSE) # ��̬ѧ�任��Ӧ�Ŀ� �ղ���

    # 7. ��ֵ����ֵ�����Եõ���ǰ��������ֵѡ��ȥ��αǰ��
    cv.Threshold(res, res, 10, 255, cv.CV_THRESH_BINARY) # ��ֵ�� cv.Threshold(src, dst, threshold, maxValue, thresholdType)


    # 8. blob��ȡ�������� ------------- δ��
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


    # cv.ShowImage("Image", frame2) # ��ǰԭʼͼ��


    # 9. ���洦�����res��Ϊmask
    cv.ShowImage("Res", res)
    # out_mask.write(np.array(res))  # ����Ϊmask
    cv.WriteFrame(out_mask, cv.GetImage(res))


    # #----------- ��������ʾ���ã������Ƿ���Ƶ�䶯
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

    # 10. ѭ����������������
    cv.Copy(frame2gray, frame1gray) # cv.Copy(src, dst, mask=None)
        # if mask != 0: dst = src
        # �൱��ѭ������
    index += 1

    # 11. ��Ƶ̫�̣������ӳٷ�����Ƶ
    c = cv.WaitKey(30) # �ȴ�ʱ��

    # 12. ��Ƶ̫������ǰ�˳�
    if c == 27: #Break if user enters 'Esc'.
        break

# �˳�����
# capture.release()
# out_foreground.release()
# out_mask.release()
cv2.destroyAllWindows()
