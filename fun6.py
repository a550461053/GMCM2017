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
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import preprocessing

print(cv2.__version__)

# x = [0.92052284949507324, 0.98971537862358705, 0.76169271922177861, 0.98848679968394748, 0.89683460290619033, 0.7482079301691541, 0.9902365529813606, 0.78983344634410635, 0.99157764637840529, 0.77838060277251631, 0.38601642402438735, 0.99125462344184045, 0.91108260653506257, 0.99092651416265065, 0.29942088786560883, 0.30569199873537595, 0.90186448576991007, 0.90710941965016634, 0.77732781012856045, 0.89703013038549606, 0.76953571199412785, 0.99285975203675925, 0.7855065844071567, 0.91699938307798334, 0.8977364345943506, 0.90091179744682159, 0.99316948246970194, 0.73920721202156692, 0.99383570442262092, 0.29672818594055395, 0.9172225745269037, 0.99390319501540936, 0.79099143277906203, 0.79634651229401809, 0.2774615705834762, 0.78158987803347957, 0.90426368946430746, 0.26640875509041051, 0.91443148951886422, 0.91936410502846921, 0.91518463343775913, 0.98623085598470073, 0.78556449875368295, 0.80554124182376297, 0.92324780357160297, 0.91895426781660661, 0.77900672373385205, 0.99025721867665273, 0.79654214402195578, 0.91758451654816464, 0.79133082057615456]
#
# y = [0.38611011093902242, 0.12596225871799188, 0.64132190362649633, 0.13457106580503048, 0.43396670773272467, 0.65247189209505474, 0.12221407625641068, 0.60980175896500088, 0.11571997230969594, 0.62277475121899906, 0.90632033798360734, 0.11600418862304113, 0.40807991206796612, 0.12089220358213065, 0.94651883130596115, 0.94586122629463087, 0.42519180167117815, 0.4135259531963052, 0.62417005198797271, 0.43561204382036611, 0.63472748889802233, 0.10421280626639473, 0.61200000834177282, 0.39535098848879391, 0.43355219191004318, 0.42725223412875241, 0.10430222927796384, 0.66982421789307933, 0.097683234025173538, 0.94879487910890692, 0.39471154358754112, 0.10147915682998943, 0.60463784019420741, 0.59804132526139542, 0.95610002157631102, 0.61715848914038651, 0.42075353163681017, 0.95957539188872265, 0.39952074202517268, 0.38869627130249335, 0.39752161944492459, 0.14615956811402864, 0.61003768146831938, 0.58692610383835309, 0.379428757032793, 0.38923395850600406, 0.62059004845582766, 0.12836713095942659, 0.59947205542015758, 0.39132162017389627, 0.60663405053527908]
# print(len(x), len(y))
# x = x[1:4]
# y = y[1:4]
# # x = np.arange(1, 17, 1)
# # y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
# z1 = np.polyfit(x, y, 2)#用3次多项式拟合
# p1 = np.poly1d(z1)
# print(p1) #在屏幕上打印拟合多项式
# yvals=p1(x)#也可以使用yvals=np.polyval(z1,x)
# plot1=plt.plot(x, y, '*', label='original values')
# plot2=plt.plot(x, yvals, 'r',label='polyfit values')
# plt.show()
# input()

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

def hog_test(pic):
    img = cv2.imread(pic)
    hog = cv2.HOGDescriptor((32, 64), (16, 16), (8, 8), (8, 8), 9)
    svm = pk.load(open('svm.pickle'))
    hog.setSVMDetector(np.array(svm))
    del svm
    found , w = hog.detectMultiScale(img)

# 二次拟合
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot #准确率

    plot1 = plt.plot(x, y, '*', label='original values')
    plot2 = plt.plot(x, yhat, 'r', label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('polyfitting')
    plt.show()
    # plt.savefig('p1.png')
    return results


x=[ 1 ,2  ,3 ,4 ,5 ,6]
y=[ 2.5 ,3.51 ,4.45 ,5.52 ,6.47 ,7.2]
# z1 = polyfit(x, y, 4)
# print z1

def svm_test():
    # xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Generate train data
    # X = 0.3 * np.random.randn(100, 2)
    # X_train = np.r_[X + 2, X - 2]
    X_train = [[1,1], [1,-1],[1]]

    # Generate some regular novel observations
    # X = 0.3 * np.random.randn(20, 2)
    # X_test = np.r_[X + 2, X - 2]
    X_test = [[1, 2], [-1, 1], [-1, -1], [1, 1]]

    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)

    # predict the results
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    print(y_pred_test)  # -1 is not in oneclass
                        # 1 is in oneclass

    # return the error:
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    print(n_error_test)

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plt.title("Novelty Detection")
    # plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    # a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    # plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    # s = 40
    # b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    # b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
    #                  edgecolors='k')
    # c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
    #                 edgecolors='k')
    # plt.axis('tight')
    # plt.xlim((-5, 5))
    # plt.ylim((-5, 5))
    # plt.legend([a.collections[0], b1, b2, c],
    #            ["learned frontier", "training observations",
    #             "new regular observations", "new abnormal observations"],
    #            loc="upper left",
    #            prop=matplotlib.font_manager.FontProperties(size=11))
    # plt.xlabel(
    #     "error train: %d/200 ; errors novel regular: %d/40 ; "
    #     "errors novel abnormal: %d/40"
    #     % (n_error_train, n_error_test, n_error_outliers))
    # plt.show()
# svm_test()



def kmeans_test(data, degree, test):
    # data = np.random.rand(100, 3) #生成一个随机数据，样本大小为100, 特征数为3
    # print(type(data))
    #假如我要构造一个聚类数为3的聚类器
    estimator = KMeans(n_clusters=degree)#构造聚类器
    estimator.fit(data)#聚类
    # label_pred = estimator.label_ #获取聚类标签
    test_predict = estimator.predict(test)
    centroids = estimator.cluster_centers_ #获取聚类中心
    inertia = estimator.inertia_ # 获取聚类准则的总和
    # print(inertia, test_predict)
    # return test_predict
    return centroids

# kmeans_test()

def normalization_test(x):
    # x = [[1.,-1., 2.],
    #      [2., 0., 0.],
    #      [0., 1., -1.]]
    # print(x)
    x_normalization = preprocessing.normalize(x, norm='l2')
    print(x_normalization)
    return(x_normalization)
# normalization_test()


def load_p(path):
    node = []
    sum1 = 0
    print('the path is:', path)
    with open('data/q6/'+ path +'.txt', 'r') as file:
        lines = file.readlines()
        F = True
        for line in lines:
            if F:
                fps = float(line.split()[0])
                F = False
                sum1 += 1
                continue
            # print('123123:', line.split())
            if int(line.split()[0]) == 0: # 不存在坐标
                # if int(line) == 0:
                sum1 += 1
                continue
            if len(line.split()) > 1:
                continue
            length = int(line)
            # print('the length is:', length)
            pos = []
            for i in range(0, length):
                if len(lines[i + sum1 + 1].split()) == 1 and int(lines[i + sum1 + 1].split()[0]) == 0:
                    # sum1 += 1
                    continue
                tmp = lines[i + sum1 + 1].split()[1:3]
                # if len(lines[i + sum1 + 1].split()) == 1:
                # print(lines[i + sum1 + 1].split(), len(lines[i + sum1 + 1].split()))
                # print(tmp)
                pos.append([int(tmp[0]), int(tmp[1])])
            node.append(pos)
            sum1 += length + 1
        # print(len(node))
        # print(node[0])
    return node

def all_test():
    res = []
    p = []
    pb = []
    for i in ['p1', 'p2', 'p3', 'p4']:
        p.append(load_p(i))
    for i in ['pb1', 'pb2', 'pb3', 'pb4']:
        pb.append(load_p(i))

    res.append(p)
    res.append(pb)

    p1 = []
    for i in p[0]:
        p1.append(normalization_test(i))
    p1_1 = []
    for i in p1[0]:
        # print(i)
        p1_1.append(i)
    print(len(p1_1))

    # p1_np = np.array(p1)
    # print(p1_np)
    print(p1_1)
    p1_kmeans = kmeans_test(p1_1, 3, p1_1[1])

    print(p1_kmeans)
    # input()

    # 正则化
    nor_p, nor_pb = [], []
    for i in range(0, 4):
        tmp = []
        # print(i, len(p[i])) # 0, 51帧数
        # input()
        for j in range(0, len(p[i])):
            # print(j, len(p[i][j]))
            # input()
            if len(p[i][j]) < 4: # 一张图片所含点数太少则略过
                continue
            tmp.append(normalization_test(p[i][j]))
        nor_p.append(tmp)
    for i in range(0, 4):
        tmp = []
        for j in range(0, len(pb[i])):
            if len(pb[i][j]) < 4:
                continue
            tmp.append(normalization_test(pb[i][j]))
        nor_pb.append(tmp)
    print('normalization finished.')
    # nor_p = [normalization_test(j for j in p[i]) for i in range(0, 4)]
    # nor_pb = [normalization_test(j for j in pb[i]) for i in range(0, 4)]

    # 开始聚类坐标点, 一个视频的一张frame有一个聚类中心，一个视频有frame个聚类中心
    kmeans_p, kmeans_pb = [], []
    for i in range(0, len(nor_p)):
        # print(i, len(nor_p), len(nor_p[i]))
        # input()
        tmp = []
        for j in range(0, len(nor_p[i])):
            tmp.append(kmeans_test(nor_p[i][j], 4, [1, 1]))
        kmeans_p.append(tmp)
        # kmeans_p = [kmeans_test(p[i], 3, p1_1[1]) for i in range(0, len(p))]
    for i in range(0, len(nor_pb)):
        # print(i)
        tmp = []
        for j in range(0, len(nor_pb[i])):
            tmp.append(kmeans_test(nor_pb[i][j], 4, [1, 1]))
        kmeans_pb.append(tmp)
    # print(kmeans_p)
    print('kmeans finish.')

    ###########


    # 聚类中心轨迹拟合
    print(len(kmeans_p)) # 聚类中心的个数==帧数
    # input()
    x1, x2, x3, x4, y1, y2, y3, y4 = [], [], [], [], [], [], [], []  # 存放4个点的坐标

    for i in range(0, len(kmeans_p)):
        # kmeans_p[i] # 每个视频
        d1, d2, d3, d4 = [], [], [], []
        for j in range(0, len(kmeans_p[i])): # 每个图
            # print(len(kmeans_p[i])) # 51对坐标点,frame
            # print(len(kmeans_p[i][j])) # 4个聚类中心
            # input()
            d1.append(kmeans_p[i][j][0]) # 每个图的4个聚类中心
            d2.append(kmeans_p[i][j][1])
            d3.append(kmeans_p[i][j][2])
            d4.append(kmeans_p[i][j][3])
        x1.append([d1[i][0] for i in range(len(d1))])
        x2.append([d2[i][0] for i in range(len(d2))])
        x3.append([d3[i][0] for i in range(len(d3))])
        x4.append([d4[i][0] for i in range(len(d4))])
        y1.append([d1[i][1] for i in range(len(d1))])
        y2.append([d2[i][1] for i in range(len(d2))])
        y3.append([d3[i][1] for i in range(len(d3))])
        y4.append([d4[i][1] for i in range(len(d4))])

        # print(len(x1), x1)
        print(x1[0])
        print()
        print(y1[0])
        node1 = polyfit(x1[0], y1[0], 1)
        node2 = polyfit(x2[0], y2[0], 2)
        node3 = polyfit(x3[0], y3[0], 3)

        # x = np.arange(1, 17, 1)
        # y = np.array(
        #     [4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
        x = x1[0]
        y = y1[0]
        z1 = np.polyfit(x, y, 3)  # 用3次多项式拟合
        p1 = np.poly1d(z1)
        print(p1)  # 在屏幕上打印拟合多项式
        yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
        plot1 = plt.plot(x, y, '*', label='original values')
        plot2 = plt.plot(x, yvals, 'r', label='polyfit values')

        # x_data, y_data = [], []
        # # x_data_list, y_data_list = [], []
        # for i in range(0, 4): # 4个视频
        #     tmp = []
        #     # print(len(x[i]), x[i])
        #     # input()
        #     for j in range(0, len(x[i])): # 51个frame帧
        #         tmp.append(x[i][j])
        #     x_data.append(tmp)
        # # x_data_list.append(x_data)
        #
        # for i in range(0, 4): # 4个视频
        #     tmp = []
        #     for j in range(0, len(y[i])): # 51个frame帧
        #         tmp.append(y[i][j])
        #     y_data.append(tmp)
        # print(len(x_data), x_data)
        # print(len(x_data[0]), x_data[0])
        # print(len(x_data[0]))
        # input()
        # node_0_0 = polyfit(x_data[0][0], y_data[0][0], 3) # 第0个视频的第0个聚类中心
        # print(node_0_0)
        # node_0_1 = polyfit(x_data[0][1], y_data[0][1], 3)
        # node_0_0 = polyfit(x_data[0][0], y_data[0][0], 3)
        # node_0_0 = polyfit(x_data[0][0], y_data[0][0], 3)
        # # polyfit(x[2], y[2], 3)
        # # polyfit(x[3], y[3], 3)
        # input()

all_test()

def v_avg(num):
    with open('data/q6/v' + num + '.txt') as file:
        lines = file.readlines()
        res = 0
        index = 0
        for line in lines:
            res += float(line.split()[0])
            index += 1
        return res / index

v = []
for i in range(1,5):
    v.append(v_avg(str(i)))
vb = []
for i in range(1,5):
    vb.append(v_avg('b'+str(i)))
print(v, vb)

'''
向量定义：
    [总目标数，聚类中心的坐标点，聚类中心拟合轨迹，目标的平均速度]
'''
# video1_pic1 = [41, [[0.1, 0.9], [0.3, 0.7], [0.2, 0.8], [0.4, 0.7]], v[0]]
# video1_pic2 = [51, [[0.1, 0.9], [0.3, 0.7], [0.2, 0.8], [0.4, 0.7]], v[1]]
# video1_pic3 = [21, [[0.1, 0.9], [0.3, 0.7], [0.2, 0.8], [0.4, 0.7]], v[2]]
# video1_pic4 = [31, [[0.1, 0.9], [0.3, 0.7], [0.2, 0.8], [0.4, 0.7]], v[3]]
# video1_pic5 = [11, [[0.1, 0.9], [0.3, 0.7], [0.2, 0.8], [0.4, 0.7]], v[4]]
# video1 = [video1_pic1, video1_pic2, ...]


if __name__ == '__main__':


    noshake_static = 'data/noshake_static/hall/input.avi'
    out_path_static = 'data/noshake_static/hall/fun2out.avi'
    out_path_static_mask = 'data/noshake_static/hall/fun2mask.avi'

    video_dynamic = 'data/noshake_dynamic/waterSurface/input.avi' # input foreground
    out_path_dynamic = 'data/noshake_dynamic/waterSurface/fun2out.avi'
    out_path_dynamic_mask = 'data/noshake_dynamic/waterSurface/fun2mask.avi'

    video_shake = 'data/shake/people2/people2.avi' # input
    out_path_shake = 'data/shake/people2/fun2out.avi'
    out_path_shake_mask = 'data/shake/people2/fun2mask.avi'

    # test function:
    # MOG_test(noshake_static, out_path_static, out_path_static_mask)  # 静态效果一般
    # save(out_path_static, out_path_static_mask)
    # MOG2_test(noshake_static, out_path_static)  # 效果需要调试

    # MOG_test(video_dynamic, out_path_dynamic, out_path_dynamic_mask) # 动态效果不错
    # MOG2_test(video_dynamic, out_path_dynamic) # 效果很差，。。

    # MOG_test(video_shake, out_path_shake, out_path_shake_mask)
    # MOG2_test(video_shake, out_path_shake)
