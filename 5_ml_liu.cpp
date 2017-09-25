#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace cv;

int main()
{

	VideoCapture capture("E:\\数模\\video_6\\正常-4.avi");
	ofstream fout;
	fout.open("E:\\数模\\video_6\\6题视频正常-1-4.txt", ios::out);
	//检测是否正常打开:成功打开时，isOpened返回ture
	if (!capture.isOpened()) {
		cout << "fail to open!" << endl;
		system("pause");
	}

	//获取整个帧数
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "整个视频共" << totalFrameNumber << "帧" << endl;

	//设置开始帧()
	long frameToStart = 1;
	long frameToStop = totalFrameNumber   ;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);

	//获取帧率
	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "帧率为:" << rate << endl;

	//定义一个用来控制读取视频循环结束的变量
	bool stop = false;
	//承载每一帧的图像
	Mat frame;
	//显示每一帧的窗口

	/////////////
	
	
	//namedWindow("Extracted frame");
	//两帧间的间隔时间:
	int delay = 1000 / rate;

	//利用while循环读取帧
	//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量
	long currentFrame = frameToStart;

	//滤波器的核
	int kernel_size = 3;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
	int count = 0;
	fout << rate << endl;
	while (!stop)
	{
		count++;
		if(count%10==1)
			cout << "第" << count << "帧" << endl;
		//读取下一帧
		if (!capture.read(frame))
		{
			cout << "读取视频失败" << endl;
			return -1;
		}
		//这里加滤波程序

		//imshow("Extracted frame", frame);
		filter2D(frame, frame, -1, kernel);


		int c = waitKey(delay);
		//按下esc或者到达指定的结束帧后退出读取视频
		if ((char)c == 27 || currentFrame > frameToStop)
		{
			stop = true;
		}
		//按下按键后会停留在当前帧，等待下一次按键
		if (c >= 0)
		{
			waitKey(0);
		}
		currentFrame++;





		//cout << "Red:Hog+svm------Green:Hog+cascade" << endl;
		Mat src = frame;
		vector<Rect> found1, found_filtered1, found2, found_filtered2;//矩形框数组

		clock_t start1, end1, start2, end2;//
		//方法1，Hog+svm
		//start1 = clock();
		//HOGDescriptor hog;//HOG特征检测器
		//hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//设置SVM分类器为默认参数	
		//hog.detectMultiScale(src, found1, 0, Size(2, 2), Size(0, 0), 1.05, 2);//对图像进行多尺度检测，检测窗口移动步长为(8,8)
		//end1 = (double)(1000 * (clock() - start1) / CLOCKS_PER_SEC);
		//方法2.Hog+cascade
		start2 = clock();
		CascadeClassifier *cascade = new CascadeClassifier;
		cascade->load("hogcascade_pedestrians.xml");
		cascade->detectMultiScale(src, found2);
		end2 = (double)(1000 * (clock() - start2) / CLOCKS_PER_SEC);

		//cout << "Hog+svm:  " << end1 << "ms" << "    Hog+cascade:  " << end2 << "ms" << endl;
		//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
		/*for (int i = 0; i < found1.size(); i++)
		{
			Rect r = found1[i];
			int j = 0;
			for (; j < found1.size(); j++)
				if (j != i && (r & found1[j]) == r)
					break;
			if (j == found1.size())
				found_filtered1.push_back(r);
		}*/
		for (int i = 0; i < found2.size(); i++)
		{
			Rect r = found2[i];
			int j = 0;
			for (; j < found2.size(); j++)
				if (j != i && (r & found2[j]) == r)
					break;
			if (j == found2.size())
				if(r.width>110)
					found_filtered2.push_back(r);
		}

		//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
		/*for (int i = 0; i<found_filtered1.size(); i++)
		{
			Rect r = found_filtered1[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(src, r.tl(), r.br(), Scalar(0, 0, 255), 3);
		}*/
		fout << found_filtered2.size()<<endl;
		for (int i = 0; i<found_filtered2.size(); i++)
		{
			Rect r = found_filtered2[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			
			rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);
				
			fout <<(i+1)<<"  "<< (r.tl().x + r.br().x) / 2 << "  " << (r.tl().y + r.br().y) / 2 << endl;

		}
		//imshow("src", src);

	}
	capture.release();
	waitKey(10);
	system("pause");
	return 0;
}