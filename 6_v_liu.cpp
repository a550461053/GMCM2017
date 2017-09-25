#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <fstream>

using namespace std;
using namespace cv;

ofstream fout;
string face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "人脸识别";

void detectAndDisplay(Mat frame);

int main() {

	
	fout.open("E:\\数模\\video_6\\异常-1-4.txt", ios::out);

	VideoCapture capture("E:\\数模\\video_6\\异常-4.avi");
	//ofstream fout;
	//fout.open("6题视频1.txt", ios::out);
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
	long frameToStop = totalFrameNumber-20 ;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);

	//获取帧率
	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "帧率为:" << rate << endl;

	//定义一个用来控制读取视频循环结束的变量
	bool stop = false;
	//承载每一帧的图像
	Mat frame;
	//显示每一帧的窗口
	//////////////////
	//namedWindow("Extracted frame");
	//两帧间的间隔时间:
	int delay = 1000 / rate;

	//利用while循环读取帧
	//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量
	long currentFrame = frameToStart;

	//滤波器的核
	int kernel_size = 3;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);


	if (!face_cascade.load(face_cascade_name)) {
		printf("[error] 无法加载级联分类器文件！\n");
		system("pause");
		return -1;
	}
	int ii = 0 ;
	fout << "每帧的圆形个数不一样，请注意，有标号" << endl;
	fout << rate << endl;
	while (!stop)
	{
		ii++;
		//读取下一帧
		if (!capture.read(frame))
		{
			cout << "读取视频失败" << endl;
			return -1;
		}
		//这里加滤波程序
		//imshow("Extracted frame", frame);
		filter2D(frame, frame, -1, kernel);

		//	cout << "正在读取第" << currentFrame << "帧" << endl;

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

		if (ii % 10 == 0) {
			cout << "第"<<ii<<"帧" << endl;
		}

		Mat image = frame;




		detectAndDisplay(image);

		//waitKey(0);
		//imshow("src",image);
	}
	cout << "数据处理完成1" << endl;
	fout.close();
	return 0;
}

void detectAndDisplay(Mat frame) {
	std::vector<Rect> faces;
	Mat frame_gray;


	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	fout << faces.size() << endl;
	for (int i = 0; i < faces.size(); i++) {
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		
		fout <<(i+1)<<"  "<< (center.x) << "  " << (center.y) << endl;
	}
	

	//imshow(window_name, frame);
}