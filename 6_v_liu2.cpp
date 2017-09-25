#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

int  get_distance(int x1,int x2,int y1,int y2)
{
	return (sqrt(pow(abs(x2-x1),2)+ pow(abs(y2 - y1), 2)));
}

struct point1 {
	int index;
	int x;
	int y;
};
#define num 20
point1 pre_arr[num];
point1 cur_arr[num];
//int num = 20;
float v_zhen;
float T_video;/// = 1000 / v_zhen; //周期ms
ofstream fout;

void clear_pre_arr()
{
	int i;
	for (i = 0; i < num; i++) {
		pre_arr[i].x = 0;
		pre_arr[i].index = 0;
		pre_arr[i].y = 0;
	}
}

void clear_cur_arr()
{
	int i;
	for (i = 0; i < num; i++) {
		cur_arr[i].x = 0;
		cur_arr[i].index = 0;
		cur_arr[i].y = 0;
	}
}

void get_v()
{
	int i, j;
	int min = 10000;
	int max = 500;
	float v = 0.0;
	int distance;
	for (i=0;i<num;i++)
	{
		for (j = 0; j < num; j++) 
		{
			distance = get_distance(pre_arr[i].x,cur_arr[j].x,pre_arr[i].y,cur_arr[j].y);
			//if (distance < max) {
			if (distance < min) {
				min = distance;
				//cout << "执行了" << endl;
			}
			//
		}
		if(min<max){
			v = (min * 0.1 * 10) / (T_video * 0.1 * 10);
			if (v != 0.0) {
				/*if (v < 3) {
					fout << v << endl;
				}*/
				fout << v << endl;
					
			}
			
			//cout << "速度为： " << v << endl;
		}
		min = 10000;
		

	}
	//return v;
}


int main()
{
	//vector<>


	ifstream fin;
	fin.open("E:\\数模\\video_6\\坐标数据\\异常-1-1.txt",ios::in);
	float v = 0.0;

	
	fout.open("E:\\数模\\video_6\\坐标数据\\行人速度-异常-1.txt",ios::out);

	int count, ch;
	fin >> v_zhen;
	T_video = 1000 * 0.1 * 10 / (v_zhen * 0.1 * 10);
	fin >> count;
	int i, j;
	int a, b, c;
	for (i = 0; i < count; i++) {
		fin >> a;
		fin >> b; 
		fin >> c;
		pre_arr[i].index = a;
		pre_arr[i].x = b;
		pre_arr[i].y = c;

	}
	fin >> count;
	for (i = 0; i < count; i++) {
		fin >> a;
		fin >> b;
		fin >> c;
		cur_arr[i].index = a;
		cur_arr[i].x = b;
		cur_arr[i].y = c;

	}
	while (!fin.eof()) {

		fout.setf(ios::fixed, ios::floatfield);
		fout.precision(4);

		get_v();
		//cout << "速度为： " << v << endl;
		//fout << v << endl;

		clear_pre_arr();

		for (i = 0; i < count; i++) {

			pre_arr[i].index = cur_arr[i].index;
			pre_arr[i].x = cur_arr[i].x;
			pre_arr[i].y = cur_arr[i].y;

		}

		clear_cur_arr();

		fin >> count;
		for (i = 0; i < count; i++) {
			fin >> a;
			fin >> b;
			fin >> c;
			cur_arr[i].index = a;
			cur_arr[i].x = b;
			cur_arr[i].y = c;

		}
	}
	system("pause");
	return 0;
}

