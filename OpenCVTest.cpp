#include<opencv2/opencv.hpp>
#include<iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include<windows.h>
#include<cstdio> 
#include "opencv2/core/core.hpp"
#include <iostream>
#include <conio.h>
using namespace std;
using namespace cv;

RNG rng(12345);
/*此类用于产生随机数
rng.uniform(1, 3);     在[1,3)区间，随机生成一个整数.
RNG rng(1234);      如果改成   RNG rng((unsigned)time(NULL));  代码每次运行结果都不一样.
*/

Mat image;
Mat image1;
Mat image2;
Mat image3;
Mat image4;


//获取轮廓的中心点
Point Center_cal(vector<vector<Point> > contours, int i)
{
	int centerx = 0, centery = 0, n = contours[i].size();
	//在提取的小正方形的边界上每隔周长个像素提取一个点的坐标，
	//求所提取四个点的平均坐标（即为小正方形的大致中心）
	centerx = (contours[i][n / 4].x + contours[i][n * 2 / 4].x + contours[i][3 * n / 4].x + contours[i][n - 1].x) / 4;
	centery = (contours[i][n / 4].y + contours[i][n * 2 / 4].y + contours[i][3 * n / 4].y + contours[i][n - 1].y) / 4;
	Point point1 = Point(centerx, centery);
	return point1;
}
int camera() {
	//namedWindow("图像采集窗口", cv::WINDOW_NORMAL);
	VideoCapture cap;
	// 读取摄像头
	cap.open(0);
	// 判断摄像头是否打开
	if (!cap.isOpened()) {
		std::cerr << "Could't open capture" << std::endl;
		return -1;
	}
	Mat frame;
	// 接收键盘上的输入
	char keyCode;
	// 保存的图片名称
	string imgName = "C:\\r.jpg";
	while (1) {

		// 把读取的摄像头传入Mat对象中
		cap >> frame;
		// 判断是否成功
		if (frame.empty()) {
			continue;
		}
		// 把每一帧图片表示出来
		namedWindow("shipin", WINDOW_NORMAL);
		imshow("shipin", frame);

		// 在300毫秒内等待是否存在键盘输入
		keyCode = waitKey(300);
		// 把图片保存起来
		imwrite(imgName, frame);
		image = frame;
		//imgName.at(0)++;
		frame.release();

		if (keyCode == 32) {//空格推出截取
			break;
		}
	}
	return 1;

}


int input() {
	return 1;
}


//此函数用于将截取到的图片进行定位输出
int output(Mat scr) {

	Mat src = scr;

	namedWindow("yuantu", WINDOW_NORMAL);// 显示窗口命名为input ；WINDOW_AUTOSIZE显示大小为图片自定义大小，且不可以更改大小
	imshow("yuantu", src); //显示原图

	Mat src_all = src.clone();
	Mat src_gray, drawing_gray;
	Mat img, img1;
	Mat threshold_output;
	//src = src + Scalar(100, 100, 100);
	//预处理cvtColor、blur、equalizeHist、threshold
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	//对图像进行平滑处理  
	blur(src_gray, src_gray, Size(3, 3));//模糊，去除毛刺
	//使灰度图象直方图均衡化  
	equalizeHist(src_gray, src_gray);
	//指定112阀值进行二值化
	
	// 使用内置的API
	//Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	//filter2D(src_gray, src_gray, src_gray.depth(), kernel);

	threshold(src_gray, threshold_output, 112, 255, THRESH_BINARY);
	namedWindow("threshold", WINDOW_NORMAL);
	imshow("threshold", threshold_output); //二值化后输出

	//需要的变量定义
	Scalar color = Scalar(1, 1, 255);
	vector<vector<Point>> contours, contours1, contours2, contours3;
	vector<Vec4i> hierarchy, hierarchy1, hierarchy2;
	Mat drawing = Mat::zeros(src.size(), CV_8UC3);
	Mat drawing1 = Mat::zeros(drawing.size(), CV_8UC3);
	Mat drawing2 = Mat::zeros(drawing1.size(), CV_8UC3);
	Mat drawing3 = Mat::zeros(drawing2.size(), CV_8UC3);
	Mat drawing4 = Mat::zeros(drawing3.size(), CV_8UC3);
	Mat drawing5 = Mat::zeros(drawing4.size(), CV_8UC3);
	Mat drawing6 = Mat::zeros(drawing5.size(), CV_8UC3);
	Mat drawing7 = Mat::zeros(drawing6.size(), CV_8UC3);
	Mat drawingAllContours = Mat::zeros(src.size(), CV_8UC3);

	//利用二值化输出寻找轮廓
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
	if (contours.size() == 0) {
		waitKey(4);//显示的毫秒时间，如果函数参数<=0表示一直显示。>0表示显示的时间
		destroyWindow("yuantu");
		destroyWindow("threshold");
		return -1;
	}
	else
	{
		int c = 0, ic = 0, k = 0, area = 0, x = 0;
		//通过黑色定位角作为父轮廓，有两个子轮廓的特点，筛选出三个定位角
		int parentIdx = -1, parentIdx_1 = -1;
		for (int i = 0; i < contours.size(); i++)//contours.size():表示轮廓的个数
		{
			//画出所有轮廓图
			drawContours(drawingAllContours, contours, parentIdx, CV_RGB(255, 255, 255), 1, 8);

			if (hierarchy[i][2] != -1 && ic == 0)//如果IC=0说明它是父级
			{
				parentIdx = i;//记录父级的轮廓号
				ic++;
			}
			else if (hierarchy[i][2] != -1)//有子轮廓
			{
				ic++;
			}
			else if (hierarchy[i][2] == -1)//无子轮廓，重置操作。
			{
				ic = 0;
				parentIdx = -1;
			}

			//找到定位点信息
			if (ic >= 2) //有两层嵌套轮廓时
			{

				printf("输出第个%i点的轮廓编号", x);
				cout << parentIdx << endl;
				x++;
				//保存找到的三个黑色定位角
				contours1.push_back(contours[parentIdx]);//添加父轮廓(最外层)。push_back是在vector类中作用为在vector尾部加入一个数据。

				/*参数意义如下：
				//Hierarchy:层次结构
				//contours:轮廓
					image---轮廓画到image上。
					contours---待画图的轮廓，每个轮廓都是一个由像素的坐标值组成的向量。
					contourIdx---指定哪些轮廓需要被绘制，如果这个值为负，则表示所有的轮廓都需要绘制。
					color---轮廓颜色。
					thickness---轮廓线宽，如果参数为负数，则绘制轮廓的内部。
					lineType---线类型。
					hierarchy---可选参数，表示轮廓的拓扑结构。
					maxLevel---表示要绘制轮廓的最大层级。在参数hierarchy有效的情况下，这个参数为0表示只绘制指定的轮廓；为1表示绘制所有的外轮廓和内嵌轮廓；为2表示绘制所有的外轮廓、内嵌轮廓和内嵌轮廓之间的联接轮廓。以此类推.....
					offset：可选的轮廓偏移参数，按指定的移动距离绘制所有的轮廓。*/
					//画出三个黑色定位角的轮廓
				drawContours(drawing, contours, parentIdx, CV_RGB(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 3, 8);
				ic = 0;
				parentIdx = -1;

			}
		}
		if (contours1.size() < 3) {//如果找到的轮廓数量小于3 就返回异常
			waitKey(4);//显示的毫秒时间，如果函数参数<=0表示一直显示。>0表示显示的时间
			destroyWindow("yuantu");
			destroyWindow("threshold");
			return -1;
		}
		else {

			//画出所有轮廓图
			namedWindow("DrawingAllContours");
			imshow("DrawingAllContours", drawingAllContours);
			//画出三个黑色定位角的轮廓
			namedWindow("Drawing");
			imshow("Drawing", drawing);

			//填充的方式画出三个黑色定位角的轮廓
			for (int i = 0; i < contours1.size(); i++) {
				drawContours(drawing2, contours1, i, CV_RGB(rng.uniform(100, 255), rng.uniform(100, 255), rng.uniform(100, 255)), -1, 4, hierarchy[k][2], 0, Point());
			}

			//填充的方式画出三个黑色定位角的轮廓
			namedWindow("Drawing2");
			imshow("Drawing2", drawing2);

			Point point[10];
			//填充的方式画出三个黑色定位角的轮廓
			for (int i = 0; i < contours1.size(); i++)
			{
				point[i] = Center_cal(contours1, i);
				printf("输出第%i个点的中心坐标", i);
				cout << point[i] << endl;
			}

			//计算轮廓的面积，计算定位角的面积，从而计算出边长
			area = contourArea(contours1[1]);

			int area_side = cvRound(sqrt(double(area)));
			cout << "图形面积为" << area << endl;
			cout << "图形面积为" << double(area) << endl;
			cout << "图形面积开平方根为" << sqrt(double(area)) << endl;
			cout << "图形边长为" << cvRound(sqrt(double(area))) << endl;


			for (int i = 0; i < contours1.size(); i++)
			{
				cout << "轮廓" << i % contours1.size() << "链接轮廓" << (i + 1) % contours1.size() << endl;
				cout << "轮廓" << i << " 的面积：" << contourArea(contours1[i]) << endl;
				cout << "  周长约：" << arcLength(contours1[i], true) << endl;
				cout << "  边长约：" << cvRound(arcLength(contours1[i], true) / 4) << endl;
				//画出三个定位角的中心连线
				//line(drawing2, point[i % contours1.size()], point[(i + 1) % contours1.size()], color, area_side / 2, 8);

				line(drawing2, point[i % contours1.size()], point[(i + 1) % contours1.size()], color, area_side, 8);
			}

			namedWindow("Drawing2_1");
			imshow("Drawing2_1", drawing2);
			//接下来要框出这整个二维码
			Mat gray_all, threshold_output_all;


			cvtColor(drawing2, gray_all, COLOR_BGR2GRAY);
			namedWindow("drawing2_1_gray_all");
			imshow("drawing2_1_gray_all", gray_all);

			threshold(gray_all, threshold_output_all, 45, 255, THRESH_BINARY);
			namedWindow("drawing2_1_gray_all_threshold_output_all");
			imshow("drawing2_1_gray_all_threshold_output_all", threshold_output_all);

			findContours(threshold_output_all, contours2, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));//RETR_EXTERNAL表示只寻找最外层轮廓


			cout << "最终轮廓数量：" << contours2.size() << endl;
			Point2f fourPoint2f[4];
			RotatedRect rectPoint = minAreaRect(contours2[0]);

			//imshow("222", threshold_output_all);
			/*RotatedRect minAreaRect(InputArray points)
			points：输入信息，可以为包含点的容器(vector)或是Mat。
			RotatedRect ：返回一个轮廓的外接矩形，是一个RotatedRect的类。
			包覆输入信息的最小斜矩形，是一个Box2D结构rect：
			（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）。
			但是要绘制这个矩形，我们需要矩形的4个顶点坐标box,
			通过函数 cv2.cv.BoxPoints() 获得，返回形式[ [x0,y0], [x1,y1], [x2,y2], [x3,y3] ]。
			得到的最小外接矩形的4个顶点顺序、中心坐标、宽度、高度、旋转角度（是度数形式，不是弧度数）
			*/

			cout << "最小包围矩形's angle:" << rectPoint.angle << endl;
			cout << "最小包围矩形's center:" << rectPoint.center << endl;
			cout << "最小包围矩形's width:" << rectPoint.size.width << endl;
			cout << "最小包围矩形's height:" << rectPoint.size.height << endl;
			cout << "最小包围矩形's area:" << rectPoint.size.area() << endl;
			cout << endl;

			// 对得到的轮廓填充一下
			drawContours(drawing4, contours2, -1, Scalar(255, 255, 255), FILLED);
			namedWindow("drawing4");
			imshow("drawing4", drawing4);

			//将rectPoint变量中存储的坐标值放到 fourPoint的数组中
			rectPoint.points(fourPoint2f);

			for (int i = 0; i < 4; i++)
			{

				cout << fourPoint2f[i] << endl;
				cout << "端点" << fourPoint2f[i % 4] << "链接" << fourPoint2f[(i + 1) % 4] << endl;

				line(drawing4, fourPoint2f[i % 4], fourPoint2f[(i + 1) % 4], Scalar(0, 0, 255), 3);
			}
			namedWindow("drawing4_1");
			imshow("drawing4_1", drawing4);

			cvtColor(drawing4, drawing5, COLOR_BGR2GRAY);
			findContours(drawing5, contours3, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
			drawContours(drawing5, contours3, -1, Scalar(255), FILLED);
			namedWindow("drawing5");
			imshow("drawing5", drawing5);
			//抠图到drawing6

			drawing6.setTo(255);

			src.copyTo(drawing6, drawing5);
			namedWindow("drawing6", 1);
			imshow("drawing6", drawing6);
			imwrite("C:\\r1.jpg", drawing6);
			Rect rect = boundingRect(Mat(contours3[0]));
			cout << "第1个最小包围矩形 area:" << rect.area() << endl;
			//求最小包围矩形
			RotatedRect rectPoint_1 = minAreaRect(contours3[0]);//minAreaRect()函数计算并返回指定点集的最小区域边界斜矩形。
			cout << "最小包围矩形's area:" << rectPoint_1.size.area() << endl;

			float angle = rectPoint.angle;
			Point2f center = rectPoint.center;  //中心点
			if (angle != 90 && angle != -90)
			{
				angle = angle - 90;//有问题
				Mat M2 = getRotationMatrix2D(center, angle, 1);//计算旋转加缩放的变换矩阵 
				warpAffine(drawing6, drawing7, M2, src_all.size(), 1, 0, Scalar(0));//仿射变换 
				namedWindow("drawing7", WINDOW_NORMAL);
				imshow("drawing7", drawing7);
				img = drawing7(rect);
				namedWindow("img", WINDOW_NORMAL);
				imshow("img", img);
				imwrite("C:\\r1.jpg", img);
			}
			else {
				namedWindow("drawing7", WINDOW_NORMAL);
				imshow("drawing7", drawing6);
				img = drawing6(rect);
				namedWindow("img", WINDOW_NORMAL);
				imshow("img", img);
				imwrite("C:\\r1.jpg", img);
			}

			image4 = img;
			waitKey(4);//显示的毫秒时间，如果函数参数<=0表示一直显示。>0表示显示的时间
			destroyWindow("yuantu");
			destroyWindow("threshold");
			destroyWindow("DrawingAllContours");
			destroyWindow("Drawing");
			destroyWindow("Drawing2");
			destroyWindow("Drawing2_1");
			destroyWindow("drawing2_1_gray_all");
			destroyWindow("drawing2_1_gray_all_threshold_output_all");
			destroyWindow("drawing4");
			destroyWindow("drawing4_1");
			destroyWindow("drawing5");
			destroyWindow("drawing6");
			destroyWindow("drawing7");
			destroyWindow("img");

			return 1;

		}
	}

}


int windows() {
	int x = MessageBox(GetForegroundWindow(), "已完成识别，是否继续。", "OpenCVTest", 3);
	printf("%d\n", x);
	return x;
}



int main(int argc, char** argv) {

	while (true) {
		//调用摄像头传入图像
		camera();
		Mat src = image;
		//Mat src = imread("C:\\r.jpg");
		//imshow("mmm", image);
		if (output(src) == 1) {
			//对拿到的imagge进行操作
			cout << "001" << endl;
			//waitKey(2000);//显示的毫秒时间，如果函数参数<=0表示一直显示。>0表示显示的时间
			//destroyWindow("mmm");
			//continue;
			int x = windows();
			if (x == 7 || x == 2) {
				break;
			}
			else
			{
				continue;
			}

		}
		else {
			cout << "-1" << endl;
			waitKey();//显示的毫秒时间，如果函数参数<=0表示一直显示。>0表示显示的时间
			destroyWindow("mmm");

			/*如果一直没有识别到就无法返回，这是一个问题。
			char keyCode = waitKey(300);
			if (keyCode == 27) {
				destroyAllWindows();
				break;
			}*/
			continue;
		}

	}
	destroyAllWindows();

}
