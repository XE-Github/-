//"C:\\Users\\Leon-MiWork\\Downloads\\001.png"
//"C:\\Users\\Leon-MiWork\\Pictures\\2.jpg
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include<windows.h>
#include <direct.h>
#include<cstdio> 
#include<opencv2/opencv.hpp>
#include<iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/core/core.hpp"
#include <iostream>
#include <conio.h>
using namespace std;
using namespace cv;

string folderPath = "C:\\testFolder";
string command;
string rd;

RNG rng(12345);
/*�������ڲ��������
rng.uniform(1, 3);     ��[1,3)���䣬�������һ������.
RNG rng(1234);      ����ĳ�   RNG rng((unsigned)time(NULL));  ����ÿ�����н������һ��.
*/

Mat image;
Mat image1;
Mat image2;
Mat image3;
Mat image4;


//��ȡ���������ĵ�
Point Center_cal(vector<vector<Point> > contours, int i)
{
	int centerx = 0, centery = 0, n = contours[i].size();
	//����ȡ��С�����εı߽���ÿ���ܳ���������ȡһ��������꣬
	//������ȡ�ĸ����ƽ�����꣨��ΪС�����εĴ������ģ�
	centerx = (contours[i][n / 4].x + contours[i][n * 2 / 4].x + contours[i][3 * n / 4].x + contours[i][n - 1].x) / 4;
	centery = (contours[i][n / 4].y + contours[i][n * 2 / 4].y + contours[i][3 * n / 4].y + contours[i][n - 1].y) / 4;
	Point point1 = Point(centerx, centery);
	return point1;
}
int camera() {
	//namedWindow("ͼ��ɼ�����", cv::WINDOW_NORMAL);
	VideoCapture cap;
	// ��ȡ����ͷ
	cap.open(0);
	// �ж�����ͷ�Ƿ��
	if (!cap.isOpened()) {
		std::cerr << "Could't open capture" << std::endl;
		return -1;
	}
	Mat frame;
	// ���ռ����ϵ�����
	char keyCode;
	// �����ͼƬ����
	string imgName = "C:\\testFolder\\r.jpg";
	while (1) {

		// �Ѷ�ȡ������ͷ����Mat������
		cap >> frame;
		// �ж��Ƿ�ɹ�
		if (frame.empty()) {
			continue;
		}
		// ��ÿһ֡ͼƬ��ʾ����
		namedWindow("shipin", WINDOW_NORMAL);
		imshow("shipin", frame);

		// ��300�����ڵȴ��Ƿ���ڼ�������
		keyCode = waitKey(300);
		// ��ͼƬ��������
		imwrite(imgName, frame);
		image = frame;
		//imgName.at(0)++;
		frame.release();

		if (keyCode == 32) {
			break;
		}
	}
	return 1;

}


int input() {
	return 1;
}


//�˺������ڽ���ȡ����ͼƬ���ж�λ���
int output(Mat scr) {

	Mat src = scr;

	namedWindow("yuantu", WINDOW_NORMAL);// ��ʾ��������Ϊinput ��WINDOW_AUTOSIZE��ʾ��СΪͼƬ�Զ����С���Ҳ����Ը��Ĵ�С
	imshow("yuantu", src); //��ʾԭͼ

	Mat src_all = src.clone();
	Mat src_gray, drawing_gray;
	Mat img, img1;
	Mat threshold_output;
	//src = src + Scalar(100, 100, 100);
	//Ԥ����cvtColor��blur��equalizeHist��threshold
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	//��ͼ�����ƽ������  
	blur(src_gray, src_gray, Size(3, 3));//ģ����ȥ��ë��
	//ʹ�Ҷ�ͼ��ֱ��ͼ���⻯  
	equalizeHist(src_gray, src_gray);
	//ָ��112��ֵ���ж�ֵ��

	// ʹ�����õ�API
	//Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	//filter2D(src_gray, src_gray, src_gray.depth(), kernel);

	threshold(src_gray, threshold_output, 112, 255, THRESH_BINARY);
	namedWindow("threshold", WINDOW_NORMAL);
	imshow("threshold", threshold_output); //��ֵ�������

	//��Ҫ�ı�������
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

	//���ö�ֵ�����Ѱ������
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
	if (contours.size() == 0) {
		waitKey(4);//��ʾ�ĺ���ʱ�䣬�����������<=0��ʾһֱ��ʾ��>0��ʾ��ʾ��ʱ��
		destroyWindow("yuantu");
		destroyWindow("threshold");
		return -1;
	}
	else
	{
		int c = 0, ic = 0, k = 0, area = 0, x = 0;
		//ͨ����ɫ��λ����Ϊ�����������������������ص㣬ɸѡ��������λ��
		int parentIdx = -1, parentIdx_1 = -1;
		for (int i = 0; i < contours.size(); i++)//contours.size():��ʾ�����ĸ���
		{
			//������������ͼ
			drawContours(drawingAllContours, contours, parentIdx, CV_RGB(255, 255, 255), 1, 8);

			if (hierarchy[i][2] != -1 && ic == 0)//���IC=0˵�����Ǹ���
			{
				parentIdx = i;//��¼������������
				ic++;
			}
			else if (hierarchy[i][2] != -1)//��������
			{
				ic++;
			}
			else if (hierarchy[i][2] == -1)//�������������ò�����
			{
				ic = 0;
				parentIdx = -1;
			}

			//�ҵ���λ����Ϣ
			if (ic >= 2) //������Ƕ������ʱ
			{

				printf("����ڸ�%i����������", x);
				cout << parentIdx << endl;
				x++;
				//�����ҵ���������ɫ��λ��
				contours1.push_back(contours[parentIdx]);//��Ӹ�����(�����)��push_back����vector��������Ϊ��vectorβ������һ�����ݡ�

				/*�����������£�
				//Hierarchy:��νṹ
				//contours:����
					image---��������image�ϡ�
					contours---����ͼ��������ÿ����������һ�������ص�����ֵ��ɵ�������
					contourIdx---ָ����Щ������Ҫ�����ƣ�������ֵΪ�������ʾ���е���������Ҫ���ơ�
					color---������ɫ��
					thickness---�����߿��������Ϊ������������������ڲ���
					lineType---�����͡�
					hierarchy---��ѡ��������ʾ���������˽ṹ��
					maxLevel---��ʾҪ�������������㼶���ڲ���hierarchy��Ч������£��������Ϊ0��ʾֻ����ָ����������Ϊ1��ʾ�������е�����������Ƕ������Ϊ2��ʾ�������е�����������Ƕ��������Ƕ����֮��������������Դ�����.....
					offset����ѡ������ƫ�Ʋ�������ָ�����ƶ�����������е�������*/
					//����������ɫ��λ�ǵ�����
				drawContours(drawing, contours, parentIdx, CV_RGB(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 3, 8);
				ic = 0;
				parentIdx = -1;

			}
		}
		if (contours1.size() < 3) {//����ҵ�����������С��3 �ͷ����쳣
			waitKey(4);//��ʾ�ĺ���ʱ�䣬�����������<=0��ʾһֱ��ʾ��>0��ʾ��ʾ��ʱ��
			destroyWindow("yuantu");
			destroyWindow("threshold");
			return -1;
		}
		else {

			//������������ͼ
			namedWindow("DrawingAllContours");
			imshow("DrawingAllContours", drawingAllContours);
			//����������ɫ��λ�ǵ�����
			namedWindow("Drawing");
			imshow("Drawing", drawing);

			//���ķ�ʽ����������ɫ��λ�ǵ�����
			for (int i = 0; i < contours1.size(); i++) {
				drawContours(drawing2, contours1, i, CV_RGB(rng.uniform(100, 255), rng.uniform(100, 255), rng.uniform(100, 255)), -1, 4, hierarchy[k][2], 0, Point());
			}

			//���ķ�ʽ����������ɫ��λ�ǵ�����
			namedWindow("Drawing2");
			imshow("Drawing2", drawing2);

			Point point[10];
			//���ķ�ʽ����������ɫ��λ�ǵ�����
			for (int i = 0; i < contours1.size(); i++)
			{
				point[i] = Center_cal(contours1, i);
				printf("�����%i�������������", i);
				cout << point[i] << endl;
			}

			//������������������㶨λ�ǵ�������Ӷ�������߳�
			area = contourArea(contours1[1]);

			int area_side = cvRound(sqrt(double(area)));
			cout << "ͼ�����Ϊ" << area << endl;
			cout << "ͼ�����Ϊ" << double(area) << endl;
			cout << "ͼ�������ƽ����Ϊ" << sqrt(double(area)) << endl;
			cout << "ͼ�α߳�Ϊ" << cvRound(sqrt(double(area))) << endl;


			for (int i = 0; i < contours1.size(); i++)
			{
				cout << "����" << i % contours1.size() << "��������" << (i + 1) % contours1.size() << endl;
				cout << "����" << i << " �������" << contourArea(contours1[i]) << endl;
				cout << "  �ܳ�Լ��" << arcLength(contours1[i], true) << endl;
				cout << "  �߳�Լ��" << cvRound(arcLength(contours1[i], true) / 4) << endl;
				//����������λ�ǵ���������
				//line(drawing2, point[i % contours1.size()], point[(i + 1) % contours1.size()], color, area_side / 2, 8);

				line(drawing2, point[i % contours1.size()], point[(i + 1) % contours1.size()], color, area_side, 8);
			}

			namedWindow("Drawing2_1");
			imshow("Drawing2_1", drawing2);
			//������Ҫ�����������ά��
			Mat gray_all, threshold_output_all;


			cvtColor(drawing2, gray_all, COLOR_BGR2GRAY);
			namedWindow("drawing2_1_gray_all");
			imshow("drawing2_1_gray_all", gray_all);

			threshold(gray_all, threshold_output_all, 45, 255, THRESH_BINARY);
			namedWindow("drawing2_1_gray_all_threshold_output_all");
			imshow("drawing2_1_gray_all_threshold_output_all", threshold_output_all);

			findContours(threshold_output_all, contours2, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));//RETR_EXTERNAL��ʾֻѰ�����������


			cout << "��������������" << contours2.size() << endl;
			Point2f fourPoint2f[4];
			RotatedRect rectPoint = minAreaRect(contours2[0]);

			//imshow("222", threshold_output_all);
			/*RotatedRect minAreaRect(InputArray points)
			points��������Ϣ������Ϊ�����������(vector)����Mat��
			RotatedRect ������һ����������Ӿ��Σ���һ��RotatedRect���ࡣ
			����������Ϣ����Сб���Σ���һ��Box2D�ṹrect��
			����С��Ӿ��ε����ģ�x��y��������ȣ��߶ȣ�����ת�Ƕȣ���
			����Ҫ����������Σ�������Ҫ���ε�4����������box,
			ͨ������ cv2.cv.BoxPoints() ��ã�������ʽ[ [x0,y0], [x1,y1], [x2,y2], [x3,y3] ]��
			�õ�����С��Ӿ��ε�4������˳���������ꡢ��ȡ��߶ȡ���ת�Ƕȣ��Ƕ�����ʽ�����ǻ�������
			*/

			cout << "��С��Χ����'s angle:" << rectPoint.angle << endl;
			cout << "��С��Χ����'s center:" << rectPoint.center << endl;
			cout << "��С��Χ����'s width:" << rectPoint.size.width << endl;
			cout << "��С��Χ����'s height:" << rectPoint.size.height << endl;
			cout << "��С��Χ����'s area:" << rectPoint.size.area() << endl;
			cout << endl;

			// �Եõ����������һ��
			drawContours(drawing4, contours2, -1, Scalar(255, 255, 255), FILLED);
			namedWindow("drawing4");
			imshow("drawing4", drawing4);

			//��rectPoint�����д洢������ֵ�ŵ� fourPoint��������
			rectPoint.points(fourPoint2f);

			for (int i = 0; i < 4; i++)
			{

				cout << fourPoint2f[i] << endl;
				cout << "�˵�" << fourPoint2f[i % 4] << "����" << fourPoint2f[(i + 1) % 4] << endl;

				line(drawing4, fourPoint2f[i % 4], fourPoint2f[(i + 1) % 4], Scalar(0, 0, 255), 3);
			}
			namedWindow("drawing4_1");
			imshow("drawing4_1", drawing4);

			cvtColor(drawing4, drawing5, COLOR_BGR2GRAY);
			findContours(drawing5, contours3, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
			drawContours(drawing5, contours3, -1, Scalar(255), FILLED);
			namedWindow("drawing5");
			imshow("drawing5", drawing5);
			//��ͼ��drawing6

			drawing6.setTo(255);

			src.copyTo(drawing6, drawing5);
			namedWindow("drawing6", 1);
			imshow("drawing6", drawing6);
			imwrite("C:\\testFolder\\r1.jpg", drawing6);
			Rect rect = boundingRect(Mat(contours3[0]));
			cout << "��1����С��Χ���� area:" << rect.area() << endl;
			//����С��Χ����
			RotatedRect rectPoint_1 = minAreaRect(contours3[0]);//minAreaRect()�������㲢����ָ���㼯����С����߽�б���Ρ�
			cout << "��С��Χ����'s area:" << rectPoint_1.size.area() << endl;

			float angle = rectPoint.angle;
			Point2f center = rectPoint.center;  //���ĵ�
			if (angle != 90 && angle != -90)
			{
				angle = angle - 90;//������
				Mat M2 = getRotationMatrix2D(center, angle, 1);//������ת�����ŵı任���� 
				warpAffine(drawing6, drawing7, M2, src_all.size(), 1, 0, Scalar(0));//����任 
				namedWindow("drawing7", WINDOW_NORMAL);
				imshow("drawing7", drawing7);
				img = drawing7(rect);
				namedWindow("img", WINDOW_NORMAL);
				imshow("img", img);
				imwrite("C:\\testFolder\\r1.jpg", img);
			}
			else {
				namedWindow("drawing7", WINDOW_NORMAL);
				imshow("drawing7", drawing6);
				img = drawing6(rect);
				namedWindow("img", WINDOW_NORMAL);
				imshow("img", img);
				imwrite("C:\\testFolder\\r1.jpg", img);
			}

			image4 = img;
			waitKey(4);//��ʾ�ĺ���ʱ�䣬�����������<=0��ʾһֱ��ʾ��>0��ʾ��ʾ��ʱ��
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
	int x = MessageBox(GetForegroundWindow(), "�����ʶ���Ƿ������", "OpenCVTest", 3);
	printf("%d\n", x);
	return x;
}



int main(int argc, char** argv) {

	rd = "rd/s/q " + folderPath;
	system(rd.c_str());
	command = "mkdir -p " + folderPath;
	system(command.c_str());


	while (true) {
		//��������ͷ����ͼ��
		camera();
		Mat src = image;
		//Mat src = imread("C:\\testFolder\\r.jpg");
		//imshow("mmm", image);
		if (output(src) == 1) {
			//���õ���imagge���в���
			cout << "001" << endl;
			//waitKey(2000);//��ʾ�ĺ���ʱ�䣬�����������<=0��ʾһֱ��ʾ��>0��ʾ��ʾ��ʱ��
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
			waitKey();//��ʾ�ĺ���ʱ�䣬�����������<=0��ʾһֱ��ʾ��>0��ʾ��ʾ��ʱ��
			destroyWindow("mmm");

			/*���һֱû��ʶ�𵽾��޷����أ�����һ�����⡣
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