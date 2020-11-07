#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>   
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>              
#include "math.h"
#include <stdlib.h>

using namespace cv;
using namespace std;


//得到A4纸区域
void getRoi(Mat &src,Point2f vertices[4])
{
    Mat frame = src.clone();
	Mat temp = src.clone();
    cvtColor(frame,frame,COLOR_RGB2GRAY);
    threshold(frame,frame, 100, 255, THRESH_BINARY);
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(frame,frame,MORPH_OPEN,element);  //进行开运算(去除小的噪点)

    //提取轮廓
    vector<vector<Point>> contours;
    findContours(frame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
   //定义外接拟合矩形
   RotatedRect rect;

	for (int i = 0; i < contours.size(); ++i) 
	{
		rect = minAreaRect(Mat(contours[i]));  //最小外接矩形 ，但应该为梯形才对
		rect.points(vertices);                               //外接矩形的四个顶点  右下，左下，左上，右上

		for (int l = 0; l < 4; l++)
		{
			line(temp, vertices[l], vertices[(l + 1) % 4], Scalar(0, 255, 255), 1);   
            circle(temp,vertices[l],1,(0,0,255),3);
		}
	}
    imshow("ROI",temp);

}

//对比度与亮度的改变
void  change_intensity (Mat input, Mat dst)
{

	int height = input.rows;
	int width = input.cols;
	//调整对比度为1.3
	float alpha = 1.3;
	//调整亮度30
	float beta = 30;

	for(int row = 0;row < height; row++)
	{
		for(int col = 0;col < width;col++)
		{
			if(input.channels() == 3)//判断是否为3通道图片
			{
				//将遍历得到的原图像素值，返回给变量b,g,r
				float b = input.at<Vec3b>(row,col)[0];//blue
				float g = input.at<Vec3b>(row,col)[1];//green
				float r= input.at<Vec3b>(row,col)[2];//red
				//开始操作像素，对变量b,g,r做改变后再返回到新的图片。
				dst.at<Vec3b>(row,col)[0] = saturate_cast<uchar>(b*alpha + beta);
				dst.at<Vec3b>(row,col)[1] = saturate_cast<uchar>(g*alpha + beta);
				dst.at<Vec3b>(row,col)[2] = saturate_cast<uchar>(r*alpha + beta);
			}
			else if(input.channels() == 1)//判断是否为单通道的图片
			{
				float v = input.at<uchar>(row,col);
				dst.at<uchar>(row,col) = saturate_cast<uchar>(v*alpha+beta);
			}

		}
	}
}

Mat  reverse (Mat input)
{
	//阈值
	int threshold = 225;
	int height = input.rows;
	int width = input.cols;
	cvtColor(input,input,COLOR_RGB2GRAY);

	for(int row = 0;row < height; row++)
	{
		for(int col = 0;col < width;col++)
		{
			if (input.at<uchar>(row,col) < threshold)
			{
				input.at<uchar>(row,col) = 255;
			}
			else
			{
				input.at<uchar>(row,col) = 0;
			} 
		}
	}
	return input;

}


int main()
{
    //在网页上打开程序，调用手机摄像头

    //显示热区，当A4纸4个角点在热区时，连续检测几帧，坐标变化不大时，截取A4纸图像

    //对A4纸透视变换
     
   Mat src =  imread("/home/shouyiyang/桌面/视觉实习生任务/额外1/build/1.jpg",1);
   if(!src.data) cout << "读取图像失败！！" << endl;
   int width =  src.cols;
   int height = src.rows;

    //1. 找到A4纸边缘，框出ROI区域    A4纸297mm*210mm
    Point2f points[4];
    getRoi(src, points);
    for (int l = 0; l < 4; l++)
	{
        cout <<points[l] <<endl;   //***** 右下，左下，左上，右上*****
	}


    //2. 透视变换，得到仅有A4纸的图像
    //变换后的坐标 210*3 = 630 , 297*3 = 891
	int ratio = 3;
    Point2f trans_points[4] ;
    trans_points[0] = Point2f(210.0*ratio,297.0*ratio); 
    trans_points[1]= Point2f(0.0,297.0*ratio);
    trans_points[2]= Point2f(0.0,0.0); 
    trans_points[3]= Point2f(210.0*ratio,0.0); 
    Mat warp_dst = Mat::zeros(1000,900,src.type());
    Mat matrix(2,3,CV_32FC1);

    //求得旋转矩阵
    matrix =  getAffineTransform(points,trans_points);
    //旋转变换图像
    warpAffine( src, warp_dst,matrix,warp_dst.size());//！！！！！！！！好像应该把ROI的图像进行变化，而不是src
    //imshow("warp_dst",warp_dst);

	//****************************不是完全得到ROI区域！！！！！！！！

	//3. 对比度与亮度的改变
	Mat intensity = Mat::zeros(warp_dst.size(),warp_dst.type());  //创建一个与原图一样大小的空白图片
	change_intensity (warp_dst, intensity);
	//namedWindow(" intensity",0);
	//imshow(" intensity", intensity);
	//如何不受手机白平衡的影响？

	//4. 得到圆的黑白二值图像
	
	Mat circle = intensity.clone();
	Mat circle_rev = reverse(circle);

    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(circle_rev,circle_rev,MORPH_OPEN,element);  //进行开运算(去除小的噪点)
	//imshow("morphology",circle_rev);

	//5. 得到圆的中点坐标和半径
	Mat circle_dst = circle_rev.clone();
	GaussianBlur(circle_dst,circle_dst,Size(5,5),2,2);
	Canny(circle_dst,circle_dst,10,250,5);
	//imshow("Canny",circle_dst);

	vector<Vec3f> circles;
	//霍夫圆检测：（原图，存储圆的vec3d，分辨率，两圆心最小距离，para1，para2，圆半径最小值，圆半径最大值）
	HoughCircles(circle_dst , circles, CV_HOUGH_GRADIENT, 2, 10, 100, 35 ,0, 100 );

	if(circles.size() == 0) 
	{
		cout<< "error! no circle detected!"<<endl;
	}
	else
	{
		for (int i = 0; i < circles.size(); i++) 
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]) );
			int radius = cvRound(circles[i][2]);
			cout << i+1 <<"   Center:" << center <<"    Radius:"<< radius <<endl; 
			//circle(circle_dst , center, radius,Scalar(255,255,0), 2);

			//单位为毫米
			double real_x = center.x/ratio;
			double real_y = center.y/ratio;
			double real_radius = radius/ratio;

			char  str[100];
			sprintf(str, "center: [%f , %f ]  radius: %d",real_x,real_y, real_radius);
			//(图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细)
			putText(warp_dst , str , center , 4 , 0.5 , (255,255,255),1 );
			
		}
	}
	
	//imshow("框出区域", circle_dst);
	imshow("text",warp_dst);

  if(waitKey(0) == 27)
  {
      return -1;
  }
}