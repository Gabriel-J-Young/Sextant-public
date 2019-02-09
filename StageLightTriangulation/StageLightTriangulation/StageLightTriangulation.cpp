// Vector Vision.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

Mat src; Mat src_unwarped; Mat src_gray; Mat src_thresh; Mat src_blur; Mat src_dilate; Mat src_erode; Mat canny_output; Mat drawing, erosion_dst;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

void thresh_callback(int, void*);

int main(int argc, char** argv)
{
	//Open default camera
	VideoCapture cap(4);

	if (cap.isOpened() == false)
	{
		cout << "Cannot open camera" << endl;
		cin.get();
		return -1;
	}
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	double dWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

	cout << "Resolution is: " << dWidth << " x " << dHeight << endl;
	string rv_name = "Raw Video";
	string vv_name = "Vector View";
	namedWindow(rv_name);
	//createTrackbar(" Canny thresh:", "Raw Video", &thresh, max_thresh);
	namedWindow(vv_name);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Mat K = (Mat_<double>(3,3) << 238.82483251892208, 0.0, 318.16929539273366, 0.0, 239.06534916420654, 242.86520890990545, 0.0, 0.0, 1.0);
	Mat D = (Mat_<double>(1, 4) << -0.04042847278703006, 0.002813389381558989, -0.006067430365909259, 0.0012053547649747928);
	//Mat R = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	cout << K;
	cout << D;

	while (true)
	{
		bool bSuccess = cap.read(src); // read a new frame from video 
		//Breaking the while loop if the frames cannot be captured
		if (bSuccess == false)
		{
			cout << "Video camera is disconnected" << endl;
			cin.get(); //Wait for any key press
			break;
		}
		imshow("before", src);
		cv::Mat map1, map2;
		cv::Size image_size;
		image_size = src.size();
		//fisheye::undistortImage(src, src, K, D);

		Mat newCamMatForUndistort;
		
		cv::Size image_size_big;
		image_size_big = src.size()*2;

		//OptCam = getOptimalNewCameraMatrix(K, D, image_size, 1, image_size, 0);

		//gives us a new camera Mat that works for the fucntion: "fisheye::initUndistortRectifyMap"
		fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, Matx33d::eye(), newCamMatForUndistort, 1, image_size_big);

		//gives us outputarrays containing data needed for unwarping
		fisheye::initUndistortRectifyMap(K, D, Matx33d::eye(), newCamMatForUndistort, image_size_big, CV_16SC2, map1, map2);

		//remaps the Mat accoring to unwarping data from "fisheye::initUndistortRectifyMap"
		remap(src, src_unwarped, map1, map2, cv::INTER_LINEAR);

		//turns source from BGR to grayscale
		cvtColor(src_unwarped, src_gray, CV_BGR2GRAY);
		imshow("gray", src_gray);

		//blurs the image; try this more
		blur(src_gray, src_blur, Size(3, 3));
		imshow("blur", src_blur);

		//thersholds grayscaled image
		cv::threshold(src_blur, src_thresh, 230, 255, THRESH_BINARY);
		imshow("thresh", src_thresh);

		

		/*
		//Dilates the image- makes white parts bigger

		int dilation_size = 5;
		Mat element1 = getStructuringElement(0,
			Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			Point(dilation_size, dilation_size));
		dilate(src_gray, src_dilate, element1);
		imshow("dilate", src_dilate);

		//erodes the images- makes white parts smaller 
		
		int erosion_size = 15;
		Mat element = getStructuringElement(0,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));
		erode(src_gray, src_erode, element);
		imshow("erode", src_erode);
		*/
		/*try to replace this contours stuff with find center and draw points*/

		/// Detect edges using canny
		Canny(src_thresh, canny_output, 240, 255, 3);


		/// Find contours
		findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		
		solvePnPRansac();

		/*
		//get moments 
		vector<Moments> mu(contours.size());
		for (int i = 0; i < contours.size(); i++) {
			mu[i] = moments(contours[i], false);
		}

		//get centroids
		vector<Point2f> mc(contours.size());
		for (int i = 0; i < contours.size(); i++) {
			mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}

		//draw contours 
		Mat drawing(canny_output.size(), CV_8UC3, Scalar(0, 0, 0));
		for (int i = 0; i < contours.size(); i++) {
			Scalar color = Scalar(167, 151, 0);
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
			circle(drawing, mc[i], 4, color, -1, 8, 0);
		}

		namedWindow("contours", WINDOW_AUTOSIZE);
		imshow("contours", drawing);
		*/

			
		//Draw contours
		drawing = Mat::zeros(src_thresh.size(), CV_8UC3);

		for (int i = 0; i < contours.size(); i++)
		{
			
			//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			Scalar color = Scalar(255, 255, 255);
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		}
		


		//show the frame in the created window
		//imshow(rv_name, src);
		imshow(vv_name, drawing);



		//wait for for 10 ms until any key is pressed.  
		//If the 'Esc' key is pressed, break the while loop.
		//If the any other key is pressed, continue the loop 
		//If any key is not pressed withing 10 ms, continue the loop 
		if (waitKey(10) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}
	}
	return 0;
}