#include <opencv2/opencv.hpp>
#include <iostream>
#include "picojson.h"

using namespace cv;
using namespace std;

Mat src; Mat src_unwarped; Mat src_gray; Mat src_thresh; Mat src_blur; Mat src_dilate; Mat src_erode; Mat canny_output; Mat drawing, erosion_dst;


int main(int argc, char** argv)
{
	//Open default camera
	VideoCapture cap(2);

	if (cap.isOpened() == false)
	{
		cout << "Cannot open camera" << endl;
		cin.get();
		return -1;
	}

	cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);

	cout << cap.get(CAP_PROP_FOURCC) << endl;
	double dWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
	cout << "Resolution is: " << dWidth << " x " << dHeight << endl;

	Mat K = (Mat_<double>(3,3) << 238.82483251892208, 0.0, 318.16929539273366, 0.0, 239.06534916420654, 242.86520890990545, 0.0, 0.0, 1.0);
	Mat D = (Mat_<double>(1, 4) << -0.04042847278703006, 0.002813389381558989, -0.006067430365909259, 0.0012053547649747928);
	cout << "K = " << K << endl;
	cout << "D = " << D << endl;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point3f>> contours1;
	Mat newCamMatForUndistort;
	Mat map1, map2;

	bool bSuccess = cap.read(src); // read a new frame from video, breaking the while loop if the frames cannot be captured
	if (bSuccess == false)
	{
		cout << "Video camera is disconnected" << endl;
		cin.get(); //Wait for any key press
		return 0;
	}

	cout << "Camera open!" << endl;
	Size image_size = src.size();
	//cv::Size image_size_big = src.size() * 2;

	//gives us a new camera Mat that works for the function: "fisheye::initUndistortRectifyMap"
	fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, Matx33d::eye(), newCamMatForUndistort, 1, image_size);

	//gives us outputarrays containing data needed for unwarping
	fisheye::initUndistortRectifyMap(K, D, Matx33d::eye(), newCamMatForUndistort, image_size, CV_16SC2, map1, map2);

	//remaps the Mat accoring to unwarping data from "fisheye::initUndistortRectifyMap"
	remap(src, src_unwarped, map1, map2, cv::INTER_LINEAR);

	//turns source from BGR to grayscale
	cvtColor(src_unwarped, src_gray, COLOR_BGR2GRAY);
	//imshow("gray", src_gray);

	//blurs the image
	blur(src_gray, src_blur, Size(3, 3));
	//imshow("blur", src_blur);

	//thersholds grayscaled image
	cv::threshold(src_blur, src_thresh, 230, 255, THRESH_BINARY);
	//imshow("thresh", src_thresh);

	/// Detect edges using canny
	Canny(src_thresh, canny_output, 240, 255, 3);

	/// Find contours
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));


	for (int i = 0; i < contours.size(); i++) {
		for (int j = 0; j < contours[i].size(); j++) {
			contours1[i][j].x = (float)contours[i][j].x;
			contours1[i][j].y = (float)contours[i][j].y;
			contours1[i][j].z = 1.0;
		}
	}

	Mat rvec; Mat tvec; Mat inliers;
	
	while (true)
	{
		bSuccess = cap.read(src); // read a new frame from video 
		//Breaking the while loop if the frames cannot be captured
		if (bSuccess == false)
		{
			cout << "Video camera is disconnected" << endl;
			cin.get(); //Wait for any key press
			break;
		}
		imshow("before", src);
		
		//remaps the Mat accoring to unwarping data from "fisheye::initUndistortRectifyMap"
		remap(src, src_unwarped, map1, map2, cv::INTER_LINEAR);

		//turns source from BGR to grayscale
		cvtColor(src_unwarped, src_gray, COLOR_BGR2GRAY);
		imshow("gray", src_gray);

		//blurs the image; try this more
		blur(src_gray, src_blur, Size(3, 3));
		//imshow("blur", src_blur);

		//thersholds grayscaled image
		cv::threshold(src_blur, src_thresh, 230, 255, THRESH_BINARY);
		imshow("thresh", src_thresh);


		/// Detect edges using canny
		Canny(src_thresh, canny_output, 240, 255, 3);


		/// Find contours
		findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

		//solvePnPRansac(contours1, contours, K, D, rvec, tvec, 0, 100);
			
		//Draw contours
		drawing = Mat::zeros(src_thresh.size(), CV_8UC3);

		for (int i = 0; i < contours.size(); i++)
		{
			
			//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			Scalar color = Scalar(255, 255, 255);
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		}
		
		//wait for 1 ms until any key is pressed.  
		//If the 'Esc' key is pressed, break the while loop.
		//If the any other key is pressed, continue the loop 
		//If any key is not pressed withing 10 ms, continue the loop 
		int keyValue = waitKey(1);
		if (keyValue == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}
	}
	return 0;
}