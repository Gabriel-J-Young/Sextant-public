#include <opencv2/opencv.hpp>
#include <iostream>
#include "picojson.h"

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
	VideoCapture cap(0);

	if (cap.isOpened() == false)
	{
		cout << "Cannot open camera" << endl;
		cin.get();
		return -1;
	}

	cap.set(CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CAP_PROP_FRAME_HEIGHT, 480);
	double dWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

	cout << "Resolution is: " << dWidth << " x " << dHeight << endl;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Mat K = (Mat_<double>(3, 3) << 238.82483251892208, 0.0, 318.16929539273366, 0.0, 239.06534916420654, 242.86520890990545, 0.0, 0.0, 1.0);
	Mat D = (Mat_<double>(1, 4) << -0.04042847278703006, 0.002813389381558989, -0.006067430365909259, 0.0012053547649747928);
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

		Mat newCamMatForUndistort;

		cv::Size image_size_big;
		image_size_big = src.size() * 2;

		//gives us a new camera Mat that works for the fucntion: "fisheye::initUndistortRectifyMap"
		fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, Matx33d::eye(), newCamMatForUndistort, 1, image_size_big);

		//gives us outputarrays containing data needed for unwarping
		fisheye::initUndistortRectifyMap(K, D, Matx33d::eye(), newCamMatForUndistort, image_size_big, CV_16SC2, map1, map2);

		//remaps the Mat accoring to unwarping data from "fisheye::initUndistortRectifyMap"
		remap(src, src_unwarped, map1, map2, cv::INTER_LINEAR);
		imshow("after", src_unwarped);

		//turns source from BGR to grayscale
		cvtColor(src_unwarped, src_gray, COLOR_BGR2GRAY);
		imshow("gray", src_gray);

		//blurs the image; try this more
		blur(src_gray, src_blur, Size(3, 3));
		imshow("blur", src_blur);

		//thersholds grayscaled image
		cv::threshold(src_blur, src_thresh, 230, 255, THRESH_BINARY);
		imshow("thresh", src_thresh);
		
		/// Detect edges using canny
		Canny(src_thresh, canny_output, 240, 255, 3);

		/// Find contours
		findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

		//Draw contours
		drawing = Mat::zeros(src_thresh.size(), CV_8UC3);

		for (int i = 0; i < contours.size(); i++)
		{

			//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			Scalar color = Scalar(255, 255, 255);
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		}
		imshow("drawing", drawing);


		if (waitKey(10) == 112) {
			picojson::value v;
			std::cout << v;
		}

		//wait for for 10 ms until any key is pressed.  
		//If the 'Esc' key is pressed, break the while loop.
		//If the any other key is pressed, continue the loop 
		//If any key is not pressed withing 10 ms, continue the loop 
		if (waitKey(10) == 27)
		{
			cout << "Esc key is pressed by user. Stopping the video" << endl;
			break;
		}
	}
	return 0;
}