#include <opencv2/opencv.hpp>
#include <iostream>
#include "picojson.h"

using namespace cv;
using namespace std;

vector<Point2d> getCentriods(Mat& src)
{

	Mat src_gray, canny_output, src_working;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//turns source from BGR to grayscale
	cvtColor(src, src_working, COLOR_BGR2GRAY);
	imshow("gray", src_working);

	//blurs the image
	blur(src_working, src_working, Size(5, 5));
	imshow("blur", src_working);

	//thersholds grayscaled image
	threshold(src_working, src_working, 240, 255, THRESH_BINARY);
	imshow("thresh", src_working);

	//erodes the images- makes white parts smaller 
	int erosion_size = 10;
	Mat element = getStructuringElement(0,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	erode(src_working, src_working, element);
	imshow("erode", src_working);

	//Dilates the image- makes white parts bigger
	int dilation_size = 10;
	Mat element1 = getStructuringElement(0,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	dilate(src_working, src_working, element1);
	imshow("dilate", src_working);
	waitKey(1);
	/// Detect edges using canny
	Canny(src_working, canny_output, 250, 255, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//get moments
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++) 
	{
		mu[i] = moments(contours[i], false);
	}

	//get centroids of figures 
	vector<Point2d> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}


	Mat drawing(canny_output.size(), CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(255, 255, 0); // B G R values
		//drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		circle(drawing, mc[i], 4, color, -1, 8, 0);
	}
	imshow("Centroids", drawing);
	return mc;
}

int main(int argc, char** argv)
{
	int i = 0;
	//declaration of Mats, vectors, and vectors of vectors
		//camera frame accessors
		Mat src; Mat src_unwarped; Mat drawing;
		//output Mats for calibration 
		Mat newCamMatForUndistort;
		Mat map1, map2;
		//vectors and  vectors of vectors of important points found by filtering
		vector<Point2d> centroids;
		vector<Point3d> centroids3D;

	//Open default camera
	VideoCapture cap(4);

	if (cap.isOpened() == false)
	{
		cout << "Cannot open camera" << endl;
		cin.get();
		return -1;
	}

	//I have no idea what this does 
	cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
	cout << cap.get(CAP_PROP_FOURCC) << endl;

	//manually sets camera dimensions
	cap.set(CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
	
	//finds & prints camera dimensions
	double dWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
	cout << "Resolution is: " << dWidth << " x " << dHeight << endl;

	//hardcoded calibration data
	Mat K = (Mat_<double>(3, 3) << 540.6884489226692, 0.0, 951.3635524878698, 0.0, 540.4187901470385, 546.9124878500451, 0.0, 0.0, 1.0);
	Mat D = (Mat_<double>(1, 4) << -0.04517325603821452, 0.001435732351585509, -0.004105241869408653, 0.0009228132505096691);
	//Mat K = (Mat_<double>(3,3) << 238.82483251892208, 0.0, 318.16929539273366, 0.0, 239.06534916420654, 242.86520890990545, 0.0, 0.0, 1.0);
	//Mat D = (Mat_<double>(1, 4) << -0.04042847278703006, 0.002813389381558989, -0.006067430365909259, 0.0012053547649747928);
	cout << "K = " << K << endl;
	cout << "D = " << D << endl;

	// read a new frame from video, breaking the while loop if the frames cannot be captured
	bool bSuccess = cap.read(src); 
	if (bSuccess == false)
	{
		cout << "Video camera is disconnected" << endl;
		cin.get(); //Wait for any key press
		return 0;
	}
	cout << "Camera open!" << endl;

	Size image_size = src.size();
	//gives us a new camera Mat that works for the function: "fisheye::initUndistortRectifyMap"
	fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, Matx33d::eye(), newCamMatForUndistort, 1, image_size);

	//gives us outputarrays containing data needed for unwarping
	fisheye::initUndistortRectifyMap(K, D, Matx33d::eye(), newCamMatForUndistort, image_size, CV_16SC2, map1, map2);

	//remaps the Mat accoring to unwarping data from "fisheye::initUndistortRectifyMap"
	remap(src, src_unwarped, map1, map2, cv::INTER_LINEAR);
	cout << "houston, this is yeet one " << endl;
	string str;
	imshow("KJDG", src_unwarped);
	waitKey(1);

	//getCentriods returns the centroids of found shapes
	centroids = getCentriods(src_unwarped);

	//assigns the important points found by getCentroids from the first frame to centroids3D
	for (int i = 0; i < centroids.size(); i++) 
	{
		Point3d point;
		point.x = (float)centroids[i].x;
		point.y = (float)centroids[i].y;
		point.z = 0;
		centroids3D.push_back(point);
		cout << "loop: " << i << "___" << centroids3D << endl;
	}

	while (true)
	{
		// read a new frame from video breaking the while loop if the frames cannot be captured
		bSuccess = cap.read(src);
		if (bSuccess == false)
		{
			cout << "Video camera is disconnected" << endl;
			//Wait for any key press
			cin.get();
			break;
		}
		imshow("before", src);

		//remaps the Mat accoring to unwarping data from "fisheye::initUndistortRectifyMap"
		remap(src, src_unwarped, map1, map2, cv::INTER_LINEAR);

		//getHulls returns the corners of external contours as a vector of vector of 2d Points
		//hulls = getHulls(src_unwarped);

		//getCentroids returns the centroids of found shapes
		centroids = getCentriods(src_unwarped);

		Mat rvec; Mat tvec; Mat inliers;
		vector<Point3d> centroidPoint3D;
		vector<Point2d> centroidPoint2D;
		centroidPoint3D.push_back(Point3d(0, 0, 1000.0));

		//cout << "centroids3D size: " << centroids3D.size() << "centroids size: " << centroids.size() << endl;
		if (centroids3D.size() == centroids.size())
		{
			Mat temptvec = tvec;
			solvePnPRansac(centroids3D, centroids, K, D, rvec, tvec, false);
			cout << "You called solvePnPRansac!" << "rvec is: " << rvec << "tvec is: " << tvec << endl;

		projectPoints(centroidPoint3D, rvec, tvec, K, D, centroidPoint2D);

		for (int i = 0; i < centroids3D.size(); i++)
		{
			circle(src_unwarped, centroids[i], 3, Scalar(0, 0, 255), -1);
		}

		line(src_unwarped, centroids[0], centroidPoint2D[0], Scalar(255, 0, 0), 2);

		}
		imshow("THE LINE", src_unwarped);
	
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