#include <opencv2/opencv.hpp>
#include <iostream>
#include "picojson.h"

using namespace cv;
using namespace std;

vector<vector<Point2d>> getHulls(Mat& src)
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
	/*cin.get();*/
	/// Detect edges using canny
	Canny(src_working, canny_output, 250, 255, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	cout << "houston, this is yeet one and a half" << endl;
	vector<vector<Point2d>> hulls;
	
	vector<Point> hull;
	cout << "size of contours: " << contours.size() << endl;
	for (int i = 0; i < contours.size(); i++)
	{
		convexHull(contours[i], hull);
		//if (hull.size() >= 4) {
			vector<Point2d> hullout;
			for (int j = 0; j < hull.size(); j++)
			{
				hullout.push_back(Point2d((double)hull[j].x, (double)hull[j].y));
			}
			hulls.push_back(hullout);
		//}
	}
	//Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	Scalar color = Scalar(167, 151, 0); // B G R values
	//	drawContours(drawing, hull, (int)i, color);
	//}
	//imshow("Hull demo", drawing);
	//cout << "size of hullllllllllllllllls: " << hulls.size() << endl;
	return hulls;
}

//vector<Point2f> getCentriods(Mat& src)
//{
//
//	Mat src_gray, canny_output, src_working;
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	//turns source from BGR to grayscale
//	cvtColor(src, src_working, COLOR_BGR2GRAY);
//	imshow("gray", src_working);
//
//	//blurs the image
//	blur(src_working, src_working, Size(5, 5));
//	imshow("blur", src_working);
//
//	//thersholds grayscaled image
//	threshold(src_working, src_working, 240, 255, THRESH_BINARY);
//	imshow("thresh", src_working);
//
//	//erodes the images- makes white parts smaller 
//	int erosion_size = 10;
//	Mat element = getStructuringElement(0,
//		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
//		Point(erosion_size, erosion_size));
//	erode(src_working, src_working, element);
//	imshow("erode", src_working);
//
//	//Dilates the image- makes white parts bigger
//	int dilation_size = 10;
//	Mat element1 = getStructuringElement(0,
//		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
//		Point(dilation_size, dilation_size));
//	dilate(src_working, src_working, element1);
//	imshow("dilate", src_working);
//	waitKey(1);
//	/// Detect edges using canny
//	Canny(src_working, canny_output, 250, 255, 3);
//	/// Find contours
//	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//	//get moments
//	vector<Moments> mu(contours.size());
//	for (int i = 0; i < contours.size(); i++) 
//	{
//		mu[i] = moments(contours[i], false);
//	}
//
//	//get centroids of figures 
//	vector<Point2f> mc(contours.size());
//	for (int i = 0; i < contours.size(); i++)
//	{
//		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
//	}
//
//
//	Mat drawing(canny_output.size(), CV_8UC3, Scalar(255, 255, 255));
//	for (int i = 0; i < contours.size(); i++)
//	{
//		Scalar color = Scalar(167, 151, 0); // B G R values
//		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
//		circle(drawing, mc[i], 4, color, -1, 8, 0);
//	}
//	imshow("Contours", drawing);
//	return mc;
//}

int main(int argc, char** argv)
{
	//declaration of Mats, vectors, and vectors of vectors
		//camera frame accessors
		Mat src; Mat src_unwarped; Mat drawing;
		//output Mats for calibration 
		Mat newCamMatForUndistort;
		Mat map1, map2;
		//vectors and  vectors of vectors of important points found by filtering
		vector<vector<Point2d>> hulls;
		vector<vector<Point3d>> contours3D;
		vector<Point2f> centriods;

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

	//getHulls returns the corners of external contours as a vector of vector of 2d Points
	hulls = getHulls(src_unwarped);

	cout << "houston, this is yeet two " << endl;

	//assigns the important points found by getHulls from the first frame to contours3D
	for (int i = 0; i < hulls.size(); i++) 
	{
		vector<Point3d> newvec;
		for (int j = 0; j < hulls[i].size(); j++) 
		{
			Point3d point;
			point.x = (double)hulls[i][j].x;
			point.y = (double)hulls[i][j].y;
			point.z = 1.0;
			newvec.push_back(point);
		}
		contours3D.push_back(newvec);
	}

	//prints out contours3D
	//for (int i = 0; i < contours3D.size(); i++)
	//{
	//	for (int j = 0; j < contours3D[i].size(); j++)
	//	{
	//		cout << "contours3D: " << i << ", " << j << endl;
	//		cout << contours3D[i][j] << endl;
	//	}

	//}

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
		hulls = getHulls(src_unwarped);

		Mat rvec; Mat tvec; Mat inliers;
		//Mat incontour = contours3D[0];
		//centriods = getCentriods(src_unwarped);
		
		/*for (int i = 0; i <= hulls.size(); i++) {
			cout << hulls[i];
		}*/

		vector<Point2d> image_points;
		image_points.push_back(cv::Point2d(359, 391));    // Nose tip
		image_points.push_back(cv::Point2d(399, 561));    // Chin
		image_points.push_back(cv::Point2d(337, 297));     // Left eye left corner
		image_points.push_back(cv::Point2d(513, 301));    // Right eye right corner
		image_points.push_back(cv::Point2d(345, 465));    // Left Mouth corner
		image_points.push_back(cv::Point2d(453, 469));    // Right mouth corner

		// 3D model points.
		vector<Point3d> model_points;
		model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
		model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
		model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
		model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
		model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
		model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));

		int npoints = std::max(contours3D[0].checkVector(3, CV_32F), contours3D.checkVector(3, CV_64F));
		cout << "npoints: " << npoints << endl;

		//solvePnPRansac(model_points, image_points, K, D, rvec, tvec);
		cout << "contours3D[0]: " << contours3D[0] << endl;
		cout << "hulls[0]: " << hulls[0] << endl;
		solvePnPRansac(contours3D[0], hulls[0], K, D, rvec, tvec);// , false, 100, 8.0, 100, noArray(), CV_EPNP);
		cout << "rvec: " << rvec << endl;
		cout << "tvec: "<< tvec << endl;
		//Draw contours
		//drawing = Mat::zeros(src_unwarped.size(), CV_8UC3);

		//for (int i = 0; i < hulls.size(); i++)
		//{
		//	cout << "Attempting to draw hull " << i << endl;
		//	//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//	Scalar color = Scalar(255, 0, 0);
		//	drawContours(drawing, hulls, (int)i, color);//, 2, 8, vector<Vec4i>(), 0, Point());
		//}
		//imshow("drawing", drawing);
		
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