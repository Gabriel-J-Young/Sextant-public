#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src; Mat src_unwarped; Mat newCamMatForUndistort; Mat map1; Mat map2;

	VideoCapture cap(4);

	if (cap.isOpened() == false) {
		cout << "Cannot open camera" << endl;
		return -1;
	}

	cap.set(CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(CAP_PROP_FRAME_HEIGHT, 1080); 
	double dWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
	cout << "Resolution is: " << dWidth << " x " << dHeight << endl;

	//hardcoded calibration data
	Mat K = (Mat_<double>(3, 3) << 540.6884489226692, 0.0, 951.3635524878698, 0.0, 540.4187901470385, 546.9124878500451, 0.0, 0.0, 1.0);
	Mat D = (Mat_<double>(1, 4) << -0.04517325603821452, 0.001435732351585509, -0.004105241869408653, 0.0009228132505096691);
	cout << "K = " << K << endl;
	cout << "D = " << D << endl;

	// read a new frame from video, breaking the while loop if the frames cannot be captured
	bool bSuccess = cap.read(src);
	if (bSuccess == false)
	{
		cout << "Video camera is disconnected" << endl;
		//Wait for any key press
		cin.get(); 
		return 0;
	}
	cout << "Camera open!" << endl;

	Size image_size = src.size();
	//gives us a new camera Mat that works for the function: "fisheye::initUndistortRectifyMap"
	fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, Matx33d::eye(), newCamMatForUndistort, 1, image_size);

	//gives us outputarrays containing data needed for unwarping
	fisheye::initUndistortRectifyMap(K, D, Matx33d::eye(), newCamMatForUndistort, image_size, CV_16SC2, map1, map2);

	//remaps the Mat accoring to unwarping data from "fisheye::initUndistortRectifyMap"
	//no reason to use INTER_LINEAR here, I just don't know what else to put here
	remap(src, src_unwarped, map1, map2, INTER_LINEAR);
	imshow("Unwarped", src_unwarped);
}