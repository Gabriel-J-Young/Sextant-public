#include <opencv2/opencv.hpp>
#include <iostream>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(int argc, char** argv) {
	Mat newCamMatForUndistort;
	Mat src; Mat src_unwarped;
	Mat map1, map2;

	//Open default camera
	VideoCapture cap(4);

	if (cap.isOpened() == false)
	{
		cout << "Cannot open camera" << endl;
		cin.get();
		return -1;
	}

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
	cout << "K = " << K << endl;
	cout << "D = " << D << endl;

	cap.read(src);
	//generates undistortions maps from first frame
	Size image_size = src.size();
	fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, Matx33d::eye(), newCamMatForUndistort, 1, image_size);
	fisheye::initUndistortRectifyMap(K, D, Matx33d::eye(), newCamMatForUndistort, image_size, CV_16SC2, map1, map2);

	cout << "Video name (.avi is added automatically): ";
	string nameIn;
	cin >> nameIn;
	stringstream nameS;
	nameS << nameIn << ".avi";
	string name = nameS.str();
	cout << name << endl;


	VideoWriter video(name, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(dWidth, dHeight));

	//no endl here for user input
	cout << "how many seconds: ";
	int seconds;
	cin >> seconds;
	cout << endl << "writing a " << seconds << " second video" << endl;
	for (int i = 0; i < seconds * 10; i++) {
		Mat frame;
		cap >> frame;

		if (frame.empty()) {
			break;
		}
		remap(frame, frame, map1, map2, cv::INTER_LINEAR);

		video.write(frame);
	}
	video.release();
	cout << "wrote a " << seconds << " second video called: " << name << endl;
	waitKey(50000);
	return 0;
}