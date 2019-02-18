#include <opencv2/opencv.hpp>
#include <iostream>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

const float GOOD_MATCH_PERCENT = 0.10f;
string dashes = "---------------------------------------------------------------";

void computeC2MC1(const Mat &R1, const Mat &tvec1, const Mat &R2, const Mat &tvec2,
	Mat &R_1to2, Mat &tvec_1to2)
{
	//c2Mc1 = c2Mo * oMc1 = c2Mo * c1Mo.inv()
	R_1to2 = R2 * R1.t();
	/*
	cout <<
		endl << dashes << endl << "R2: " << endl << R2 <<
		endl << dashes << endl << "R1: " << endl << R1 <<
		endl << dashes << endl << "tvec1: " << endl << tvec1 <<
		endl << dashes << endl << "tvec2: " << endl << tvec2 <<
		endl << dashes << endl;
	*/

	tvec_1to2 = R2 * (-R1.t()*tvec1) + tvec2;
}

Mat computeHomography(const Mat &R_1to2, const Mat &tvec_1to2, const double d_inv, const Mat &normal)
{
	Mat homography = R_1to2 + d_inv * tvec_1to2*normal.t();
	return homography;
}

int main(int argc, char** argv)
{

	Mat src; Mat src_unwarped; 
	Mat newCamMatForUndistort;
	Mat map1, map2;

	//Open default camera
	VideoCapture cap(0);

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
	imshow("first unwarp", src_unwarped);
	//imwrite("C:\\Users\\Gabriel Young\\Desktop\\output\\yyeet.jpeg", src_unwarped);

	Mat ref = imread("Ref_POC.jpeg");
	resize(ref, ref, Size(), .5, .5);
	imshow("ok", ref);
	vector<KeyPoint> keypointsRef;
	Mat descriptorsRef;

	Ptr<GFTTDetector> detector = GFTTDetector::create();
	detector->detect(ref, keypointsRef);

	Ptr<BriefDescriptorExtractor> extractor = BriefDescriptorExtractor::create();
	extractor->compute(ref, keypointsRef, descriptorsRef);

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
		resize(src_unwarped, src_unwarped, Size(), .5, .5);
		vector<KeyPoint> keypointsLive;
		Mat descriptorsLive;

		detector->detect(src_unwarped, keypointsLive);
		extractor->compute(src_unwarped, keypointsLive, descriptorsLive);

		//Matcher
		vector<DMatch> matches;
		Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);
		matcher->match(descriptorsRef, descriptorsLive, matches);

		//sorts matches by confidence level
		sort(matches.begin(), matches.end());

		//kills bad matches
		const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
		matches.erase(matches.begin() + numGoodMatches, matches.end());

		Mat imMatches;
		drawMatches(ref, keypointsRef, src_unwarped, keypointsLive, matches, imMatches);
		imshow("THE LINES", imMatches);

		//get location of good matches 
		vector<Point2f> pointsRef, pointsLive;
		for (size_t i = 0; i < matches.size(); i++) {
			pointsRef.push_back(keypointsRef[matches[i].queryIdx].pt);
			pointsLive.push_back(keypointsLive[matches[i].queryIdx].pt);
		}

		vector<Point3f> pointsRef3D, pointsLive3D;

		for (int i = 0; i < pointsRef.size(); i++) {
			Point3f point;
			point.x = pointsRef[i].x;
			point.y = pointsRef[i].y;
			point.z = 0;
			pointsRef3D.push_back(point);
		}
		for (int i = 0; i < pointsLive.size(); i++) {
			Point3f point;
			point.x = pointsLive[i].x;
			point.y = pointsLive[i].y;
			point.z = 0;
			pointsLive3D.push_back(point);
		}

		Mat rvecRef, tvecRef;
		Mat rvecLive, tvecLive;
		solvePnP(pointsRef3D, pointsRef, K, D, rvecRef, tvecRef);
		solvePnP(pointsLive3D, pointsLive, K, D, rvecLive, tvecLive);

		Mat R_1to2, t_1to2;
		Mat RRef, RLive;
		Rodrigues(rvecRef, RRef);
		Rodrigues(rvecLive, RLive);

		computeC2MC1(RRef, tvecRef, RLive, tvecLive, R_1to2, t_1to2);

		Mat rvec_1to2;
		Rodrigues(R_1to2, rvec_1to2);
		//cout << endl << "I MIGHT BE DISPLACEMENT: Mat: " << R_1to2 << endl;
		//cout << endl << "I MIGHT BE DISPLACEMENT : vector: " << rvec_1to2 << endl;
		cout << endl << rvec_1to2.at<double>(0, 0) << "    " << rvec_1to2.at<double>(0, 1) << "    " << rvec_1to2.at<double>(0, 2) << endl;

		//Mat normalTemp = (Mat_<double>(3, 1) << 0, 0, 1);
		//Mat normal = RRef * normalTemp;

		//Mat originTemp(3, 1, CV_64F, Scalar(0));
		//Mat origin = RRef * originTemp + tvecRef;

		//double d_inv1 = 1.0 / normal.dot(origin);

		//Mat homo_dist = computeHomography(R_1to2, t_1to2, d_inv1, normal);
		//Mat homo = K * homo_dist * K.inv();

		////dist = euclidean
		//homo /= homo.at<double>(2, 2);
		//homo_dist /= homo_dist.at<double>(2, 2);

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