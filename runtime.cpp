#include <opencv2/opencv.hpp>
#include <iostream>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

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

void cameraPoseFromHomography(Mat homography, Mat& pose) {
	//eye is col then rows. yes, it's retarded
	pose = Mat::eye(3, 4, CV_32FC1);
	float norm1 = (float)norm(homography.col(0));
	float norm2 = (float)norm(homography.col(1));
	float tnorm = (norm1 + norm2) / 2.0f;

	Mat p1 = homography.col(0); // Pointer to first column of H
	Mat p2 = pose.col(0); // Pointer to first column of pose (empty)

	normalize(p1, p2); // Normalize the rotation and copies the column to pose

	p1 = homography.col(1); // Pointer to second column of H
	p2 = pose.col(1); // Pointer to second column of pose (empty)

	normalize(p1, p2); // Normalize the rotation and copies the column to

	p1 = pose.col(0);
	p2 = pose.col(1);

	Mat p3 = p1.cross(p2); // Computes the cross-product of p1 and p2
	Mat c2 = pose.col(2); // Pointer to third column of pose
	p3.copyTo(c2); // Third column is the crossproduct of columns one and two


	pose.col(3) = homography.col(3) / tnorm; //vector t [R|t] is the last column of pose
}

void warpWithHomography(float GOOD_MATCH_PERCENT, Mat src_unwarped, Mat ref, Mat &out) {
	resize(src_unwarped, src_unwarped, Size(), .5, .5);
	resize(ref, ref, Size(), .5, .5);
	vector<KeyPoint> keypointsLive, keypointsRef;
	vector<DMatch> matches;
	vector<Point2f> pointsRef, pointsLive;
	Mat descriptorsLive, descriptorsRef, homography;

	// detect orb features and compute descriptors.
	Ptr<Feature2D> orb = ORB::create();
	orb->detectAndCompute(src_unwarped, Mat(), keypointsLive, descriptorsLive);
	orb->detectAndCompute(ref, Mat(), keypointsRef, descriptorsRef);

	//Matcher
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
	for (size_t i = 0; i < matches.size(); i++) {
		pointsRef.push_back(keypointsRef[matches[i].queryIdx].pt);
		pointsLive.push_back(keypointsLive[matches[i].trainIdx].pt);
	}

	homography = findHomography(pointsRef, pointsLive, RANSAC);
	Mat pose;
	cout << "me!" << endl;
	//it break here
	cameraPoseFromHomography(homography, pose);
	cout << "you!" << endl;
	cout << pose << endl;
	warpPerspective(src_unwarped, out, homography, ref.size());

}


int main(int argc, char** argv)
{
	const float GOOD_MATCH_PERCENT = 0.10f;
	Mat src; Mat src_unwarped; 
	Mat newCamMatForUndistort;
	Mat map1, map2;
	Mat ref;

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
	bSuccess = cap.read(src);
	if (bSuccess == false)
	{
		cout << "Video camera is disconnected" << endl;
		cin.get(); //Wait for any key press
		return 0;
	}
	cout << "Camera open!" << endl;

	Size image_size = src.size();
	//camera unwarping
	fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, Matx33d::eye(), newCamMatForUndistort, 1, image_size);
	fisheye::initUndistortRectifyMap(K, D, Matx33d::eye(), newCamMatForUndistort, image_size, CV_16SC2, map1, map2);
	remap(src, src_unwarped, map1, map2, cv::INTER_LINEAR);

	//writing a referance image
	cout << "do you want to write a referance image? (y/n)";
	string a;
	cin >> a;
	if (a.compare("y") == 0) {
		cout << endl << "writing to file" << endl;
		//writes to build folder.
		imwrite("C:\\Users\\Gabriel Young\\Desktop\\X-Bot\\Vision\\sextant\\build\\Ref_POC.jpeg", src_unwarped);
	}else {
		"not writing to file, using stored image";
	}
	ref = imread("Ref_POC.jpeg");
	resize(src_unwarped, src_unwarped, Size(), .5, .5);
	imshow("first unwarp", src_unwarped);
	imshow("ok", ref);

	/// detect orb features and compute descriptors.
	//Ptr<Feature2D> orb = ORB::create();
	//orb->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
	//orb->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

	/*
	//GFTT-Brief Method
	Ptr<GFTTDetector> detector = GFTTDetector::create();
	detector->detect(ref, keypointsRef);

	Ptr<BriefDescriptorExtractor> extractor = BriefDescriptorExtractor::create();
	extractor->compute(ref, keypointsRef, descriptorsRef);
	*/

	while (true)
	{
		Mat out;
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

		warpWithHomography(GOOD_MATCH_PERCENT, src_unwarped, ref, out);
		imshow("out", out);

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