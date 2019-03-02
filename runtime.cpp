#include "runtime.h"

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

void cameraPoseFromHomography(const Mat& homography, Mat& pose) {

	//for this, the "p1" and "p2" and ect. don't properly copy you need to use
	//".copyTo()" to copy Mats

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

	pose.col(3) = homography.col(2) / tnorm; //vector t [R|t] is the last column of pose
	//cout << pose.col(3) << endl;
}

void keypointMatches(float GOOD_MATCH_PERCENT, vector<KeyPoint> keypointsRef, Mat descriptorsRef, Mat src_unwarped, Mat ref, Mat& img_matches, vector<Point2f>& pointsRef, vector<Point2f>& pointsLive) {
	vector<KeyPoint> keypointsLive;
	vector<DMatch> matches;
	Mat descriptorsLive;
	Mat img_keypointsLive;

	// detect orb features and compute descriptors.
	Ptr<Feature2D> orb = ORB::create();
	orb->detectAndCompute(src_unwarped, Mat(), keypointsLive, descriptorsLive);

	drawKeypoints(src_unwarped, keypointsLive, img_keypointsLive, Scalar(0, 0, 255));
	imshow("better keypoints?", img_keypointsLive);

	//Matcher
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);
	matcher->match(descriptorsRef, descriptorsLive, matches);

	//sorts matches by confidence level
	sort(matches.begin(), matches.end());

	//kills bad matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());

	resize(ref, ref, Size(), .5, .5);
	resize(src_unwarped, src_unwarped, Size(), .5, .5);

	drawMatches(ref, keypointsRef, src_unwarped, keypointsLive, matches, img_matches);

	//get location of good matches 
	for (size_t i = 0; i < matches.size(); i++) {
		pointsRef.push_back(keypointsRef[matches[i].queryIdx].pt);
		pointsLive.push_back(keypointsLive[matches[i].trainIdx].pt);
	}
}

void videoKeypointMatches(float GOOD_MATCH_PERCENT, vector<vector<KeyPoint>> videoKeypointsRef, vector<Mat> videoDescriptorsRef, Mat src_unwarped, vector<vector<Point2f>>& videoPointsRef, vector<vector<Point2f>>& videoPointsLive) {
	VideoWriter video("vid_matches.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0f, Size(1920, 1080));
	vector<KeyPoint> keypointsLive;
	vector<vector<DMatch>> matches;
	Mat descriptorsLive;
	Mat img_keypointsLive;

	// detect orb features and compute descriptors.
	Ptr<Feature2D> orb = ORB::create();
	orb->detectAndCompute(src_unwarped, Mat(), keypointsLive, descriptorsLive);

	drawKeypoints(src_unwarped, keypointsLive, img_keypointsLive, Scalar(0, 0, 255));
	imshow("live keypoints", img_keypointsLive);
	
	for (int i = 0; i < videoDescriptorsRef.size(); i++) {
		//Matcher
		vector<DMatch> someMatches;
		Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);
		matcher->match(videoDescriptorsRef[i], descriptorsLive, someMatches);
		matches.push_back(someMatches);
	}

	//sorts match vectors by number of matches
	sort(matches.begin(), matches.end(), vectorBigToSmall);

	const int numGoodMatchVectors = matches.size() * .5f;
	matches.erase(matches.begin() + numGoodMatchVectors, matches.end());

	for (int i = 0; i < matches.size(); i++) {
		//sorts matches by confidence level
		sort(matches[i].begin(), matches[i].end());

		//kills bad matches
		const int numGoodMatches = matches[i].size() * GOOD_MATCH_PERCENT;
		matches[i].erase(matches[i].begin() + numGoodMatches, matches[i].end());
	}

	//get location of good matches 
	for (size_t j = 0; j < matches.size(); j++) {
		vector<Point2f> videoPointsRefFrame;
		vector<Point2f> videoPointsLiveFrame;
		for (size_t i = 0; i < matches[j].size(); i++) {
			videoPointsRefFrame.push_back(videoKeypointsRef[j][matches[j][i].queryIdx].pt);
			videoPointsLiveFrame.push_back(keypointsLive[matches[j][i].trainIdx].pt);
		}
		videoPointsRef.push_back(videoPointsRefFrame);
		videoPointsLive.push_back(videoPointsLiveFrame);
		videoPointsRefFrame.clear();
		videoPointsLiveFrame.clear();
	}
}

int refVideoProcessor(vector<vector<KeyPoint>>& refVideoKeypoints,  vector<Mat>& refVideoDescriptors) {
	VideoWriter video("vid_ref_keypoints.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0f, Size(1920, 1080));
	VideoCapture cap("vid_ref.avi");
	if (!cap.isOpened()) {
		cout << "Could not open reference video" << endl;
		return -1;
	}

	for (int i = 0; i < cap.get(CAP_PROP_FRAME_COUNT); i++){
		Mat frame;
		cap >> frame;
		if (frame.empty()) {
			break;
		}

	Ptr<Feature2D> orb = ORB::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;
	orb->detectAndCompute(frame, Mat(), keypoints, descriptors);

	drawKeypoints(frame, keypoints, frame, Scalar(0, 255, 0));
	video.write(frame);
	refVideoKeypoints.push_back(keypoints);
	refVideoDescriptors.push_back(descriptors);
	}
}

void displacement(vector<Point2f> pointsRef, vector<Point2f> pointsLive, Mat K, Mat D, Mat& rvec, Mat& tvec) {
	vector<Point3f> pointsRef3D;

	for (int i = 0; i < pointsRef.size(); i++) {
		Point3d point;
		point.x = (float)pointsRef[i].x;
		point.y = (float)pointsRef[i].y;
		point.z = 1.0f;
		pointsRef3D.push_back(point);
	}

	solvePnPRansac(pointsRef3D, pointsLive, K, D, rvec, tvec);
	cout << "rotation vector length: " << sum(rvec) << endl;
	cout << "translation vector length: " << sum(tvec) << endl;
}

void homographyPerspectiveWarp (vector<Point2f> pointsRef, vector<Point2f> pointsLive, Mat src_unwarped, Mat ref, Mat& homography, Mat &img_warpedToPerspective) {
	resize(src_unwarped, src_unwarped, Size(), .5, .5);
	resize(ref, ref, Size(), .5, .5);

	homography = findHomography(pointsRef, pointsLive, RANSAC);
	
	Mat pose;
	//it break here because Mat assignment is broken right now. see method for details.
	//two people on stack overflow thread say this answer is trash
	//cameraPoseFromHomography(homography, pose);
	warpPerspective(src_unwarped, img_warpedToPerspective, homography, ref.size());
}



int main(int argc, char** argv) {
	const float GOOD_MATCH_PERCENT = 0.1f;
	Mat src; Mat src_unwarped; 
	Mat newCamMatForUndistort;
	Mat map1, map2;
	Mat ref;
	Mat descriptorsRef;
	vector<KeyPoint> keypointsRef;

	
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

	//spam these to get a later frame. there doesn't seem to be a better way
	cap.read(src);

	bSuccess = cap.read(src);
	if (bSuccess == false)
	{
		cout << "Video camera is disconnected" << endl;
		cin.get(); //Wait for any key press
		return 0;
	}
	cout << "Camera open!" << endl;

	Size image_size = src.size();
	//generates undistortions maps from first frame
	fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, Matx33d::eye(), newCamMatForUndistort, 1, image_size);
	fisheye::initUndistortRectifyMap(K, D, Matx33d::eye(), newCamMatForUndistort, image_size, CV_16SC2, map1, map2);
	
	remap(src, src_unwarped, map1, map2, cv::INTER_LINEAR);

	/*
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
	*/

	vector<vector<KeyPoint>> refVideoKeypoints;
	vector<Mat> refVideoDescriptors;
	refVideoProcessor(refVideoKeypoints, refVideoDescriptors);

	Ptr<Feature2D> orb = ORB::create();
	orb->detectAndCompute(ref, Mat(), keypointsRef, descriptorsRef);

	//Mat img_keypointsRef;
	//drawKeypoints(src_unwarped, keypointsRef, img_keypointsRef, Scalar(255, 0, 0));


	while (true)
	{
		//imshow("ref keys: ", img_keypointsRef);
		Mat img_warpedToPerspective, homography, img_matches, rvec, tvec;
		vector<Mat> rotations, translations, normals;
		vector<Point2f> pointsRef, pointsLive;
		// read a new frame from video breaking the while loop if the frames cannot be captured
		bSuccess = cap.read(src);
		if (bSuccess == false)
		{
			cout << "Video camera is disconnected" << endl;
			//Wait for any key press
			cin.get();
			break;
		}

		//imshow("before", src);

		//remaps the Mat accoring to unwarping data from "fisheye::initUndistortRectifyMap"
		remap(src, src_unwarped, map1, map2, cv::INTER_LINEAR);

		//keypointMatches(GOOD_MATCH_PERCENT, keypointsRef, descriptorsRef, src_unwarped, ref, img_matches, pointsRef, pointsLive);
		//imshow("THE LINES", img_matches);

		vector<vector<Point2f>> videoPointsRef;
		vector<vector<Point2f>> videoPointsLive;
		videoKeypointMatches(GOOD_MATCH_PERCENT, refVideoKeypoints, refVideoDescriptors, src_unwarped, videoPointsRef, videoPointsLive);
		//assume rvec is 3 col and one row
		Mat sum_rvec = Mat::zeros(Size(3, 1), CV_64FC1);
		Mat sum_tvec = Mat::zeros(Size(3, 1), CV_64FC1);

		cout << "videoPointsRef.size()" << videoPointsRef.size() << endl;
		for (int i = 0; i < videoPointsRef.size(); i++) {
			displacement(videoPointsRef[i], videoPointsLive[i], K, D, rvec, tvec);
			sum_rvec.at<float>(0, 0) += rvec.at<float>(0, 0);
			sum_rvec.at<float>(1, 0) += rvec.at<float>(1, 0);
			sum_rvec.at<float>(2, 0) += rvec.at<float>(2, 0);

			sum_tvec.at<float>(0, 0) += tvec.at<float>(0, 0);
			sum_tvec.at<float>(1, 0) += tvec.at<float>(1, 0);
			sum_tvec.at<float>(2, 0) += tvec.at<float>(2, 0);
		}

		cout << "sum_rvec: " << sum_rvec << endl;
		cout << "sum_tvec: " << sum_tvec << endl;

		sum_rvec /= videoPointsRef.size();
		sum_tvec /= videoPointsRef.size();


		cout << "average rotation vector lenght: " << sum(sum_rvec) << endl;
		cout << "average translation vector lenght: " << sum(sum_tvec) << endl;
		//displacement(pointsRef, pointsLive, K, D, rvec, tvec);

		//homographyPerspectiveWarp(pointsRef, pointsLive, src_unwarped, ref, homography, img_warpedToPerspective);
		//imshow("img_warpedToPerspective", img_warpedToPerspective);

		//now, I'm trying this: https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#int%20decomposeHomographyMat%28InputArray%20H,%20InputArray%20K,%20OutputArrayOfArrays%20rotations,%20OutputArrayOfArrays%20translations,%20OutputArrayOfArrays%20normals%29
		//the data generated by this function might be useful. show to jeff
		//there should be a better way with less uncertainty
		//also the opencv tutorials indicate we should ne using solvePnP for camera pose
		/*
		decomposeHomographyMat(homography, K, rotations, translations, normals);
		//change to auto and &Mat if this dont work also rename rotMat in cout
		for (Mat &rotMat : rotations) {
			//cout << "rot Mat: " << endl << rotMat << endl;
			cout << "rot Mat vector: " << sum(rotMat) << endl; 
		}
		for (Mat &transMat : translations) {
			//cout << endl << "trans Mat: " << transMat << endl;
			cout << "trans Mat vector: " << sum(transMat) << endl;
		}
		*/

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