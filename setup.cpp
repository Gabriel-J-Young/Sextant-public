#include <opencv2/opencv.hpp>
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

	cout << 
		endl << dashes << endl << "R2: " << endl << R2 << 
		endl << dashes << endl << "R1: " << endl << R1 << 
		endl << dashes << endl << "tvec1: " << endl << tvec1 << 
		endl << dashes << endl << "tvec2: " << endl << tvec2 << 
		endl << dashes << endl;

	tvec_1to2 = R2 * (-R1.t()*tvec1) + tvec2;
}

Mat computeHomography(const Mat &R_1to2, const Mat &tvec_1to2, const double d_inv, const Mat &normal)
{
	Mat homography = R_1to2 + d_inv * tvec_1to2*normal.t();
	return homography;
}

int main(int argc, char** argv) {
	Mat img1Reg;
	Mat img1 = imread("lower.jpg");
	Mat img2 = imread("upper.jpg");
	//resize(img1, img1, Size(), .25, .25);
	resize(img1, img1, Size(), .25, .25);
	resize(img2, img2, Size(), .25, .25);

	Mat K = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	Mat D = (Mat_<double>(1, 4) << 0, 0, 0, 0);

	//cvtColor(img1, img1, COLOR_BGR2GRAY);
	//cvtColor(img2, img2, COLOR_BGR2GRAY);

	// Variables to store keypoints and descriptors
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	//// detect orb features and compute descriptors.
	//Ptr<Feature2D> orb = ORB::create();
	//orb->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
	//orb->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

	//auto extractor = BRISK::create();
	//extractor->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
	//extractor->detectAndCompute(img2, Mat(), keypoints2, descriptors2);
	 
	Ptr<GFTTDetector> detector = GFTTDetector::create();
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);


	//for (int i = 0; i < keypoints1.size(); i++) {
	//	cout << keypoints1[i].pt;
	//}

	Ptr<BriefDescriptorExtractor> extractor = BriefDescriptorExtractor::create();
	extractor->compute(img1, keypoints1, descriptors1);
	extractor->compute(img2, keypoints2, descriptors2);


	/*cout << keypoints1[0].pt;
	for (int i = 0; i < keypoints1.size(); i++) {
		cout << keypoints1[i].pt;
	}*/

	// Match features.
	vector<DMatch> matches;
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);
	matcher->match(descriptors1, descriptors2, matches);

	//// Match features.
	//vector<DMatch> matches;
	//Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
	//matcher->match(descriptors1, descriptors2, matches);

	cout << "did you get to me?" << endl;


	//sort matches`
	sort(matches.begin(), matches.end());

	//kills bad matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	cout << matches.size() << endl;
	cout << numGoodMatches << endl;
	
	matches.erase(matches.begin() + numGoodMatches, matches.end());
	
	Mat imMatches;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, imMatches);

	imshow("Display window", imMatches); 

	//extract location of good matches
	vector<Point2f> points1, points2;

	for (size_t i = 0; i < matches.size(); i++) {
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].queryIdx].pt);
	}

	vector<Point3f> points13D, points23D;

	for (int i = 0; i < points1.size(); i++) {
		Point3f point; 
		point.x = points1[i].x;
		point.y = points1[i].y;
		point.z = 0;
		points13D.push_back(point);
	}

	for (int i = 0; i < points2.size(); i++) {
		Point3f point;
		point.x = points2[i].x;
		point.y = points2[i].y;
		point.z = 0;
		points23D.push_back(point);
	}

	Mat rvec1, tvec1;
	Mat rvec2, tvec2;
	cout << "points13D length: " << points13D.size() << "point1 length: " << points1.size() << endl;
	cout << "points23D length: " << points23D.size() << "point2 length: " << points2.size() << endl;
	solvePnP(points13D, points1, K, D, rvec1, tvec1);
	solvePnP(points23D, points2, K, D, rvec2, tvec2);

	Mat R_1to2, t_1to2;
	Mat R1, R2;
	Rodrigues(rvec1, R1);
	Rodrigues(rvec2, R2);

	computeC2MC1(R1, tvec1, R2, tvec2, R_1to2, t_1to2);

	Mat rvec_1to2;
	Rodrigues(R_1to2, rvec_1to2);
	cout << "I MIGHT BE DISPLACEMENT: Mat: " << R_1to2 << endl;
	cout << "I MIGHT BE DISPLACEMENT : vector: " << rvec_1to2 << endl;

	cout << endl << "RVEC1 IS:        --------" << endl << rvec1 << endl;
	cout << endl << "TVEC1 IS:        --------" << endl << tvec1 << endl;

	Mat normal = (Mat_<double>(3, 1) << 0, 0, 1);
	Mat normal1 = R1 * normal;

	Mat origin(3, 1, CV_64F, Scalar(0));
	Mat origin1 = R1 * origin + tvec1;
	cout << endl << "origin is: " << origin1 << endl;
	double d_inv1 = 1.0 / normal1.dot(origin1);
	cout << "d_inv1: " << d_inv1 << endl;

	cout <<
		endl << dashes << endl << "R_1to2: " << endl << R_1to2 << 
		endl << dashes <<endl << "t_1to2: " << endl << t_1to2 << 
		endl << dashes << endl << "d_inv1: " << endl << d_inv1 << 
		endl << dashes << endl << "normal1: " << endl << normal1 << 
		endl << dashes << endl;
	Mat homo_dist = computeHomography(R_1to2, t_1to2, d_inv1, normal1);
	Mat homo = K * homo_dist * K.inv();

	cout << "homo_dist: " << endl <<  homo_dist << endl;
	homo /= homo.at<double>(2, 2);
	homo_dist /= homo_dist.at<double>(2, 2);


	////find homo
	//homo = findHomography(points1, points2, RANSAC);
	//cout << "THE SIZE OF HOMO" << homo.size() << endl;

	cout << "homo is: " << endl << homo << endl;

	////use homo to warp image
	warpPerspective(img2, img1Reg, homo, img2.size());
	imshow("yeet window", img1Reg);
	
	imwrite("C:\\Users\\Gabriel Young\\Desktop\\X-Bot\\Vision\\sextant\\build\\yeet.jpg", img1Reg);

	waitKey(0);
}