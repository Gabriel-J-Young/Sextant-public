#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

const float GOOD_MATCH_PERCENT = 0.15f;

int main(int argc, char** argv) {
	Mat homo;
	Mat img1Reg;
	Mat img1 = imread("IMG_1365.JPG");
	Mat img2 = imread("image.png");
	resize(img1, img1, Size(), .25, .25);

	//cvtColor(img1, img1, COLOR_BGR2GRAY);
	//cvtColor(img2, img2, COLOR_BGR2GRAY);

	// Variables to store keypoints and descriptors
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors.
	//Ptr<Feature2D> orb = ORB::create();
	//orb->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
	//orb->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

	//auto extractor = BRISK::create();
	//extractor->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
	//extractor->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

	// Match features.
	vector<DMatch> matches;
	/*Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());*/
	//Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);
	//matcher->match(descriptors1, descriptors2, matches);

	Ptr<HarrisLaplaceFeatureDetector> detector = HarrisLaplaceFeatureDetector::create();

	Ptr<BriefDescriptorExtractor> extractor = BriefDescriptorExtractor::create();

	//sort matches
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

	//find homo
	homo = findHomography(points1, points2, RANSAC);

	//use homo to warp image
	warpPerspective(img1, img1Reg, homo, img2.size());
	imshow("yeet window", img1Reg);
	
	waitKey(0);
}