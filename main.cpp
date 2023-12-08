#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <math.h>
#include <vector>
#include <map>
using namespace cv;
using namespace std;

enum Signs {
	Rock,
	Paper,
	Scissors
};

String getSign(int num) {
	switch (num) {
	case Rock: return "Rock";
	case Paper: return "Paper";
	case Scissors: return "Scissors";
	default: return "Invalid sign number passed to getSign()";
	}
}

void loadReferenceImages(vector<vector<vector<Point>>> &library) {
	Mat ref;
	vector<vector<Point>> refContours;

	ref = imread("Reference/scissors.jpg", IMREAD_GRAYSCALE);
	GaussianBlur(ref, ref, Size(7, 7), 2.0, 2.0);
	Canny(ref, ref, 50, 100);
	findContours(ref, refContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	library.push_back(refContours);
	refContours.clear();

	ref = imread("Reference/rock.jpg", IMREAD_GRAYSCALE);
	GaussianBlur(ref, ref, Size(7, 7), 2.0, 2.0);
	Canny(ref, ref, 50, 100);
	findContours(ref, refContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	library.push_back(refContours);
	refContours.clear();

	ref = imread("Reference/paper.jpg", IMREAD_GRAYSCALE);
	GaussianBlur(ref, ref, Size(7, 7), 2.0, 2.0);
	Canny(ref, ref, 50, 100);
	findContours(ref, refContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	library.push_back(refContours);
	refContours.clear();
}


//vector<vector<Point>> loadContours() {
//	vector<vector<Point>> contours;
//	vector<vector<Point>> temp;
//	int largestContour;
//
//	Mat image;
//
//	image = imread("Reference/rock.jpg");
//	cvtColor(image, image, COLOR_BGR2HSV);
//	inRange(image, Scalar(0, 45, 0), Scalar(255, 255, 255), image);
//	blur(image, image, Size(10, 10));
//	threshold(image, image, 200, 255, THRESH_BINARY);
//	findContours(image, temp, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//	largestContour = findLargestContour(temp);
//	contours.push_back(temp[largestContour]);
//	temp.clear();
//
//	image = imread("Reference/paper.jpg");
//	cvtColor(image, image, COLOR_BGR2HSV);
//	inRange(image, Scalar(0, 30, 0), Scalar(255, 255, 255), image);
//	blur(image, image, Size(10, 10));
//	threshold(image, image, 200, 255, THRESH_BINARY);
//	findContours(image, temp, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//	largestContour = findLargestContour(temp);
//	contours.push_back(temp[largestContour]);
//	temp.clear();
//
//	image = imread("Reference/scissors.jpg");
//	cvtColor(image, image, COLOR_BGR2HSV);
//	inRange(image, Scalar(0, 20, 0), Scalar(255, 255, 255), image);
//	blur(image, image, Size(10, 10));
//	threshold(image, image, 200, 255, THRESH_BINARY);
//	findContours(image, temp, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//	largestContour = findLargestContour(temp);
//	contours.push_back(temp[largestContour]);
//	temp.clear();
//
//	return contours;
//}

int findLargestContour(vector<vector<Point>>& contours) {
	int largestIndex = 0;
	double largestArea = contourArea(contours[0]);
	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > largestArea) {
			largestArea = area;
			largestIndex = i;
		}
	}

	return largestIndex;
}

struct indexPoint {
	int index;
	Point point;
};

bool pointClusterCheck(const Point& point1, const Point& point2, double maxDistance) {
	if (norm(point1 - point2) < maxDistance) return true;
	return false;
}

Point calculateCenter(const vector<indexPoint>& points) {
	Point center = Point(0, 0);

	for (int i = 0; i < points.size(); i++) {
		center += points[i].point;
	}
	center *= (1.0 / points.size());

	return center;
}

double calculatePointDistance(const Point& point1, const Point& point2) {
	return norm(point1 - point2);
}

bool comparePoints(const indexPoint& iPoint1, const indexPoint& iPoint2, const Point &center) {
	if (calculatePointDistance(iPoint1.point, center) < calculatePointDistance(iPoint2.point, center)) return true;
	return false;
}

struct mostCentralPointFinder {
	indexPoint operator()(const vector<indexPoint>& points, const Point& center) {
		indexPoint mostCentralPoint = points[0];
		double minDistance = calculatePointDistance(mostCentralPoint.point, center);

		double distance;
		for (int i = 0; i < points.size(); i++) {
			distance = calculatePointDistance(points[i].point, center);
			if (distance < minDistance) {
				mostCentralPoint.point = points[i].point;
				mostCentralPoint.index = i;
				minDistance = distance;
			}
		}

		return mostCentralPoint;
	}
};

vector<int> generateHull(const vector<Point>& contour, double maxDistance) {
	vector<int> hull;
	convexHull(contour, hull);

	vector<Point> contourPoints = contour;

	vector<indexPoint> hullIndexPoints;
	for (int i = 0; i < hull.size(); i++) {
		hullIndexPoints.push_back({i, contourPoints[i]});
	}

	vector<Point> hullPoints;
	for (int i = 0; i < hullIndexPoints.size(); ++i) {
		indexPoint temp = hullIndexPoints[i];
		hullPoints.push_back(temp.point);
	}

	vector<int> clusterLabels;
	partition(hullPoints, clusterLabels, bind(pointClusterCheck, placeholders::_1, placeholders::_2, maxDistance));

	map<int, vector<indexPoint>> pointClusters;
	for (int i = 0; i < hullIndexPoints.size(); i++) {
		int label = clusterLabels[i];
		pointClusters[label].push_back(hullIndexPoints[i]);
	}

	mostCentralPointFinder cursor;
	vector<vector<indexPoint>> points;
	for (const auto& entry : pointClusters) {
		points.push_back(entry.second);
	}

	vector<int> result;
	for (int i = 0; i < points.size(); i++) {
		Point center = calculateCenter(points[i]);
		indexPoint mostCentralPoint = cursor(points[i], center);
		result.push_back(mostCentralPoint.index); 
	}

	return result;
}

int testConvexHull() {
	//vector<vector<Point>> library = loadContours();

	int H_MIN = 0; // minimum Hue
	int H_MAX = 180; // maximum Hue
	int S_MIN = 0; // minimum Saturation
	int S_MAX = 255; // maximum Saturation
	int lower = 0;
	int upper = 15;

	int rLower = 0;
	int rUpper = 255;
	int gLower = 50;
	int gUpper = 255;
	int bLower = 0;
	int bUpper = 255;

	namedWindow("Values", 0);
	//create memory to store trackbar name on window

	createTrackbar("rLower", "Values", &rLower, 255);
	createTrackbar("rUpper", "Values", &rUpper, 255);
	createTrackbar("gLower", "Values", &gLower, 255);
	createTrackbar("gUpper", "Values", &gUpper, 255);
	createTrackbar("bLower", "Values", &bLower, 255);
	createTrackbar("bUpper", "Values", &bUpper, 255);

	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Video Section
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	//VideoCapture camera;
	//camera.open(0, CAP_ANY);

	//if (!camera.isOpened()) {
	//	cout << "ERROR: Unable to open camera" << endl;
	//	return -1;
	//}

	//Mat frame, mask;
	//while (true) {
	//	camera >> frame;
	//	if (frame.empty()) break;

	//	Mat hsv;
	//	cvtColor(frame, hsv, COLOR_BGR2HSV);

	//	namedWindow("HSV", WINDOW_KEEPRATIO);
	//	resizeWindow("HSV", frame.cols / 2, frame.rows / 2);
	//	imshow("HSV", hsv);

	//	Mat mask;
	//	inRange(hsv, Scalar(bLower, gLower, rLower), Scalar(bUpper, gUpper, rUpper), mask); //hue, 0.8 * 255, 0.6 * 255
	//	//inRange(hsv, Scalar(0, 0.1 * 255, 0.05 * 255), Scalar(15, 0.8 * 255, 0.6 * 255), mask);

	//	Mat blurred;
	//	blur(mask, blurred, Size(10, 10));

	//	Mat thresh;
	//	threshold(blurred, thresh, 200, 255, THRESH_BINARY);

	//	vector<vector<Point>> contours;
	//	findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//	
	//	if (!contours.empty()) {
	//		int largestIndex = findLargestContour(contours);
	//		vector<Point> largestContour = contours[largestIndex];
	//		vector<int> hull = generateHull(largestContour, 10.0);
	//		vector<Point> points;
	//		for (int i = 0; i < hull.size(); i++) {
	//			points.push_back(largestContour[hull[i]]);
	//		}
	//		polylines(frame, points, false, Scalar(0, 255, 0), 5);
	//		//drawContours(frame, contours, largestIndex, Scalar(0, 255, 255), 5);
	//	}

	//	//putText(image, "Best Match: " + getSign(bestMatch) + " Score: " + to_string(bestScore), Point(20, 60), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 255, 255), 5);
	//	namedWindow("Reference", WINDOW_KEEPRATIO);
	//	resizeWindow("Reference", frame.cols / 2, frame.rows / 2);
	//	imshow("Reference", frame);

	//	int k = waitKey(1);
	//	if (k % 256 == 32) break;
	//}

	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Image Section
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	while (true) {
		Mat image = imread("pic2.jpg");

		Mat hls;
		cvtColor(image, hls, COLOR_BGR2HSV);
		namedWindow("hsv", WINDOW_KEEPRATIO);
		resizeWindow("hsv", image.cols / 5, image.rows / 5);
		imshow("hsv", hls);

		Mat mask;
		inRange(hls, Scalar(bLower, gLower, rLower), Scalar(bUpper, gUpper, rUpper), mask);

		Mat blurred;
		blur(mask, blurred, Size(10, 10));

		Mat thresh;
		threshold(blurred, thresh, 200, 255, THRESH_BINARY);

		vector<vector<Point>> contours;
		findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		int largestIndex = findLargestContour(contours);
		vector<Point> largestContour = contours[largestIndex];

		/*vector<int> hull = generateHull(largestContour, 10.0);
		vector<Point> points;
		for (int i = 0; i < hull.size(); i++) {
			points.push_back(largestContour[hull[i]]);
		}*/
		//polylines(image, points, false, Scalar(0, 255, 0), 5);

		vector<Point> cHull;
		convexHull(largestContour, cHull);
		drawContours(image, cHull, -1, Scalar(0, 255, 0), 5);

		namedWindow("Reference", WINDOW_KEEPRATIO);
		resizeWindow("Reference", image.cols / 5, image.rows / 5);
		imshow("Reference", image);

		int k = waitKey(1);
		if (k % 256 == 32) break;
	}

	return 1;
}

int main(int argc, char* argv[]) {
	testConvexHull();
}