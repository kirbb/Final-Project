#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <math.h>
#include <vector>
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

Mat greenscreen(const Mat& foreground, const Scalar& replacementColor) {

	Mat image = foreground.clone(); // Initialize a copy of the foreground image to alter

	int size = 4; // Size for the dimensions of the histogram

	// create an array of the histogram dimensions
	// size is a constant - the # of buckets in each dimension
	int dims[] = { size, size, size };

	// create 3D histogram of integers initialized to zero	
	Mat hist(3, dims, CV_32S, Scalar::all(0));

	int bucketSize = 256 / size;
	Vec3b colors;
	int r, g, b;

	for (int row = 0; row < foreground.rows; row++) {
		// std::cout << row << std::endl;
		for (int col = 0; col < foreground.cols; col++) {
			//std::cout << col << std::endl;
			if (row == 7 && col == 294) {
				int i = 0;
			}
			colors = foreground.at<Vec3b>(row, col); // channel is 0,1,2 (blue, green, red)
			r = colors[2] / bucketSize;
			g = colors[1] / bucketSize;
			b = colors[0] / bucketSize;

			hist.at<int>(r, g, b)++;
		}
	}

	int redBucket, greenBucket, blueBucket;
	int highest = 0;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				if (hist.at<int>(i, j, k) > highest) {
					highest = hist.at<int>(i, j, k);
					redBucket = i;
					greenBucket = j;
					blueBucket = k;
				}
			}
		}
	}


	int cRed = (redBucket * bucketSize) + (bucketSize / 2);
	int cGreen = (greenBucket * bucketSize) + (bucketSize / 2);
	int cBlue = (blueBucket * bucketSize) + (bucketSize / 2);

	for (int row = 0; row < foreground.rows; row++) {
		for (int col = 0; col < foreground.cols; col++) {
			colors = foreground.at<Vec3b>(row, col); // channel is 0,1,2 (blue, green, red)

			if (abs(cRed - colors[2]) < bucketSize / 2 &&
				abs(cGreen - colors[1]) < bucketSize / 2 &&
				abs(cBlue - colors[0]) < bucketSize / 2) {
				image.at<Vec3b>(row, col)[0] = replacementColor[0];
				image.at<Vec3b>(row, col)[1] = replacementColor[1];
				image.at<Vec3b>(row, col)[2] = replacementColor[2];
			}
		}
	}

	return image;
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

int RPS() {
	vector<vector<vector<Point>>> library;
	loadReferenceImages(library);

	Mat image = imread("Reference/scissors.jpg");
	Size targetSize = image.size();

	VideoCapture camera;
	Mat frame;
	camera.open(0, CAP_ANY);
	if (!camera.isOpened()) {
		cout << "ERROR: Unable to open camera" << endl;
		return -1;
	}

	while (true) {

		for (int i = 0; i < 5; i++) {
			camera >> frame;
		}

		if (frame.empty()) {
			cout << "ERROR: Couldn't find frame" << endl;
			break;
		}

		// Convert frame to grayscale
		//frame = greenscreen(frame, Scalar(0, 255, 0));
		Mat compare = frame.clone();
		cvtColor(compare, compare, COLOR_BGR2GRAY);

		// Might need to apply smoothing

		// Apply edge detection for comparison
		Canny(compare, compare, 50, 100);
		//adaptiveThreshold(compare, compare, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 5);

		/*imshow("test", compare);
		waitKey(0);*/

		// Get contours from the frame
		vector<vector<Point>> contours;
		findContours(compare, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		drawContours(frame, contours, -1, Scalar(0, 255, 255));

		int bestMatch = -1;
		double bestScore = -10.0, matchScore = 0.0;

		for (const auto& frameContour : contours) {
			// Compare contours to each image in the library
			for (int i = 0; i < library.size(); i++) {
				// Compare the two contours
				matchScore = matchShapes(frameContour, library[i][0], CONTOURS_MATCH_I1, 0);
				cout << "i: " << i << " score: " << matchScore << endl;

				// Update best match if needed
				if (matchScore > bestScore && matchScore < 100.0) {
					bestScore = matchScore;
					bestMatch = i;
				}
			}
		}


		putText(frame, "Best Match: " + to_string(bestMatch) + " Score: " + to_string(matchScore), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
		imshow("Camera", frame);

		//imshow("Camera", frame);

		int k = waitKey(1);
		if (k % 256 == 32) break;
	}


	camera.release();
	destroyAllWindows();

	return 1;
}

int testHandDetection() {
	VideoCapture camera;
	Mat frame;
	camera.open(0, CAP_ANY);
	if (!camera.isOpened()) {
		cout << "ERROR: Unable to open camera" << endl;
		return -1;
	}

	while (true) {

		camera >> frame;

		if (frame.empty()) {
			cout << "ERROR: Couldn't find frame" << endl;
			break;
		}

		// Convert frame to grayscale
		//frame = greenscreen(frame, Scalar(0, 255, 0));
		Mat compare = frame.clone();
		compare = greenscreen(compare, Scalar(0, 255, 0));
		imshow("test", compare);
		cvtColor(compare, compare, COLOR_BGR2GRAY);

		// Might need to apply smoothing

		// Apply edge detection for comparison
		Canny(compare, compare, 50, 100);
		//adaptiveThreshold(compare, compare, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 5);

		// Get contours from the frame
		vector<vector<Point>> contours;
		findContours(compare, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		drawContours(frame, contours, -1, Scalar(0, 255, 255));

		imshow("Camera", frame);

		int k = waitKey(1);
		if (k % 256 == 32) break;
	}
}

int testBGSubtractor() {
	vector<vector<vector<Point>>> library;
	loadReferenceImages(library);

	Ptr<BackgroundSubtractor> bgSubtractor;
	bgSubtractor = createBackgroundSubtractorKNN();

	VideoCapture camera;
	camera.open(0, CAP_ANY);

	if (!camera.isOpened()) {
		cout << "ERROR: Unable to open camera" << endl;
		return -1;
	}

	Mat frame, mask;
	while (true) {
		camera >> frame;
		if (frame.empty()) break;

		bgSubtractor->apply(frame, mask);

		Mat compare = mask.clone();

		// Apply edge detection for comparison
		Canny(compare, compare, 50, 100);
		//adaptiveThreshold(compare, compare, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 5);

		// Get contours from the frame
		vector<vector<Point>> contours;
		findContours(compare, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		drawContours(frame, contours, -1, Scalar(0, 255, 255));

		int bestMatch = -1;
		double bestScore = -10.0, matchScore = 0.0;

		//for (int i = 0; i < library.size(); i++) {
		//	// Compare the two contours
		//	matchScore = matchShapes(contours, library[i][0], CONTOURS_MATCH_I1, 0);
		//	cout << "i: " << i << " score: " << matchScore << endl;

		//	// Update best match if needed
		//	if (matchScore > bestScore && matchScore < 100.0) {
		//		bestScore = matchScore;
		//		bestMatch = i;
		//	}
		//}

		for (int i = 0; i < library.size(); i++) {
			for (int j = 0; j < library[i].size(); j++) {
				if (contours.empty() || library[i].empty() || library[i][j].empty()) {
					// Add appropriate error handling or continue to the next iteration
					continue;
				}

				// Compare the contours
				matchScore = matchShapes(contours[0], library[i][j], CONTOURS_MATCH_I1, 0);
				cout << "i: " << i << " j: " << j << " score: " << matchScore << endl;

				// Update best match if needed
				if (matchScore > bestScore && matchScore < 100.0) {
					bestScore = matchScore;
					bestMatch = i;
				}
			}
		}

		putText(frame, "Best Match: " + getSign(bestMatch) + " Score: " + to_string(matchScore), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);

		imshow("Frame", frame);
		imshow("Mask", mask);

		int k = waitKey(1);
		if (k % 256 == 32) break;
	}

	return 1;
}

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

vector<vector<Point>> loadContours() {
	vector<vector<Point>> contours;
	vector<vector<Point>> temp;
	int largestContour;

	Mat image;

	image = imread("Reference/rock.jpg");
	cvtColor(image, image, COLOR_BGR2HSV);
	inRange(image, Scalar(0, 45, 0), Scalar(255, 255, 255), image);
	blur(image, image, Size(10, 10));
	threshold(image, image, 200, 255, THRESH_BINARY);
	findContours(image, temp, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	largestContour = findLargestContour(temp);
	contours.push_back(temp[largestContour]);
	temp.clear();

	image = imread("Reference/paper.jpg");
	cvtColor(image, image, COLOR_BGR2HSV);
	inRange(image, Scalar(0, 30, 0), Scalar(255, 255, 255), image);
	blur(image, image, Size(10, 10));
	threshold(image, image, 200, 255, THRESH_BINARY);
	findContours(image, temp, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	largestContour = findLargestContour(temp);
	contours.push_back(temp[largestContour]);
	temp.clear();

	image = imread("Reference/scissors.jpg");
	cvtColor(image, image, COLOR_BGR2HSV);
	inRange(image, Scalar(0, 20, 0), Scalar(255, 255, 255), image);
	blur(image, image, Size(10, 10));
	threshold(image, image, 200, 255, THRESH_BINARY);
	findContours(image, temp, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	largestContour = findLargestContour(temp);
	contours.push_back(temp[largestContour]);
	temp.clear();

	return contours;
}

int testConvexHull() {
	vector<vector<Point>> library = loadContours();

	int H_MIN = 0; // minimum Hue
	int H_MAX = 180; // maximum Hue
	int S_MIN = 0; // minimum Saturation
	int S_MAX = 255; // maximum Saturation
	int lower = 0;
	int upper = 15;

	int rLower = 0;
	int rUpper = 255;
	int gLower = 70;
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

	//	cvtColor(frame, frame, COLOR_BGR2HLS);

	//	// Filter by skin color
	//	cv::Mat rangeMask;
	//	cv::inRange(frame, Scalar(lower, 0.8 * 255, 0.6 * 255), Scalar(upper, 0.1 * 255, 0.05 * 255), rangeMask);

	//	// Remove noise
	//	blur(frame, frame, cv::Size(10, 10));

	//	// Threshold the blurred image
	//	threshold(frame, frame, 200, 255, cv::THRESH_BINARY);

	//	imshow("Result", frame);

	//	int k = waitKey(1);
	//	if (k % 256 == 32) break;
	//}

	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Image Section
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	/*while (true) {

		int k = waitKey(1);
		if (k % 256 == 32) break;
	}*/

	Mat image = imread("Reference/paperGuess.jpg");

	Mat hls;
	cvtColor(image, hls, COLOR_BGR2HSV);

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

	int bestMatch = -1;
	double bestScore = 0.0, score = 0.0;
	for (int i = 0; i < library.size(); i++) {
		score = matchShapes(largestContour, library[i], CONTOURS_MATCH_I1, 0);
		cout << "Index: " << i << " Score: " << score << endl;
		if (score > bestScore) {
			bestScore = score;
			bestMatch = i;
		}
	}

	drawContours(image, contours, largestIndex, Scalar(0, 255, 255), 5);
	drawContours(image, library, 0, Scalar(0, 0, 255), 5);
	drawContours(image, library, 1, Scalar(255, 0, 255), 5);
	drawContours(image, library, 2, Scalar(255, 0, 0), 5);

	putText(image, "Best Match: " + getSign(bestMatch) + " Score: " + to_string(bestScore), Point(20, 60), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 255, 255), 5);
	namedWindow("Reference", WINDOW_KEEPRATIO);
	resizeWindow("Reference", image.cols / 5, image.rows / 5);
	imshow("Reference", image);

	waitKey(0);

	return 1;
}

int main(int argc, char* argv[]) {
	testConvexHull();
	//testBGSubtractor();
	//testHandDetection();
	//RPS();
	//Mat rock, paper, scissors;
	//rock = imread("Reference/rock.jpg");
	//paper = imread("Reference/paper.jpg");
	//scissors = imread("Reference/scissors.jpg");

	//Mat blueBG = imread("blueBG.jpg");
	//Mat noBG = imread("noBG.jpg");

	//Mat test = imread("Reference/test.jpg", IMREAD_GRAYSCALE);
	//GaussianBlur(test, test, Size(5, 5), 2.0, 2.0);
	//GaussianBlur(test, test, Size(5, 5), 2.0, 2.0);
	////adaptiveThreshold(test, test, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 5);
	//Canny(test, test, 50, 25);
	//imshow("test", test);
	//waitKey(0);
}