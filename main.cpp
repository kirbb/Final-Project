#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <vector>
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	VideoCapture camera;
	camera.open(0, CAP_ANY);
	if (!camera.isOpened()) {
		cout << "ERROR: Unable to open camera" << endl;
		return -1;
	}
	namedWindow("Camera", WINDOW_AUTOSIZE);
	while (true) {
		Mat frame;
		camera >> frame;
		if (frame.empty()) {
			cout << "ERROR: Blank frame grabbed" << endl;
			break;
		}
		imshow("Camera", frame);
		if (waitKey(30) >= 0) break;
	}
	return 0;
}