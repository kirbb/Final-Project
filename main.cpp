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
	Mat frame;
	camera.open(0, CAP_ANY);
	if (!camera.isOpened()) {
		cout << "ERROR: Unable to open camera" << endl;
		return -1;
	}
	namedWindow("Camera");
	while (true) {
		
		camera >> frame;
		if (frame.empty()) {
			cout << "ERROR: Couldn't find frame" << endl;
			break;
		}
		imshow("Camera", frame);

		int k = waitKey(1);
		if (k % 256 == 32) break;
	}

	// After this point, we should have a picture of the player's choice, we need to then run recognition

	imshow("Result", frame);



	waitKey(0);
	return 0;
}