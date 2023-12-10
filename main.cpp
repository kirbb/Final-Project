// main.cpp
// This code demonstrates using Haar Cascade Classifiers to recognize hand signs
// Author: Ibrahim Deria and Kirby Vandel

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

CascadeClassifier rock_cascade;
CascadeClassifier paper_cascade;
CascadeClassifier scissors_cascade;

int playerSign = 0;

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

// detectSign - determines all of the hand signs being shown in the provided frame
// preconditions: image is in RGB format
// postconditions: a string stating the most abundant hand sign is returned
String detectSign(const Mat& frame) {
	// Duplicates and creates a grayscale image
	Mat grayscale;
	cvtColor(frame, grayscale, COLOR_BGR2GRAY);

	// Scales the image down for more accurate detection
	const int scale = 2;
	Mat scaled(cvRound(grayscale.rows / scale), cvRound(grayscale.cols / scale), CV_8UC1);
	resize(grayscale, scaled, scaled.size(), 1.0 / scale, 1.0 / scale);

	// Create empty vectors to store the results of the detection
	vector<Rect> rockDetections;
	vector<Rect> paperDetections;
	vector<Rect> scissorsDetections;
	rock_cascade.detectMultiScale(scaled, rockDetections);
	paper_cascade.detectMultiScale(scaled, paperDetections);
	scissors_cascade.detectMultiScale(scaled, scissorsDetections);

	// Finds the most common sign in the image and stores the result in playerChoice
	String playerChoice = "";
	int numRock = rockDetections.size();
	int numPaper = paperDetections.size();
	int numScissors = scissorsDetections.size();
	if (numRock == numPaper && numScissors == numRock) playerChoice = "No sign detected";
	else {
		if (numRock > numPaper) {
			if (numRock > numScissors) {
				playerChoice = "Sign detected: Rock";
				playerSign = Rock;
			}
			else {
				playerChoice = "Sign detected: Scissors";
				playerSign = Scissors;
			}
		}
		else if (numPaper > numRock) {
			if (numPaper > numScissors) {
				playerChoice = "Sign detected: Paper";
				playerSign = Paper;
			}
			else {
				playerChoice = "Sign detected: Scissors";
				playerSign = Scissors;
			}
		}
		else {
			playerChoice = "Sign detected: Scissors";
			playerSign = Scissors;
		}
	}

	// loops for drawing all of the detected signs in the detection window
	for (int i = 0; i < rockDetections.size(); i++) {
		rectangle(scaled, rockDetections[i], Scalar(0, 255, 0));
	}

	for (int i = 0; i < paperDetections.size(); i++) {
		rectangle(scaled, paperDetections[i], Scalar(0, 0, 255));
	}

	for (int i = 0; i < scissorsDetections.size(); i++) {
		rectangle(scaled, scissorsDetections[i], Scalar(255, 0, 0));
	}

	imshow("Detection", scaled);

	return playerChoice;
}

// getWinner - computes the result of two rock paper scissors throws
// preconditions: both playerChoice and CPUchoice are integers that fall within our enumerated signs
// postconditions: returns a string representing which player won or if the result was a tie
String getWinner(int playerChoice, int CPUchoice) {
	if (playerChoice == Rock) {
		if (CPUchoice == Rock) return "It's a tie!";
		else if (CPUchoice == Paper) return "You lose!";
		else return "You win!";
	}
	else if (playerChoice == Paper) {
		if (CPUchoice == Paper) return "It's a tie!";
		else if (CPUchoice == Scissors) return "You lose!";
		else return "You win!";
	}
	else {
		if (CPUchoice == Scissors) return "It's a tie!";
		else if (CPUchoice == Rock) return "You lose!";
		else return "You win!";
	}
}

// main - loads the cascades and runs the game of rock paper scissors
// preconditions: all necessary files for the cascades are stored in the same folder as the code being run
// postconditions: runs the virtual game of rock paper scissors
//					will continue to run until the program is either forcibly closed or the user presses escape
int main(int argc, char* argv[]) {
	
	// Load all of the pre-trained cascades
	if (!rock_cascade.load("rock/Cascade/cascade.xml")) {
		cout << "Could not load Rock cascade" << endl;
		return -1;
	}

	if (!paper_cascade.load("paper/Cascade/cascade.xml")) {
		cout << "Could not load Paper cascade" << endl;
		return -1;
	}

	if (!scissors_cascade.load("scissors/Cascade/cascade.xml")) {
		cout << "Could not load Scissors cascade" << endl;
		return -1;
	}

	// Open the video camera
	VideoCapture camera;
	camera.open(0);
	if (!camera.isOpened()) {
		cout << "Could not find camera" << endl;
		return -1;
	}

	Mat frame, winner;
	String choice = "";
	while (true)
	{
		// Store the feed from the camera into frame
		camera >> frame;

		if (frame.empty()) {
			cout << "Could not find frame" << endl;
			return -1;
		}
		
		winner = frame.clone();

		choice = detectSign(frame);

		// Show necessary information on the screen
		putText(frame, choice, Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(125, 255, 0), 3);
		putText(frame, "ESC to exit", Point(450, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 125, 0), 3);
		putText(frame, "Space to lock in your sign", Point(20, 420), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 125, 0), 3);

		// Detect key presses - Escape to exit, space to "throw"
		char key = waitKey(30);
		if (key == 27) break;
		if (key % 256 == 32) {
			int CPUchoice = rand() % 3;
			putText(winner, "CPU chose: " + getSign(CPUchoice), Point(frame.cols / 4, frame.rows / 5 * 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
			putText(winner, getWinner(playerSign, CPUchoice), Point(frame.cols / 4, frame.rows / 5 * 3), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
			putText(winner, "Space to close", Point(20, 420), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 125, 0), 3);
			imshow("Winner", winner);
			waitKey(0);
			destroyWindow("Winner");
		}

		imshow("Feed", frame);
	}

	return 0;
}