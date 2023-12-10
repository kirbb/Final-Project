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

String detectSign(const Mat& frame) {
	Mat grayscale;
	cvtColor(frame, grayscale, COLOR_BGR2GRAY);

	const int scale = 2;
	Mat scaled(cvRound(grayscale.rows / scale), cvRound(grayscale.cols / scale), CV_8UC1);
	resize(grayscale, scaled, scaled.size());

	vector<Rect> rockDetections;
	vector<Rect> paperDetections;
	vector<Rect> scissorsDetections;
	rock_cascade.detectMultiScale(scaled, rockDetections);
	paper_cascade.detectMultiScale(scaled, paperDetections);
	scissors_cascade.detectMultiScale(scaled, scissorsDetections);

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

	rockDetections.clear();
	paperDetections.clear();
	scissorsDetections.clear();

	return playerChoice;
}

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

int main(int argc, char* argv[]) {
	
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
		camera >> frame;

		if (frame.empty()) {
			cout << "Could not find frame" << endl;
			return -1;
		}
		
		winner = frame.clone();

		choice = detectSign(frame);

		putText(frame, choice, Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(125, 255, 0), 3);
		putText(frame, "ESC to exit", Point(450, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 125, 0), 3);
		putText(frame, "Space to lock in your sign", Point(20, 420), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 125, 0), 3);

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