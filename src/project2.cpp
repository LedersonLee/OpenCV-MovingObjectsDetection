#include <iostream>
#include "cv.hpp"

using namespace std;
using namespace cv;

int main() {
	VideoCapture cap;
	int fps, delay;
	Mat frame, backgnd, frame_gray, foregndMask, kernel;
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	if (cap.open("../Computer_Vision_dataset2/Project2.mp4") == 0) {
		cout << "No such file" << endl;
		return -1;
	}

	fps = cap.get(CAP_PROP_FPS);
	delay = 1000 / fps;

	cap >> backgnd;
	resize(backgnd, backgnd, Size(1280, 720));
	cvtColor(backgnd, backgnd, CV_BGR2GRAY);

	while (1) {
		cap >> frame;
		if (frame.empty()) break;

		resize(frame, frame, Size(1280, 720));


		if (foregndMask.empty())
			foregndMask.create(frame.size(), frame.type());

		bg_model->apply(frame, foregndMask); //absdiff와 같은 효과 
		//imshow("foregnd", foregndMask);

		threshold(foregndMask, foregndMask, 230, 255, THRESH_BINARY);
		imshow("foregnd", foregndMask);

		//illumination & small object 움직임 무시하기
		kernel = getStructuringElement(MORPH_ELLIPSE, Size(23, 23));
		morphologyEx(foregndMask, foregndMask, MORPH_OPEN, kernel);
		imshow("gaus", foregndMask);

		//object 식별하기
		findContours(foregndMask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		if (contours.size())
			putText(frame, format("Alert! Moving object!"), Point(50, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 4);

		imshow("Project2", frame);
		waitKey(delay);
	}

	return 0;
}
