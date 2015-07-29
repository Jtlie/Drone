#include "ardrone/ardrone.h"

// --------------------------------------------------------------------------
// main(Number of arguments, Argument values)
// Description  : This is the entry point of the program.
// Return value : SUCCESS:0  ERROR:-1
// --------------------------------------------------------------------------

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	// AR.Drone class
	ARDrone ardrone;

	// Initialize
	/*if (!ardrone.open()) {
		cout << "Failed to initialize." << endl;
		return -1;
	}*/

	// Thresholds
	int minH = 0, maxH = 255;
	int minS = 0, maxS = 255;
	int minV = 0, maxV = 255;

	// XML save data
	string filename("thresholds.xml");
	FileStorage fs(filename, FileStorage::READ);

	// If there is a save file then read it
	if (fs.isOpened()) {
		maxH = fs["H_MAX"];
		minH = fs["H_MIN"];
		maxS = fs["S_MAX"];
		minS = fs["S_MIN"];
		maxV = fs["V_MAX"];
		minV = fs["V_MIN"];
		fs.release();
	}

	// Create a window
	namedWindow("binalized");
	createTrackbar("H max", "binalized", &maxH, 255);
	createTrackbar("H min", "binalized", &minH, 255);
	createTrackbar("S max", "binalized", &maxS, 255);
	createTrackbar("S min", "binalized", &minS, 255);
	createTrackbar("V max", "binalized", &maxV, 255);
	createTrackbar("V min", "binalized", &minV, 255);
	resizeWindow("binalized", 0, 0);

	// Kalman filter
	KalmanFilter kalman(4, 2, 0);

	// Sampling time [s]
	const double dt = 1.0;

	// Transition matrix (x, y, vx, vy)
	Mat1f A(4, 4);
	A << 1.0, 0.0, dt, 0.0,
		0.0, 1.0, 0.0, dt,
		0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 1.0;
	kalman.transitionMatrix = A;

	// Measurement matrix (x, y)
	Mat1f H(2, 4);
	H << 1, 0, 0, 0,
		0, 1, 0, 0;
	kalman.measurementMatrix = H;

	// Process noise covariance (x, y, vx, vy)
	Mat1f Q(4, 4);
	Q << 1e-5, 0.0, 0.0, 0.0,
		0.0, 1e-5, 0.0, 0.0,
		0.0, 0.0, 1e-5, 0.0,
		0.0, 0.0, 0.0, 1e-5;
	kalman.processNoiseCov = Q;

	// Measurement noise covariance (x, y)
	Mat1f R(2, 2);
	R << 1e-1, 0.0,
		0.0, 1e-1;
	kalman.measurementNoiseCov = R;

	// Initialize detector
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	VideoCapture cap(CV_CAP_ANY);

	// Main loop
	while (1) {
		// Key input
		int key = waitKey(33);
		if (key == 0x1b) break;

		// Get an image
		//Mat image = ardrone.getImage();

		Mat image;
		cap >> image;

		// HSV image
		Mat hsv;
		cvtColor(image, hsv, COLOR_BGR2HSV_FULL);

		// Detect
		vector<Rect> found;
		hog.detectMultiScale(image, found, 0, Size(4, 4), Size(0, 0), 1.5, 2.0);

		
		// Show bounding rect
		vector<Rect>::const_iterator it;
		for (it = found.begin(); it != found.end(); ++it) {
			try{
				Rect r = *it;
				rectangle(image, r.tl(), r.br(), Scalar(0, 255, 0), 2);

				int midX = (r.tl().x + r.br().x) / 2;
				int midY = (r.tl().y + r.br().y) / 2;
				//rectangle(image, Point(midX,midY), Point(midX,midY),Scalar(255, 0, 0), 1);

				

				if (found.size() == 1){
					//if (key == 'a'){
						Scalar intensity = image.at<uchar>(floor(midX), floor(midY));
						std::cout << intensity << "\n";
					//}
				}
			}

			catch (Exception e){
				std::cout << "error";
			}
		}
		// Binalize
		Mat binalized;
		Scalar lower(minH, minS, minV);
		Scalar upper(maxH, maxS, maxV);
		inRange(hsv, lower, upper, binalized);

		// Show result
		imshow("binalized", binalized);

		// De-noising
		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		morphologyEx(binalized, binalized, MORPH_CLOSE, kernel);
		//imshow("morphologyEx", binalized);

		// Detect contours
		vector<vector<Point>> contours;
		findContours(binalized.clone(), contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

		// Find the largest contour
		int contour_index = -1;
		double max_area = 0.0;
		for (size_t i = 0; i < contours.size(); i++) {
			double area = fabs(contourArea(contours[i]));
			if (area > max_area) {
				contour_index = i;
				max_area = area;
			}
		}

		// Object detected
		if (contour_index >= 0) {
			// Moments
			Moments moments = cv::moments(contours[contour_index], true);
			double marker_y = (int)(moments.m01 / moments.m00);
			double marker_x = (int)(moments.m10 / moments.m00);

			// Measurements
			Mat measurement = (Mat1f(2, 1) << marker_x, marker_y);

			// Correction
			Mat estimated = kalman.correct(measurement);

			// Show result
			Rect rect = boundingRect(contours[contour_index]);
			rectangle(image, rect, Scalar(0, 255, 0));
		}

		// Prediction
		Mat1f prediction = kalman.predict();
		int radius = 1e+3 * kalman.errorCovPre.at<float>(0, 0);

		// Show predicted position
		circle(image, Point(prediction(0, 0), prediction(0, 1)), radius, Scalar(0, 255, 0), 2);

		// Display the image
		imshow("camera", image);
	}

	// Save thresholds
	fs.open(filename, FileStorage::WRITE);
	if (fs.isOpened()) {
		write(fs, "H_MAX", maxH);
		write(fs, "H_MIN", minH);
		write(fs, "S_MAX", maxS);
		write(fs, "S_MIN", minS);
		write(fs, "V_MAX", maxV);
		write(fs, "V_MIN", minV);
		fs.release();
	}

	// See you
	ardrone.close();

	return 0;
}

//#include "kschluff\filter.h"
//#include "kschluff\lbp.h"
//#include "kschluff\selector.h"
//#include "kschluff\hist.h"
//#include "kschluff\state.h"
//#include "kschluff\unistd.h"
//#include "kschluff\getopt.h"
//#include <cmath>
//#include "ardrone/ardrone.h"
//
//
//#include <cstdlib>
//#include <string>
//
//using namespace cv;
//using namespace std;
//
//typedef unsigned int uint;
//
//const char* WINDOW = "Particle Tracker";
//
//const Scalar RED = Scalar(0, 0, 255);
//const Scalar BLUE = Scalar(255, 0, 0);
//const Scalar GREEN = Scalar(0, 255, 0);
//const Scalar CYAN = Scalar(255, 255, 0);
//const Scalar MAGENTA = Scalar(255, 0, 255);
//const Scalar YELLOW = Scalar(0, 255, 255);
//const Scalar WHITE = Scalar(255, 255, 255);
//const Scalar BLACK = Scalar(0, 0, 0);
//
//const uint NUM_PARTICLES = 200;
//
//inline void update_target_histogram(Mat& image, Mat& lbp_image, Rect& selection, Mat& histogram, Mat& target, bool use_lbp)
//{
//	Mat roi(image, selection), lbp_roi(lbp_image, selection);
//	roi.copyTo(target);
//	Mat new_hist;
//	float alpha = 0.2;
//
//	calc_hist(roi, lbp_roi, new_hist, use_lbp);
//	normalize(new_hist, new_hist);
//
//	if (histogram.empty())
//	{
//		histogram = new_hist;
//	}
//	else
//	{
//		// TODO - support for adaptive updates not fully implemented.
//		histogram = ((1.f - alpha) * histogram) + (alpha * new_hist);
//		normalize(histogram, histogram);
//	}
//	cout << "Target updated" << endl;
//}
//
//struct StateData
//{
//	StateData(int num_particles, bool use_lbp_) :
//		image(),
//		lbp(),
//		target(),
//		target_histogram(),
//		selector(WINDOW),
//		selection(),
//		use_lbp(use_lbp_),
//		paused(false),
//		draw_particles(false),
//		filter(num_particles)
//	{};
//
//	Mat image;
//	Mat lbp;
//	Mat target;
//	Mat target_histogram;
//	Selector selector;
//	Rect selection;
//	bool use_lbp;
//	bool paused;
//	bool draw_particles;
//	ParticleFilter filter;
//};
//
//State_ state_start(StateData& d)
//{
//
//	if (d.selector.selecting())
//	{
//		cout << "state_selecting" << endl;
//		return state_selecting;
//	}
//	else
//	{
//		return state_start;
//	}
//}
//
//State_ state_selecting(StateData& d)
//{
//	if (d.selector.valid())
//	{
//		cout << "state_initializing: (" << d.selection.x << ", " << d.selection.y << ", " << d.selection.width << ", " << d.selection.height << ")" << endl;
//		d.selection = d.selector.selection();
//		cout << "selection: (" << d.selection.x << ", " << d.selection.y << ", " << d.selection.width << ", " << d.selection.height << ")" << endl;
//		return state_initializing(d); // Call with current frame
//	}
//	else
//	{
//		Mat roi(d.image, d.selector.selection());
//		bitwise_not(roi, roi);
//		return state_selecting;
//	}
//}
//
//State_ state_initializing(StateData& d)
//{
//	if (d.selector.selecting())
//	{
//		cout << "state_selecting" << endl;
//		return state_selecting;
//	}
//
//	// Generate initial target histogram
//	update_target_histogram(d.image, d.lbp, d.selection, d.target_histogram, d.target, d.use_lbp);
//
//	// Initialize condensation filter with center of selection
//	d.filter.init(d.selection);
//
//	// Start video running if paused
//	d.paused = false;
//
//	cout << "state_tracking" << endl;
//	return state_tracking(d); // Call with current frame
//}
//
//State_ state_tracking(StateData& d)
//{
//	if (d.selector.selecting())
//	{
//		cout << "state_selecting" << endl;
//		return state_selecting;
//	}
//
//	// Update particle filter
//	d.filter.update(d.image, d.lbp, d.selection.size(), d.target_histogram, d.use_lbp);
//
//	Size target_size(d.target.cols, d.target.rows);
//
//	// Draw particles
//	if (d.draw_particles)
//		d.filter.draw_particles(d.image, target_size, WHITE);
//
//	// Draw estimated state with color based on confidence
//	float confidence = d.filter.confidence();
//
//	// TODO - Make these values not arbitrary
//	if (confidence > 0.1)
//	{
//		d.filter.draw_estimated_state(d.image, target_size, GREEN);
//	}
//	else if (confidence > 0.025)
//	{
//		d.filter.draw_estimated_state(d.image, target_size, YELLOW);
//	}
//	else
//	{
//		d.filter.draw_estimated_state(d.image, target_size, RED);
//	}
//
//	return state_tracking;
//}
//
//struct Options
//{
//	Options()
//		:num_particles(NUM_PARTICLES),
//		use_lbp(false),
//		infile(),
//		outfile()
//	{}
//
//	int num_particles;
//	bool use_lbp;
//	string infile;
//	string outfile;
//};
//
//void parse_command_line(int argc, char** argv, Options& o)
//{
//	int c = -1;
//
//
//
//
//	cout << "Num particles: " << o.num_particles << endl;
//	cout << "Input file: " << o.infile << endl;
//	cout << "Output file: " << o.outfile << endl;
//	cout << "Use LBP: " << o.use_lbp << endl;
//
//}
//
//int main(int argc, char** argv)
//{
//	Options o;
//	parse_command_line(argc, argv, o);
//
//	bool use_camera;
//	VideoCapture cap;
//	VideoWriter writer;
//
//	// Use filename if given, else use default camera
//	if (!o.infile.empty())
//	{
//		cap.open(o.infile);
//		use_camera = false;
//	}
//	else
//	{
//		cap.open(0);
//		use_camera = true;
//	}
//
//	if (!cap.isOpened())
//	{
//		cerr << "Failed to open capture device" << endl;
//		exit(2);
//	}
//
//	if (!o.outfile.empty())
//	{
//		int fps = cap.get(CV_CAP_PROP_FPS);
//		int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
//		int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
//		writer.open(o.outfile, CV_FOURCC('j', 'p', 'e', 'g'), fps, Size(width, height));
//		if (!writer.isOpened())
//		{
//			cerr << "Could not open '" << o.outfile << "'" << endl;
//			exit(1);
//		}
//		use_camera = false;
//	}
//
//	// Open window and start capture
//	namedWindow(WINDOW, CV_WINDOW_FREERATIO | CV_GUI_NORMAL);
//
//
//	StateData d(o.num_particles, o.use_lbp);
//	State state = state_start;
//	Mat frame, gray;
//
//	lbp_init();
//
//	// Main loop
//
//	for (;;)
//	{
//
//		// Start timing the loop
//
//
//		// Capture frame
//		if (!d.paused)
//		{
//			cap >> frame;
//			if (frame.empty())
//			{
//				cerr << "Error reading frame" << endl;
//				break;
//			}
//		}
//		if (use_camera)
//		{
//			flip(frame, d.image, 1);
//		}
//		else
//		{
//			frame.copyTo(d.image);
//		}
//
//		// Set up all the image formats we'll need
//		if (d.use_lbp)
//		{
//			cvtColor(d.image, gray, CV_BGR2GRAY);
//			lbp_from_gray(gray, d.lbp);
//		}
//		else
//		{
//			if (d.lbp.empty())
//				d.lbp = Mat::zeros(d.image.rows, d.image.cols, CV_8UC1);
//		}
//
//		// Handle keyboard input
//		char c = (char)waitKey(10);
//		if (c == 27)
//			break;
//		switch (c)
//		{
//		case 'p':
//			d.paused = !d.paused;
//			break;
//
//		case 'c':
//			cout << "Tracking cancelled." << endl;
//			state = state_start;
//			break;
//
//		case 'd':
//			d.draw_particles = !d.draw_particles;
//			cout << "Draw particles: " << d.draw_particles << endl;
//			break;
//		}
//
//		// Process frame in current state
//		state = state(d);
//
//
//		// Elapsed time in seconds
//		/*
//		timeval end_time;
//		gettimeofday(&end_time, 0);
//		float dt = (float)(end_time.tv_sec - start_time.tv_sec) + ((float)(end_time.tv_usec - start_time.tv_usec)) * 1E-6;
//		cout << "Frame rate: " << 1.f / dt << endl;
//		*/
//		Mat target_display_area(d.image, Rect(d.image.cols - d.selection.width, 0, d.selection.width, d.selection.height));
//		d.target.copyTo(target_display_area);
//
//
//		imshow(WINDOW, d.image);
//	}
//
//}