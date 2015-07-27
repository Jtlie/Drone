#include "ardrone/ardrone.h"

// --------------------------------------------------------------------------
// main(Number of arguments, Argument values)
// Description  : This is the entry point of the program.
// Return value : SUCCESS:0  ERROR:-1
// --------------------------------------------------------------------------
int main(int argc, char *argv[])
{
	// AR.Drone class
	ARDrone ardrone;

	// Initialize
	if (!ardrone.open()) {
		std::cout << "Failed to initialize." << std::endl;
		return -1;
	}

	// Battery
	std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]" << std::endl;

	// Map
	cv::Mat map = cv::Mat::zeros(500, 500, CV_8UC3);

	// Kalman filter
	cv::KalmanFilter kalman(6, 4, 0);

	// Sampling time [s]
	const double dt = 0.033;

	// Transition matrix (x, y, z, vx, vy, vz)
	cv::Mat1f F(6, 6);
	F << 1.0, 0.0, 0.0, dt, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0, dt, 0.0,
		0.0, 0.0, 1.0, 0.0, 0.0, dt,
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
	kalman.transitionMatrix = F;

	// Measurement matrix (0, 0, z, vx, vy, vz)
	cv::Mat1f H(4, 6);
	H << 0, 0, 1, 0, 0, 0,
		0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 1;
	kalman.measurementMatrix = H;

	// Process noise covariance (x, y, z, vx, vy, vz)
	cv::Mat1f Q(6, 6);
	Q << 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.3, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.3, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.3;
	kalman.processNoiseCov = Q;

	// Measurement noise covariance (z, vx, vy, vz)
	cv::Mat1f R(4, 4);
	R << 0.1, 0.0, 0.00, 0.00,
		0.0, 0.1, 0.00, 0.00,
		0.0, 0.0, 0.05, 0.00,
		0.0, 0.0, 0.00, 0.05;
	kalman.measurementNoiseCov = R;

	// Main loop
	while (1) {
		// Key input
		int key = cv::waitKey(33);
		if (key == 0x1b) break;

		// Get an image
		cv::Mat image = ardrone.getImage();

		// Prediction
		cv::Mat prediction = kalman.predict();

		// Altitude
		double altitude = ardrone.getAltitude();

		// Orientations
		double roll = ardrone.getRoll();
		double pitch = ardrone.getPitch();
		double yaw = ardrone.getYaw();

		// Velocities
		double vx, vy, vz;
		double velocity = ardrone.getVelocity(&vx, &vy, &vz);
		cv::Mat V = (cv::Mat1f(3, 1) << vx, vy, vz);

		// Rotation matrices
		cv::Mat RZ = (cv::Mat1f(3, 3) << cos(yaw), -sin(yaw), 0.0,
			sin(yaw), cos(yaw), 0.0,
			0.0, 0.0, 1.0);
		cv::Mat RY = (cv::Mat1f(3, 3) << cos(pitch), 0.0, sin(pitch),
			0.0, 1.0, 0.0,
			-sin(pitch), 0.0, cos(pitch));
		cv::Mat RX = (cv::Mat1f(3, 3) << 1.0, 0.0, 0.0,
			0.0, cos(roll), -sin(roll),
			0.0, sin(roll), cos(roll));

		// Time [s]
		static int64 last = cv::getTickCount();
		double dt = (cv::getTickCount() - last) / cv::getTickFrequency();
		last = cv::getTickCount();

		// Local movements (z, vx, vy, vz)
		cv::Mat1f M = RZ * RY * RX * V * dt;
		cv::Mat measurement = (cv::Mat1f(4, 1) << altitude, M(0, 0), M(1, 0), M(2, 0));

		// Correction
		cv::Mat1f estimated = kalman.correct(measurement);

		// Position (x, y, z)
		double pos[3] = { estimated(0, 0), estimated(1, 0), estimated(2, 0) };
		std::cout << "x = " << pos[0] << "[m], " << "y = " << pos[1] << "[m], " << "z = " << pos[2] << "[m]" << std::endl;

		// Take off / Landing 
		if (key == ' ') {
			if (ardrone.onGround()) ardrone.takeoff();
			else                    ardrone.landing();
		}

		// Move
		double x = 0.0, y = 0.0, z = 0.0, r = 0.0;
		if (key == 'i' || key == CV_VK_UP)    vx = 1.0;
		if (key == 'k' || key == CV_VK_DOWN)  vx = -1.0;
		if (key == 'u' || key == CV_VK_LEFT)  vr = 1.0;
		if (key == 'o' || key == CV_VK_RIGHT) vr = -1.0;
		if (key == 'j') vy = 1.0;
		if (key == 'l') vy = -1.0;
		if (key == 'q') vz = 1.0;
		if (key == 'a') vz = -1.0;
		ardrone.move3D(x, y, z, r);

		// Change camera
		static int mode = 0;
		if (key == 'c') ardrone.setCamera(++mode % 4);

		// Display the image
		cv::circle(map, cv::Point(-pos[1] * 100.0 + map.cols / 2, -pos[0] * 100.0 + map.rows / 2), 2, CV_RGB(255, 0, 0));
		cv::imshow("map", map);
		cv::imshow("camera", image);
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