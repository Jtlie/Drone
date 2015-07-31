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
	//// AR.Drone class
	ARDrone ardrone;

	//// Initialize
	//if (!ardrone.open()) {
	//	cout << "Failed to initialize." << endl;
	//	return -1;
	//}

	// Thresholds
	int minH, maxH;
	int minS, maxS;
	int minV, maxV;


	// XML save data
	std::string filename("thresholds.xml");
	cv::FileStorage fs(filename, cv::FileStorage::READ);

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
	cv::createTrackbar("H max", "binalized", &maxH, 255);
	cv::createTrackbar("H min", "binalized", &minH, 255);
	cv::createTrackbar("S max", "binalized", &maxS, 255);
	cv::createTrackbar("S min", "binalized", &minS, 255);
	cv::createTrackbar("V max", "binalized", &maxV, 255);
	cv::createTrackbar("V min", "binalized", &minV, 255);
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
		//drone //hog.detectMultiScale(image, found, 0, Size(4, 4), Size(0, 0), 1.5, 2.0);
		//camera 
		hog.detectMultiScale(image, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		
		
		// Show bounding rect
		vector<Rect>::const_iterator it;
		Vec3b intensity;
		vector<int> sizes;
		for (it = found.begin(); it != found.end(); ++it) {
			try{
				Rect r = *it;
				rectangle(hsv, r.tl(), r.br(), Scalar(0, 255, 0), 2);

				
				//rectangle(image, Point(midX,midY), Point(midX,midY),Scalar(255, 0, 0), 1)
				
				
				if (key == 'a'){
					
					int size = (r.br().x - r.tl().x) * (r.br().y - r.tl().y);
					
					sizes.insert(sizes.end(), size);
					int highest = sizes[0];
					for (int l = 1; l < sizes.size(); l++){
						if (highest < sizes[l]){
							highest = sizes[l];
						}
					}

					for (int p = 0; p < sizes.size(); p++)
					{
						if ((it[p].br().x - it[p].tl().x) * (it[p].br().y - it[p].tl().y) == highest){
							int midX;
							midX = (it[p].tl().x + it[p].br().x) / 2;
							int midY;
							midY = (it[p].tl().y + it[p].br().y) / 2;
							Rect r = it[p];
							rectangle(hsv, r.tl(), r.br(), Scalar(0, 0, 255), 2);
							cout << midX << "\n";
							cout << midY << "\n";
							intensity = hsv.at<Vec3b>(floor(midX), floor(midY) - 85);
							cout << floor(midY) << "\n";
							cout << intensity;
							
							rectangle(image, Point(floor(midX),floor(midY - 85)), Point(floor(midX),floor(midY - 85)),Scalar(255, 0, 0), 3);
							
							cout << intensity << "\n";
							minH = intensity[0] - 35;
							minS = intensity[1] - 35;
							minV = intensity[2] - 35;
							maxH = intensity[0] + 35;
							maxS = intensity[1] + 35;
							maxV = intensity[2] + 35;
						}
					}

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
		imshow("HSV", hsv);

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

	// See you
	ardrone.close();

	return 0;
}
