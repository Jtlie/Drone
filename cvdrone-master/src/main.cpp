//Copyright

#include "ardrone/ardrone.h"

using namespace cv;
using namespace std;

KalmanFilter kalman_filter();
Mat binalize(int minH, int minS, int minV, int maxH, int maxS, int maxV, Mat hsv, vector<Point> rectanglePos);
int largestContour(int minH, int minS, int minV, int maxH, int maxS, int maxV, Mat hsv, vector<Point> rectanglePos, vector<vector<Point>> contours);
void objectDetect(vector<vector<Point>> contours, int contour_index, KalmanFilter kalman, Mat image);
Point showResult(KalmanFilter kalman, Mat image);


// AR.Drone class
ARDrone ardrone;

int main(int argc, char *argv[])
{
	ARDrone ardrone;
	vector<Point> rectanglePos;
	Vec3b intensity;
	Point result;
	double vx = 0.0, vy = 0.0, vz = 0.0, vr = 0.0;
	int step = 1;
	boolean target = false;
	

	// Initialize
	if (!ardrone.open()) {
		cout << "Failed to initialize." << endl;
	return -1;
	}

	// Thresholds
	int minH, maxH;
	int minS, maxS;
	int minV, maxV;


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

	// Initialize Kalman Filter
	KalmanFilter kalman = kalman_filter();

	// Initialize detector
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//target found
	std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]" << std::endl;
	cout << "Press space to select a target";
	// Main loop
	while (1) {
		// Key input
		int key = waitKey(33);
		if (key == 0x1b) break;

		// Get an image
		Mat image = ardrone.getImage();

		// HSV image
		Mat hsv;
		cvtColor(image, hsv, CV_BGR2HSV_FULL);

		// Detect
		vector<Rect> found;

		//drone 
		hog.detectMultiScale(image, found, 0, Size(4, 4), Size(0, 0), 1.5, 4);
		
		//reset
		if (key == 'r' || key == 'R'){
			ardrone.landing();
			target = false;
			step = 1;
			cout << "Press space to select a target"<< '\n';
		}


		// Take off / Landing 
		if (key == ' ' && target == true && step == 3) {
			if (ardrone.onGround()) ardrone.takeoff(); 
			else                    ardrone.landing();
		}

		// Show bounding rect
		vector<Rect>::const_iterator it;
		vector<int> sizes;
		
		
		for (it = found.begin(); it != found.end(); ++it) {

			Rect r = *it;
			rectangle(image, r.tl(), r.br(), Scalar(0, 255, 0), 2);

			int size = (r.br().x - r.tl().x) * (r.br().y - r.tl().y);
			sizes.insert(sizes.end(), size);
			int biggest = sizes[0];
			for (int l = 1; l < sizes.size(); l++){
				if (biggest < sizes[l]){
					biggest = sizes[l];
				}
			}
			
			for (int p = 0; p < sizes.size(); p++)
			{
				//Take Color and Rectangle of the HOG
				if ((it[p].br().x - it[p].tl().x) * (it[p].br().y - it[p].tl().y) == biggest){
					int midX;
					midX = (it[p].tl().x + it[p].br().x) / 2;
					int midY;
					midY = (it[p].tl().y + it[p].br().y) / 2;
					Rect r = it[p];
					rectangle(image, r.tl(), r.br(), Scalar(0, 0, 255), 2);

					rectanglePos.clear();
					rectanglePos.insert(rectanglePos.end(), it[p].tl());
					rectanglePos.insert(rectanglePos.end(), it[p].br());

					intensity = hsv.at<Vec3b>(floor(midY - 30), floor(midX));
					rectangle(hsv, Point(floor(midX), floor(midY - 30)), Point(floor(midX), floor(midY - 30)), Scalar(255, 0, 0), 3);
					
				}
				
			}
			
			

			//Set Target
			


			if (key == ' ' && step == 1){
				step = 2;
				cout << "Are you satisfied with the result? [y/n]" << '\n';
				minH = intensity[0] - 20;
				minS = intensity[1] - 20;
				minV = intensity[2] - 20;
				maxH = intensity[0] + 20;
				maxS = intensity[1] + 20;
				maxV = intensity[2] + 20;
					
			}
			if (key == 'y' && step == 2 || key == 'Y' && step == 2){
				step = 3;
				cout << "Put the drone on the right position and press 'space' to start tracking" << '\n';
				target = true;
			}
			if (key == 'n' && step == 2 || key == 'N' && step == 2){
				cout << "Press space to select a target" << '\n';
				step = 1;
			}

		}
	
		// Detect contours		
		vector<vector<Point>> contours;
		cv::findContours(binalize(minH, minS, minV, maxH, maxS, maxV, hsv, rectanglePos).clone(), contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

		// Find the largest contour
		int contour_index = largestContour(minH, minS, minV, maxH, maxS, maxV, hsv, rectanglePos, contours);

		// Object detected
		if (contour_index >= 0) {
			objectDetect(contours, contour_index, kalman, image);
		}

		// Display the image
		result = showResult(kalman, image);

	

		if (target){
			
			
			rectangle(image, Point(300, 160), Point(340, 200), Scalar(0, 0, 255), 2);
			int oppervlakte;
			if (result.x >= rectanglePos[0].x && result.x <= rectanglePos[1].x && result.y >= rectanglePos[0].y && result.y <= rectanglePos[1].y)
			{
				int breedte = rectanglePos[1].x - rectanglePos[0].x;
				int lengte = rectanglePos[1].y - rectanglePos[0].y;
				oppervlakte = breedte * lengte; 
				
							
				if (key == 'k' || key == CV_VK_DOWN)  vx = -1.0;

				double horizontaldifference;
				double verticaldifference;

				//Turn Left
				if (result.x <= 300)
				{
					horizontaldifference = 300 - result.x;
					vr = horizontaldifference / 500;
					
				}
				//Turn Right
				else if (result.x >= 340)
				{
					horizontaldifference = result.x - 340;
					vr = horizontaldifference / 500 * -1;
					
				}
				//Stay Still
				else if (result.x > 300 && result.x < 340){
					vr = 0.0;
				}

				//forward
				if (oppervlakte == 41472)    vx = 0.1;
				else if (oppervlakte == 18432) vx = 0.4;
				else if (oppervlakte == 8192) vx = 0.6;

				//Up
				if (result.y <= 160)
				{
					verticaldifference = 160 - result.y;
					vz = verticaldifference / 320;
					
				}
				//Down
				else if (result.y >= 200)
				{
					horizontaldifference = result.x - 200;
					vz = horizontaldifference / 320 * -1;
				}

				//Stay Still
				else if (result.y > 160 && result.y < 200){
					vz = 0.0;
				}				
				ardrone.move3D(vx, vy, vz, vr);
				
			}
			else
			{
				vx = -0.1, vy = 0.0, vz = 0.0, vr = 0.0;
				ardrone.move3D(vx, vy, vz, vr);
				
			}
			rectanglePos.clear();
		}
		
		
	}

	// See you
	ardrone.close();
	return 0;
}

KalmanFilter kalman_filter() {
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

	return kalman;
}

Mat binalize(int minH, int minS, int minV, int maxH, int maxS, int maxV, Mat hsv, vector<Point> rectanglePos){
	// Binalize		
	Mat binalized;
	Scalar lower(minH, minS, minV);
	Scalar upper(maxH, maxS, maxV);
	inRange(hsv, lower, upper, binalized);

	//Make everything black except in the HOG
	if (rectanglePos.empty() == false){
			rectangle(binalized, Point(1, 1), Point(rectanglePos[0].x, 360), Scalar(0, 0, 0), -1);
			rectangle(binalized, Point(rectanglePos[1].x, 1), Point(640, 360), Scalar(0, 0, 0), -1);
			rectangle(binalized, Point(rectanglePos[0].x, 1), Point(rectanglePos[1].x, rectanglePos[0].y), Scalar(0, 0, 0), -1);
			rectangle(binalized, Point(rectanglePos[0].x, rectanglePos[1].y), Point(rectanglePos[1].x, 360), Scalar(0, 0, 0), -1);
	}
	

	// De-noising
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(binalized, binalized, MORPH_CLOSE, kernel);
	imshow("morphologyEx", binalized);

	return binalized;
}


int largestContour(int minH, int minS, int minV, int maxH, int maxS, int maxV, Mat hsv, vector<Point> rectanglePos, vector<vector<Point>> contours){
	cv::findContours(binalize(minH, minS, minV, maxH, maxS, maxV, hsv, rectanglePos).clone(), contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	
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
	return contour_index;
}

void objectDetect(vector<vector<Point>> contours, int contour_index, KalmanFilter kalman, Mat image){
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
	rectangle(image, rect, Scalar(255, 0, 0));
}

Point showResult(KalmanFilter kalman, Mat image){
	// Prediction
	Mat1f prediction = kalman.predict();
	int radius = 1e+3 * kalman.errorCovPre.at<float>(0, 0);

	// Show predicted position
	circle(image, Point(prediction(0, 0), prediction(0, 1)), radius, Scalar(0, 255, 0), 2);

	// Display the image
	imshow("camera", image);
	Point cirkelPos = Point(prediction(0, 0), prediction(0, 1));
	return cirkelPos;
}

