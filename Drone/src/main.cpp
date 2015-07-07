#include "ardrone/ardrone.h"

using namespace std;
using namespace cv;
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

	// Initialize detector
	cv::HOGDescriptor hog;
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

	// Main loop
	while (1) {
		// Key input
		int key = cv::waitKey(1);
		if (key == 0x1b) break;

		// Get an image
		cv::Mat image = ardrone.getImage();

		// Detect
		std::vector<cv::Rect> found, found_filtered;
		hog.detectMultiScale(image, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

		// Show bounding rect
		size_t i, j;
		for (i = 0; i<found.size(); i++)
		{
			Rect r = found[i];
			for (j = 0; j<found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}
		for (i = 0; i<found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.06);
			r.height = cvRound(r.height*0.9);
			rectangle(image, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
		}

		// Display the image
		cv::imshow("hog", image);
	}

	// See you
	ardrone.close();

	return 0;
}