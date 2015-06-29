#include<opencv/cvaux.h>
#include<opencv/highgui.h>
#include<opencv/cxcore.h>
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char* argv[]){
	CvSize size640x480 = cvSize(640, 480);

	CvCapture* p_capWebcam;

	IplImage* p_imgOriginal;
	IplImage* p_imgProcessed;

	CvMemStorage* p_strStorage;
	CvSeq* p_seqCircles;

	float* p_fltXYRadius;

	int i;
	char charCheckForEscKey;

	p_capWebcam = cvCaptureFromCAM(0);
	
	if (p_capWebcam == NULL){
		printf("Geen webcam aangesloten!");
		getchar();
		return(-1);
	}

	cvNamedWindow( "Original"
				  , CV_WINDOW_AUTOSIZE);
	cvNamedWindow( "Processed"
				  , CV_WINDOW_AUTOSIZE);

	p_imgProcessed = cvCreateImage(   size640x480
									, IPL_DEPTH_8U, 1);

	while (1){
		p_imgOriginal = cvQueryFrame(p_capWebcam);
		if (p_imgOriginal == NULL) {
			printf("Geen beeld gevonden!");
			getchar();
			break;
		}

		cvInRangeS(   
			p_imgOriginal
		  , CV_RGB(175, 0, 0)
		  , CV_RGB(256, 100, 100)
		  , p_imgProcessed);

		p_strStorage = cvCreateMemStorage(0);

		cvSmooth(p_imgProcessed
			, p_imgProcessed
			, CV_GAUSSIAN
			, 9
			, 9);

		p_seqCircles = cvHoughCircles(
			  p_imgProcessed
			, p_strStorage
			, CV_HOUGH_GRADIENT
			, 2
			, p_imgProcessed->height / 4
			, 100
			, 50
			, 10
			, 400);
		
		for (i = 0; i < p_seqCircles->total; i++){
			p_fltXYRadius = (float*)cvGetSeqElem(p_seqCircles, i);

			printf("ball positie x = %f, y = %f, r = %f \n", p_fltXYRadius[0]
														   , p_fltXYRadius[1]
														   , p_fltXYRadius[2]);
			cvCircle(
				  p_imgOriginal
				, cvPoint(cvRound(p_fltXYRadius[0])
				, cvRound(p_fltXYRadius[1]))
				, 3
				, CV_RGB(0, 255, 0)
				, CV_FILLED);

			cvCircle(
				p_imgOriginal
				, cvPoint(cvRound(p_fltXYRadius[0])
				, cvRound(p_fltXYRadius[1]))
				, cvRound(p_fltXYRadius[2])
				, CV_RGB(255, 0, 0)
				, 3);
			}
		cvShowImage("Original", p_imgOriginal);
		cvShowImage("Processed", p_imgProcessed);

		cvReleaseMemStorage(&p_strStorage);

		charCheckForEscKey = cvWaitKey(10);
		if (charCheckForEscKey == 27) break;
	}
	cvReleaseCapture(&p_capWebcam);
	cvDestroyWindow("Original");
	cvDestroyWindow("Processed");

	return(0);

}
