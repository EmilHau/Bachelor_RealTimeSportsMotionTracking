#include <iostream>
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

VideoCapture cap;

float scale = 0.7;

const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

bool nextFrame(Mat& frame, Mat& frameNextColor){
    if(!cap.read(frame)) return false;
    resize(frame, frame, Size(), scale, scale, INTER_LINEAR);
    frameNextColor = frame.clone();
    cvtColor(frame, frame, CV_BGR2GRAY);
    return true;
}

bool pause = false;

void mouse(int  event, int  x, int  y, int  flag, void *param){
    switch(event){
        case EVENT_LBUTTONDOWN:
            pause = true;
            break;
        case EVENT_LBUTTONUP:
            pause = false;
            break;
    }
}

int main(int, char**){
    cap = VideoCapture("final_FIFACut.mp4");
    if(!cap.isOpened()) return -1;

    namedWindow("diff",1);
    resizeWindow("diff", 1200,1600);
    setMouseCallback("diff", mouse);

    Mat framePrev;
    Mat frameNext;
    Mat frameNextColor;

    std::vector<Point2f> blobsBall;

    vector<vector<cv::KeyPoint> > keypointsAll;

    std::vector<Point2f> blobsPlayer;

    nextFrame(frameNext,frameNextColor);

    Mat avg = frameNext.clone();

    Mat blackImage = frameNext*0;

    //cvtColor(avg, avg, CV_BGR2GRAY);
    //resize(avg, avg, Size(), scale, scale, INTER_LINEAR);
    int frameCounter = 0;

    while(1){
        if(pause){
            waitKey(1);
            continue;
        }

        framePrev = frameNext.clone();
        if(!nextFrame(frameNext,frameNextColor)) break;
        frameCounter++;
        if (frameCounter == 1500)break;

        //avg = 0.99*avg+0.01*frameNext;
        avg = imread("background.png",CV_LOAD_IMAGE_UNCHANGED);
        imshow("avg",avg);

        Mat framePrevCopy = framePrev.clone();
        Mat frameNextCopy = frameNext.clone();

        Mat diff1;
        Mat diff2;
        Mat diff;
        Mat smallBlur1, smallBlur;

        // Generate frame of moving objects

        absdiff(avg,framePrev,diff1);
        smallBlur = diff1.clone();
        blur(diff1, diff2, Size(4,4));
        blur(diff1, smallBlur1, Size(4,4));
        diff2.convertTo(diff, -1, 3, 0);
        imshow("showThis",diff);
        smallBlur1.convertTo(smallBlur, -1, 3,0);

        // Use blob detection to find players in frame
        cv::SimpleBlobDetector::Params params;
        params.minThreshold = 50;
        params.maxThreshold = 255;
        params.thresholdStep = 10;
        params.minDistBetweenBlobs = 50.0f*scale;
        params.filterByInertia = false;
        params.filterByConvexity = false;
        params.filterByColor = false;
        params.filterByCircularity = false;
        params.filterByArea = true;
        params.minArea = 80.0f*scale;
        params.maxArea = 430.0f*scale;

        Ptr<SimpleBlobDetector> blob_detector = SimpleBlobDetector::create(params);

        // Find blobs and store them to track history
        keypointsAll.push_back(vector<cv::KeyPoint>());
        blob_detector->detect(diff, keypointsAll[keypointsAll.size()-1]);
        blobsPlayer.empty();
        // Check if newly found blobs, if they are not close to any other existing blobs
        for(int w = 0; w < keypointsAll[keypointsAll.size()-1].size(); ++w){
            Point2f center = Point2f(keypointsAll[keypointsAll.size()-1][w].pt.x, keypointsAll[keypointsAll.size()-1][w].pt.y);
            circle(frameNextColor, keypointsAll[keypointsAll.size()-1][w].pt,  40, Scalar(255,255,255));
            blobsPlayer.push_back(center);
        }

        // Find new circles with Hough Circle
        vector<Vec3f> circles;
        HoughCircles( smallBlur, circles, HOUGH_GRADIENT, 1, 5, 80, 9, 2, 4);
        for( size_t i = 0; i < circles.size(); i++ ) {
            Vec3i c = circles[i];
            Point2f center = Point2f(c[0], c[1]);
            // circle center
            bool validPoint = true;
            for(size_t w = 0; w < blobsPlayer.size(); ++w){
                if ( fabs((double)blobsPlayer[w].x - (double)center.x) < 40 && fabs((double)blobsPlayer[w].y - (double)center.y) <  40  ) {
                    validPoint = false;
                }
            }

            // circle outline
            if(validPoint){
                int radius = c[2];
                circle( frameNextColor, center, radius*1.5, SCALAR_BLUE, 1, LINE_AA);
                circle( blackImage, center, radius*1.5, SCALAR_BLUE, 1, LINE_AA);
            }
        }
        imshow("frame",frameNextColor);
        waitKey(1);
    }
    imwrite("result.png",blackImage);
    return 0;
}
