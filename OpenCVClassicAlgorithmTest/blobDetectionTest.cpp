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

Mat framePrev;
Mat frameNext;
Mat frameNextColor;
Mat blackImage;

vector<vector<cv::KeyPoint> > keypointsAll;

std::vector<Point2f> blobsPlayer;


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
            cout << (int)frameNext.at<uchar>(x,y) << endl;
            // shift is pressed
            if(flag == 17){
                pause = false;
                break;
            }
            pause = true;

            break;
    }
}

int main(int, char**){
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 6; j++){
            int stepSize;
            if(i==0)stepSize = 1;
            else{stepSize = i * 5 ;}

            double inRatio = 0.5 + 0.1 * j;

            keypointsAll.clear();
            cout << keypointsAll.size() << endl;
            blobsPlayer.clear();
            cout << blobsPlayer.size() << endl;

            int frameCounter = 0;

            cap = VideoCapture("final_FIFACut.mp4");
            if(!cap.isOpened()) return -1;

            nextFrame(frameNext,frameNextColor);
            frameCounter++;

            Mat avg = frameNext.clone();

            while(1){
                if(pause){
                    waitKey(1);
                    continue;
                }
                if(frameCounter == 1500)break;

                framePrev = frameNext.clone();
                if(!nextFrame(frameNext,frameNextColor)) break;

                frameCounter++;

                //avg = 0.99*avg+0.01*frameNext;
                avg = imread("background.png",CV_LOAD_IMAGE_UNCHANGED);

                Mat framePrevCopy = framePrev.clone();
                Mat frameNextCopy = frameNext.clone();

                Mat diff1;
                Mat diff2;
                Mat diff;
                Mat smallBlur1, smallBlur;

                // Generate frame of moving objects

                absdiff(avg,framePrev,diff1);
                smallBlur = diff1.clone();
                blur(diff1, diff2, Size(3,3));
                blur(diff1, smallBlur1, Size(2,2));
                diff2.convertTo(diff, -1, 3, 0);
                smallBlur1.convertTo(smallBlur, -1, 5,0);
                blackImage = diff * 0;
                imshow("avg",diff);

                // Use blob detection to find players in frame
                cv::SimpleBlobDetector::Params params;
                params.minThreshold = 60;
                params.maxThreshold = 255;
                params.thresholdStep = stepSize;
                params.minDistBetweenBlobs = 70.0f*scale;
                params.filterByInertia = true;
                params.filterByConvexity = false;
                params.filterByColor = false;
                params.filterByCircularity = false;
                params.filterByArea = true;
                params.maxArea = 40.0f*scale;
                params.minInertiaRatio = inRatio;

                Ptr<SimpleBlobDetector> blob_detector = SimpleBlobDetector::create(params);

                // Find blobs and store them to track history
                keypointsAll.push_back(vector<cv::KeyPoint>());
                blob_detector->detect(diff, keypointsAll[keypointsAll.size()-1]);
                blobsPlayer.empty();
                // Check if newly found blobs, if they are not close to any other existing blobs
                for(int w = 0; w < keypointsAll[keypointsAll.size()-1].size(); ++w){
                    Point2f center = Point2f(keypointsAll[keypointsAll.size()-1][w].pt.x, keypointsAll[keypointsAll.size()-1][w].pt.y);
                    //circle(frameNextColor, keypointsAll[keypointsAll.size()-1][i].pt, keypointsAll[keypointsAll.size()-1][i].size, Scalar(0,255,255));
                    //circle(frameNextColor, keypointsAll[keypointsAll.size()-1][i].pt, 40, Scalar(255,255,255));
                    //circle(frameNextColor, keypointsAll[keypointsAll.size()-1][i].pt,  keypointsAll[keypointsAll.size()-1][i].size, Scalar(255,255,255));
                    blobsPlayer.push_back(center);
                }

                // Draw player blobs on screen
                for(size_t w = 0; w < blobsPlayer.size(); ++w){
                    circle(blackImage, Point2d( (double)blobsPlayer[w].x, (double)blobsPlayer[w].y  ), 5, Scalar(255,255,255));
                }

                //imshow("frame", show);

                // Find new circles with Hough Circle
                /*
                vector<Vec3f> circles;
                HoughCircles( smallBlur, circles, HOUGH_GRADIENT, 1, 5, 80, 9, 2, 4);
                for( size_t i = 0; i < circles.size(); i++ ) {
                    Vec3i c = circles[i];
                    Point2f center = Point2f(c[0], c[1]);
                    // circle center
                    //circle( frameNextColor, center, 1, SCALAR_BLUE, 1, LINE_AA);
                    bool validPoint = true;
                    for(size_t w = 0; w < blobsPlayer.size(); ++w){
                        if ( fabs((double)blobsPlayer[w].x - (double)center.x) < 25 && fabs((double)blobsPlayer[w].y - (double)center.y) <  25  ) {
                            validPoint = false;
                        }
                    }

                    // circle outline
                    if(validPoint){
                        int radius = c[2];
                        circle( frameNextColor, center, radius*1.5, SCALAR_BLUE, 1, LINE_AA);
                    }
                }*/
                imshow("frame",frameNextColor);
                waitKey(1);
            }
            string name = "testConf"+to_string(stepSize)+"_"+to_string(inRatio)+".png";
            imwrite(name,blackImage);

        }
    }
    return 0;
}
