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
    for(int i = 0; i < 10; ++i){
        for(int j = 5; j < 31; j++){
            double dpVal = 1 + i*0.1;
            int param2 = j;
            int frameCounter = 0;
            cap = VideoCapture("final_FIFACut.mp4");
            if(!cap.isOpened()) return -1;

            namedWindow("diff",1);
            resizeWindow("diff", 1200,1600);
            setMouseCallback("diff", mouse);

            nextFrame(frameNext,frameNextColor);
            frameCounter++;

            Mat avg = frameNext.clone();
            Mat blackImageGray = frameNext * 0;
            Mat blackImage;
            cvtColor(blackImageGray,blackImage,COLOR_GRAY2RGB);

            while(1){
                if(pause){
                    waitKey(1);
                    continue;
                }
                framePrev = frameNext.clone();
                if(!nextFrame(frameNext,frameNextColor)) break;
                if(frameCounter == 1500)break;
                frameCounter++;

                //avg = 0.99*avg+0.01*frameNext;
                avg = imread("background.png",CV_LOAD_IMAGE_UNCHANGED);
                imshow("avg",avg);

                Mat framePrevCopy = framePrev.clone();
                Mat frameNextCopy = frameNext.clone();

                Mat diff1;
                Mat diff2;
                Mat diff;

                // Generate frame of moving objects
                absdiff(avg,framePrev,diff1);
                blur(diff1, diff2, Size(5,5));
                diff2.convertTo(diff, -1, 4, 0);

                // Find new circles with Hough Circle
                vector<Vec3f> circles;
                HoughCircles( diff, circles, HOUGH_GRADIENT, dpVal, 5, 80, param2, 2, 4);
                for( size_t w = 0; w < circles.size(); w++ ) {
                    Vec3i c = circles[w];
                    Point2f center = Point2f(c[0], c[1]);
                    // circle center
                    //circle( frameNextColor, center, 1, SCALAR_BLUE, 1, LINE_AA);
                    // circle outline
                    int radius = c[2];
                    if(w == 0)circle( blackImage, center, radius*1.5, SCALAR_RED, 1, LINE_AA);
                    else{circle( blackImage, center, radius*1.5, SCALAR_WHITE, 1, LINE_AA);}
                }

                imshow("result",frameNextColor);
                waitKey(1);
            }

            string name = "testConf_"+to_string(dpVal)+"_"+to_string(param2)+".png";
            imwrite(name, blackImage);
        }
    }
    return 0;
}
