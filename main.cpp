#include <iostream>
#include <vector>
#include <algorithm> // std::min_element
#include <iterator>  // std::begin, std::end

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int hmin = -1, smin = 0, hmax = 150, smax = 255, blockSizeu = 3; // HSV limits
int MAX_KERNEL_LENGTH = 33; // Blur kernal

void segBCM(const Mat& inputImage, Mat& outputMask); // Tongue segmentation

int main(int argc, char** argv)
{
    String filepath;
    if (argc == 2){
        filepath = argv[1];
    } 
    else{
        filepath = "im_4.bmp"; // Defualt image  
    }
	
    // Reads the image. 4 5 8
    Mat image = imread(filepath);
    if (!image.data){
        cout << "Image not found or empty" << endl;
        return 1;
    }
    
    Mat mask;
    segBCM(image, mask);

    Mat detected(image.size(), CV_8UC3, Scalar(0, 0, 0));
    image.copyTo(detected, mask);
    imwrite("Resources/imgs/im_output.bmp", detected);
    
    return 0;
}

void segBCM(const Mat& inputImage, Mat& outputMask)
{

    
    // Applys Gaussian smoothing on the original image
    cv::Mat imageBlur;
    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 4 )
    {
        cv::GaussianBlur(inputImage, imageBlur, cv::Size(i, i), 0, 0);
    }
    
    // Converts RGB to HSV space conversion.
    Mat hsvImg;
    cvtColor(imageBlur, hsvImg, cv::COLOR_BGR2HSV);


    // Splits the HSV image to get the brightness component.
    vector<Mat> components;
    split(hsvImg, components);

    
    // Gets the mean and the standard deviation of the V component/channel.
    Scalar meanImgV;
    Scalar stdDevImgV;
    meanStdDev(components[2], meanImgV, stdDevImgV);
    cout << "meanImg: " << meanImgV[0] << endl;
    cout << "stdDevImg: " << stdDevImgV[0] << endl;


    // Calculates the eps of the BCM
    double eps {0};
    double vLower = meanImgV[0] - stdDevImgV[0];
    double vMin {0};
    minMaxLoc(components[2], &vMin);

    if (vLower < (2 * stdDevImgV[0])) {
        eps = meanImgV[0];
    }
    else if (vLower > (2 * stdDevImgV[0])){
        eps = vMin;
    }
    else
        eps = meanImgV[0] * meanImgV[0];

    double vUpper = meanImgV[0] + (eps * (stdDevImgV[0]/255));
    

    cout << "vLower: " << vLower << endl;
    cout << "bcm: " << eps << endl;
    cout << "vMin: " << vMin << endl;
    cout << "vUpper: " << vUpper << endl;


    // Thresholds the image based on the BCM
    Mat mask1;
    Mat mask2;
    
    // Hue masking
    threshold(components[0], mask1, hmax, 256, THRESH_BINARY_INV);  // below maxHue
    threshold(components[0], mask2, hmin, 256, THRESH_BINARY);  // over minHue
    Mat hueMask;
    hueMask = ~(mask1 & mask2);

    // Saturation masking
    threshold(components[1], mask1, smax, 255, THRESH_BINARY_INV);  // below maxSat
    threshold(components[1], mask2, smin, 255, THRESH_BINARY);  // over minSat
    Mat satMask;
    satMask = mask1 & mask2;
    
    // V masking
    threshold(components[2], mask1, vLower, 255, THRESH_BINARY); // Select after the vLower
    threshold(components[2], mask2, vUpper, 255, THRESH_BINARY); // Select after the vUpper
    
    
    ///-----------------------------------------------------------------------------------------------------------------------------------
    /// Extra: Applying the Erode morphological operation and deleting the small areas
    
    cv::Mat box(1, 1, CV_8U, cv::Scalar(1));
    cv::morphologyEx(mask2, mask2, cv::MORPH_ERODE, box, Point(-1,-1), 3); // ERODE 3 times
    
    // Finds the contours from the maskAdapV
    std::vector < std::vector <cv::Point> > contoursMask2;
    std::vector < cv::Vec4i > hierarchyMask2;
    cv::Mat imageContoursMask2(inputImage.size(), CV_8UC1, cv::Scalar(0));
    cv::findContours(mask2, contoursMask2, hierarchyMask2, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    // Finds the largest area from the contours
    int idx = 0;
    int largestComp = 0;
    int maxArea = 0;
    for( ; idx >= 0; idx = hierarchyMask2[idx][0] )
    {
        const std::vector<cv::Point>& c = contoursMask2[idx];
        double area = fabs(contourArea(cv::Mat(c)));
        if( area > maxArea )
        {
            maxArea = area;
            largestComp = idx;
        }
    }
    
    cv::Scalar whiteColour(255, 255, 255);
    drawContours(imageContoursMask2, contoursMask2, largestComp, whiteColour, cv::FILLED, cv::LINE_8, hierarchyMask2);
    ///-----------------------------------------------------------------------------------------------------------------------------------
    
    outputMask = imageContoursMask2 & hueMask & satMask;
}
