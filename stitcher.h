#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
using namespace std;
using namespace cv;

class ImgStitcher
{
public:
	ImgStitcher();
	~ImgStitcher();

	Mat multipleStitch(vector<Mat> vImg);
	vector<Mat> loadImgFromFolder(string strPath);
public:
	Size maxInputSize;
	float fMatchingConfid;

	bool bUseCDVS;
	bool bDisplayMatchingFlag;
private:
	int nOffset_x;
	int nOffset_y;
	int nReferImgIndex;
	bool bHmodifyFlag;

	vector<Mat>vH;
	vector<Mat>vWarpImg;
	vector<Point>vPos;

	TickMeter t;
	SiftDescriptorExtractor sift;
	FlannBasedMatcher matcher;
private:
	Size calOptimalWarpSize(Mat mOriCoord, Mat H, Point2d& pt);
	vector<DMatch> matchingFilter(vector<vector<DMatch>> vTotalMatches);
	

	void singlePlaneTransform(vector<Mat>vImgs);
	void extractSIFT(Mat mGray1, Mat mGray2, vector<KeyPoint> &vFeat1, vector<KeyPoint> &vFeat2, Mat &mDesc1, Mat &mDesc2);
	//void extractCDVS(Mat mGray1, Mat mGray2, vector<KeyPoint> &vFeat1, vector<KeyPoint> &vFeat2, Mat &mDesc1, Mat &mDesc2);

	Mat registration();
	Mat warping(Mat mImg2, Mat H, Point& pos);
	
	Mat calHomography(vector<KeyPoint> vFeatCurr, vector<KeyPoint> vFeatNext, vector<DMatch> vFinalMatches);
};

