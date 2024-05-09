#include "stitcher.h"

int main()
{
	TickMeter time;
	time.start();
	/*
	class declaration and parameter setting
	*/
	ImgStitcher stitch;
	//stitch.bUseCDVS = false;
	stitch.bDisplayMatchingFlag = false;
	stitch.fMatchingConfid = 1.0f / 2.0f;
	stitch.maxInputSize = Size(1000, 1000);

	/*
	load images from folder
	*/
	cout << "-------------------------------------" << endl;
	cout << "Image stitching starts..." << endl;
	cout << "-------------------------------------" << endl;

	cout << "Loading images..." << endl;

	vector<Mat>vImgs;
	string strPath = "img//";
	vImgs = stitch.loadImgFromFolder(strPath);

	cout << "Image loading completed" << endl;
	time.stop();

	double dTempTime = 0.0;
	dTempTime += time.getTimeSec();
	cout << "Time cost: " << dTempTime << " sec" << endl << endl;

	/*
	stitch up-down images as a wide panorama view
	*/
	time.reset();
	time.start();
	/*vector<Mat>vInputTemp(2);
	vector<Mat>vPanorama;
	for (int i = 0; i < (int)vImgs.size(); i = i + 2)
	{
		vInputTemp[0] = vImgs[i];
		vInputTemp[1] = vImgs[i + 1];

		Mat mPanorama;
		mPanorama = stitch.multipleStitch(vInputTemp);
		vPanorama.push_back(mPanorama);

		stringstream ss;
		string str;
		ss << i;
		ss >> str;

		string newname = str + ".jpg";
		imwrite(newname, mPanorama);
	}*/

	vector<Mat>vInputTemp(4);
	vector<Mat>vPanorama;

	Mat mPanorama;
	mPanorama = stitch.multipleStitch(vImgs);
	vPanorama.push_back(mPanorama);

	imwrite("save_image.jpg", vPanorama);


	cout << "Image stitching finished" << endl;
	time.stop();
	dTempTime += time.getTimeSec();
	cout << "Total time cost: " << dTempTime << " sec" << endl;
	cout << "-------------------------------------" << endl << endl;

	return 0;
}

