#include "stitcher.h"

ImgStitcher::ImgStitcher()
{
	bUseCDVS = false;
	bHmodifyFlag = false;
	bDisplayMatchingFlag = false;
	fMatchingConfid = 1.0f / 2.0f;
	maxInputSize = Size(1000, 1000);
}

ImgStitcher::~ImgStitcher()
{
}

vector<Mat> ImgStitcher::loadImgFromFolder(string strPath)
{
	vector<string>vImgNames;
	glob(strPath, vImgNames);

	vector<Mat>vImgs;
	for (int i = 0; i < (int)vImgNames.size(); i++)
	{
		Mat mInput;
		mInput = imread(vImgNames[i]);

		int nTotalPixNum, nMaxAllowedPixNum;
		nTotalPixNum = mInput.rows*mInput.cols;
		nMaxAllowedPixNum = maxInputSize.height*maxInputSize.width;

		if (nTotalPixNum / nMaxAllowedPixNum > 1)
		{
			double dRatioWidth, dRatioHeight;
			dRatioHeight = (double)mInput.rows / (double)maxInputSize.height;
			dRatioWidth = (double)mInput.cols / (double)maxInputSize.width;

			double dResizeRatio = max(dRatioHeight, dRatioWidth);
			Size newSize = Size((int)ceil(mInput.cols / dResizeRatio), (int)ceil(mInput.rows / dResizeRatio));
			resize(mInput, mInput, newSize);
		}
		vImgs.push_back(mInput);
	}
	vector<string>().swap(vImgNames);

	return vImgs;
}

vector<DMatch> ImgStitcher::matchingFilter(vector<vector<DMatch>> vTotalMatches)
{
	vector<DMatch> vFinalMatches;
	for (int i = 0; i < (int)vTotalMatches.size(); i++)
	{
		const DMatch& bestMatch = vTotalMatches[i][0];
		const DMatch& betterMatch = vTotalMatches[i][1];
		double distanceRatio = bestMatch.distance / betterMatch.distance;

		if (distanceRatio < fMatchingConfid)
			vFinalMatches.push_back(bestMatch);
	}

	return vFinalMatches;
}

Mat ImgStitcher::calHomography(vector<KeyPoint> vFeatCurr, vector<KeyPoint> vFeatNext, vector<DMatch> vFinalMatches)
{
	vector<Point2f> vimg_1, vimg_2;
	for (int i = 0; i < (int)vFinalMatches.size(); i++)
	{
		//Get the keypoints from the good matches
		vimg_1.push_back(vFeatCurr[vFinalMatches[i].queryIdx].pt);
		vimg_2.push_back(vFeatNext[vFinalMatches[i].trainIdx].pt);
	}
	Mat H = findHomography(vimg_2, vimg_1, CV_RANSAC);

	vector<Point2f>().swap(vimg_1);
	vector<Point2f>().swap(vimg_2);

	return H;
}

Size ImgStitcher::calOptimalWarpSize(Mat mOriCoord, Mat H, Point2d& pt)
{
	Mat mTransCoord;
	mTransCoord = H * mOriCoord;
	mTransCoord.row(0) = mTransCoord.row(0) / mTransCoord.row(2);
	mTransCoord.row(1) = mTransCoord.row(1) / mTransCoord.row(2);
	mTransCoord.row(2) = mTransCoord.row(2) / mTransCoord.row(2);

	Mat mCombCoord;
	hconcat(mOriCoord, mTransCoord, mCombCoord);

	double minx, miny, maxx, maxy;
	minMaxLoc(mCombCoord.row(0), &minx, &maxx, 0, 0);
	minMaxLoc(mCombCoord.row(1), &miny, &maxy, 0, 0);

	pt.x = minx;
	pt.y = miny;
	Size sz((int)ceil(maxx - minx), (int)ceil(maxy - miny));

	return sz;
}

Mat ImgStitcher::warping(Mat mImg2, Mat H, Point& pos)
{
	double p[3][4] = { { 0, mImg2.cols - 1, mImg2.cols - 1, 0 }, { 0, 0, mImg2.rows - 1, mImg2.rows - 1 }, { 1, 1, 1, 1 } };
	Mat mOriCoord = Mat(3, 4, CV_64F, p);

	Size warpSize;
	Point2d ptMinMax;
	warpSize = calOptimalWarpSize(mOriCoord, H, ptMinMax);

	double mat[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
	Mat mInden = Mat(3, 3, CV_64F, mat);

	pos.x = 0, pos.y = 0;
	if (ptMinMax.x < 0)
	{
		pos.x = (int)-ptMinMax.x;
		mInden.at<double>(0, 2) += pos.x;
		bHmodifyFlag = true;
	}
	if (ptMinMax.y < 0)
	{
		pos.y = (int)-ptMinMax.y;
		mInden.at<double>(1, 2) += pos.y;
		bHmodifyFlag = true;
	}

	if (bHmodifyFlag == true)
	{
		H = mInden*H;

		//Point2d xx;
		//warpSize = calOptimalWarpSize(mOriCoord, H, xx);
	}

	Mat mWarped;
	warpPerspective(mImg2, mWarped, H, warpSize);

	return mWarped;
}

void ImgStitcher::singlePlaneTransform(vector<Mat>vImgs)
{
	//inverse half of the homography
	nReferImgIndex = (int)vImgs.size() / 2;
	for (int i = 0; i < nReferImgIndex; i++)
		vH[i] = vH[i].inv();

	for (int i = 0; i < (int)vImgs.size(); i++)
	{
		Point pos;
		Mat mWarpedImg;
		if (i < nReferImgIndex)
		{
			if (abs(i - nReferImgIndex)>1)
				for (int j = i + 1; j < nReferImgIndex; j++)
					vH[i] = vH[j] * vH[i];

			mWarpedImg = warping(vImgs[i], vH[i], pos);
		}
		else if (i > nReferImgIndex)
		{
			if (abs(i - nReferImgIndex) > 1)
				vH[i - 1] = vH[i - 2] * vH[i - 1];

			mWarpedImg = warping(vImgs[i], vH[i - 1], pos);
		}
		else
		{
			vWarpImg.push_back(vImgs[i]);
			continue;
		}

		vPos.push_back(pos);
		vWarpImg.push_back(mWarpedImg);
	}
}

Mat ImgStitcher::multipleStitch(vector<Mat> vImgs)
{
	cout << "Total sequences: " << (int)vImgs.size() - 1 << endl;
	for (int i = 0; i < (int)vImgs.size() - 1; i++)
	{
		int nCurrent, nNext;
		nCurrent = i;
		nNext = i + 1;

		Mat mCurrImg, mNextImg;
		mCurrImg = vImgs[nCurrent];
		mNextImg = vImgs[nNext];

		cout << i + 1 << " sequence starts..." << endl;
		cout << "SIFT feature extraction starts..." << endl;
		t.start();

		Mat mDescCurr, mDescNext;
		vector<KeyPoint> vFeatCurr, vFeatNext;

		//sift feature computation
		extractSIFT(mCurrImg, mNextImg, vFeatCurr, vFeatNext, mDescCurr, mDescNext);
		
		t.stop();
		cout << i + 1 << " sequence finished" << endl;
		cout << "Time cost: " << t.getTimeSec() << " sec" << endl << endl;

		//nearest neighbour matching 
		cout << "Matching starts..." << endl;
		t.reset();
		t.start();

		vector<vector<DMatch>>vTotalMatches;
		matcher.knnMatch(mDescCurr, mDescNext, vTotalMatches, 2);

		t.stop();
		cout << i + 1 << " sequence finished" << endl;
		cout << "Time cost: " << t.getTimeSec() << " sec" << endl << endl;

		vector<DMatch>vFinalMatches;
		vFinalMatches = matchingFilter(vTotalMatches);

		if (bDisplayMatchingFlag == true)
		{
			Mat mMatchingResult;
			drawMatches(mCurrImg, vFeatCurr,
				mNextImg, vFeatNext,
				vFinalMatches, mMatchingResult,
				Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			imshow("Matching Result", mMatchingResult);
			namedWindow("Matching Result", WINDOW_AUTOSIZE);
			waitKey(0);
		}

		if ((int)vFinalMatches.size() < 4)
		{
			cout << "Too few Matches, can not establish the geometry relationship !" << endl;
			exit(0);
		}
		Mat mHomography;
		mHomography = calHomography(vFeatCurr, vFeatNext, vFinalMatches);
		vH.push_back(mHomography);

		vector<KeyPoint>().swap(vFeatCurr);
		vector<KeyPoint>().swap(vFeatNext);
		vector<DMatch>().swap(vFinalMatches);
		vector<vector<DMatch>>().swap(vTotalMatches);
	}

	cout << "Warp images..." << endl;
	t.reset();
	t.start();
	
	singlePlaneTransform(vImgs);

	cout << "Image warping finished" << endl;
	t.stop();
	cout << "Time cost: " << t.getTimeSec() << " sec" << endl << endl;

	Mat mPanorama;
	mPanorama = registration();

	vector<Mat>().swap(vH);
	vector<Mat>().swap(vWarpImg);
	vector<Point>().swap(vPos);

	return mPanorama;
}

Mat ImgStitcher::registration()
{
	Point pLeft, pRight;
	int nShift_x = 0, nShift_y = 0;
	int nImgNum = (int)vWarpImg.size();
	for (int i = nImgNum - 1; i > nReferImgIndex; i--)
	{
		nShift_x += vPos[i - 1].x;
		nShift_y += vPos[i - 1].y;
		if (nShift_y + vWarpImg[i - 1].rows > vWarpImg[nImgNum - 1].rows)
		{
			int nDiff = nShift_y - (vWarpImg[nImgNum - 1].rows - vWarpImg[i - 1].rows);
			copyMakeBorder(vWarpImg[nImgNum - 1], vWarpImg[nImgNum - 1], 0, nDiff, 0, 0, BORDER_CONSTANT, Scalar(0));
		}
		if (nShift_x + vWarpImg[i - 1].cols > vWarpImg[nImgNum - 1].cols)
		{
			int nDiff = nShift_x - (vWarpImg[nImgNum - 1].cols - vWarpImg[i - 1].cols);
			copyMakeBorder(vWarpImg[nImgNum - 1], vWarpImg[nImgNum - 1], 0, 0, nDiff, 0, BORDER_CONSTANT, Scalar(0));
		}

		Mat mMask;
		mMask = vWarpImg[i - 1] > 0;
		vWarpImg[i - 1].copyTo(vWarpImg[nImgNum - 1](Range(nShift_y, vWarpImg[i - 1].rows + nShift_y),
			Range(nShift_x, vWarpImg[i - 1].cols + nShift_x)),
			mMask);

		if (i == nReferImgIndex + 1)
		{
			pRight.x = nShift_x;
			pRight.y = nShift_y;
		}
	}

	nShift_x = 0, nShift_y = 0;
	for (int i = 0; i < nReferImgIndex; i++)
	{
		if (i<nReferImgIndex - 1)
		{
			nShift_x += vPos[i].x - vPos[i + 1].x;
			nShift_y += vPos[i].y - vPos[i + 1].y;
		}
		else
		{
			nShift_x += vPos[i].x;
			nShift_y += vPos[i].y;
		}

		if (nShift_y + vWarpImg[i + 1].rows > vWarpImg[0].rows)
		{
			int nDiff = nShift_y - (vWarpImg[0].rows - vWarpImg[i + 1].rows);
			copyMakeBorder(vWarpImg[0], vWarpImg[0], 0, nDiff, 0, 0, BORDER_CONSTANT, Scalar(0));
		}
		if (nShift_x + vWarpImg[i + 1].cols > vWarpImg[0].cols)
		{
			int nDiff = nShift_x - (vWarpImg[0].cols - vWarpImg[i + 1].cols);
			copyMakeBorder(vWarpImg[0], vWarpImg[0], 0, 0, 0, nDiff, BORDER_CONSTANT, Scalar(0));
		}

		Mat mMask;
		mMask = vWarpImg[i + 1] > 0;
		vWarpImg[i + 1].copyTo(vWarpImg[0](Range(nShift_y, vWarpImg[i + 1].rows + nShift_y),
			Range(nShift_x, vWarpImg[i + 1].cols + nShift_x)),
			mMask);

		if (i == nReferImgIndex - 1)
		{
			pLeft.x = nShift_x;
			pLeft.y = nShift_y;
		}
	}

	if ((int)vWarpImg.size() > 2)
	{
		int nLbottomBlank, nRbottomBlank;
		nLbottomBlank = vWarpImg[0].rows - pLeft.y - vWarpImg[nReferImgIndex].rows;
		nRbottomBlank = vWarpImg[nImgNum - 1].rows - pRight.y - vWarpImg[nReferImgIndex].rows;

		if (pRight.y - pLeft.y < 0)
			copyMakeBorder(vWarpImg[nImgNum - 1], vWarpImg[nImgNum - 1], abs(pRight.y - pLeft.y), 0, 0, 0, BORDER_CONSTANT, Scalar(0));
		if (nRbottomBlank - nLbottomBlank < 0)
			copyMakeBorder(vWarpImg[nImgNum - 1], vWarpImg[nImgNum - 1], 0, abs(nRbottomBlank - nLbottomBlank), 0, 0, BORDER_CONSTANT, Scalar(0));

		copyMakeBorder(vWarpImg[nImgNum - 1], vWarpImg[nImgNum - 1], 0, 0, abs(pRight.x - pLeft.x), 0, BORDER_CONSTANT, Scalar(0));

		Mat mask = vWarpImg[0]>0;
		if (pRight.y - pLeft.y < 0)
			vWarpImg[0].copyTo(vWarpImg[nImgNum - 1](Range(0, vWarpImg[0].rows), Range(0, vWarpImg[0].cols)), mask);
		else
			vWarpImg[0].copyTo(vWarpImg[nImgNum - 1](Range(abs(pRight.y - pLeft.y), vWarpImg[0].rows + abs(pRight.y - pLeft.y)), Range(0, vWarpImg[0].cols)), mask);

		vWarpImg[0] = vWarpImg[nImgNum - 1];
	}
	Mat mPanorama;
	mPanorama = vWarpImg[0].clone();

	return mPanorama;
}

void ImgStitcher::extractSIFT(Mat mGray1, Mat mGray2, vector<KeyPoint> &vFeat1, vector<KeyPoint> &vFeat2, Mat &mDesc1, Mat &mDesc2)
{
	sift.detect(mGray1, vFeat1);
	sift.detect(mGray2, vFeat2);

	sift.compute(mGray1, vFeat1, mDesc1);
	sift.compute(mGray2, vFeat2, mDesc2);
}

//void ImgStitcher::extractCDVS(Mat mGray1, Mat mGray2, vector<KeyPoint> &vFeat1, vector<KeyPoint> &vFeat2, Mat &mDesc1, Mat &mDesc2)
//{
//	int nModeID = 13;
//	int errorFlag1, errorFlag2;
//	CdvsBuffer rawFeatureBuffer1 = {};
//	CdvsBuffer rawFeatureBuffer2 = {};
//	errorFlag1 = CCE_getFeatures(mGray1.data, mGray1.cols, mGray1.rows, nModeID, &rawFeatureBuffer1);
//	errorFlag2 = CCE_getFeatures(mGray2.data, mGray2.cols, mGray2.rows, nModeID, &rawFeatureBuffer2);
//
//	if (errorFlag1 != 0 || errorFlag2 != 0) {
//		abort();
//	}
//
//	Feature_RAW* pRaw1 = (Feature_RAW*)rawFeatureBuffer1.data;
//	Feature_RAW* pRaw2 = (Feature_RAW*)rawFeatureBuffer2.data;
//	const size_t rf_count1 = rawFeatureBuffer1.size / sizeof(Feature_RAW);
//	const size_t rf_count2 = rawFeatureBuffer2.size / sizeof(Feature_RAW);
//
//	vector<LocalFeatureShort> featBuf1;
//	vector<LocalFeatureShort> featBuf2;
//	featBuf1.resize(rf_count1);
//	featBuf2.resize(rf_count2);
//
//	int nFeat1Num, nFeat2Num;
//	nFeat1Num = CCE_extractLFS(pRaw1, rf_count1, mGray1.cols, mGray1.rows, nModeID, featBuf1.data(), rf_count1);
//	nFeat2Num = CCE_extractLFS(pRaw2, rf_count2, mGray2.cols, mGray2.rows, nModeID, featBuf2.data(), rf_count2);
//
//	vFeat1 = featGenerator(featBuf1, pRaw1, nFeat1Num);
//	vFeat2 = featGenerator(featBuf2, pRaw2, nFeat2Num);
//
//	mDesc1 = descGenerator(featBuf1, nFeat1Num);
//	mDesc2 = descGenerator(featBuf2, nFeat2Num);
//}

