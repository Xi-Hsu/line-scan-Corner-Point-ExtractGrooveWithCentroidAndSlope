#include"grayCenterRow.h"
//利用两张图，棋盘格中心提取
using namespace cv;
using namespace std;

//求某一行的OTSU阈值
int getThreshInRow(Mat& lightGray, int r)
{
	vector<int> rPoints(256, 0);
	for (int j = 0; j < lightGray.cols; ++j)
	{
		++rPoints[lightGray.at<uchar>(r, j)];
	}
	float avg1, avg2;
	int thresh = myThresh(rPoints, avg1, avg2);
	return thresh;
}
//逐行OTSU,效果不好
void myThreshRow(Mat& lightGray)
{
	for (int i = 0; i < lightGray.rows; ++i)
	{
		int thresh = getThreshInRow(lightGray, i);
		//if (thresh <140)
		//{
		//	thresh *= 1.2;
		//}
		//else
		//{
		//	thresh = 220;
		//}

		for (int j = 0; j < lightGray.cols; ++j)
		{
			if (lightGray.at<uchar>(i, j) <= thresh)
			{
				lightGray.at<uchar>(i, j) = 0;
			}
		}
	}
}

bool getCenterInRow(const cv::Mat& gray, int row, double& centerX, int thresh)
{
	double sum = 0;
	double count = 0;
	const uchar * grayRow = gray.ptr<uchar>(row);
	for (int c = 0; c < gray.cols; ++c)
	{
		int val = grayRow[c];
		if (val > thresh)
		{
			sum += (c*val);
			count += val;
		}
	}
	if (count > 0)
	{
		centerX = sum / count;
		return true;
	}
	return false;
}

bool getCenterInRowSquare(const cv::Mat& gray, int row, double& centerX, int thresh /*= 0*/)
{
	double sum = 0;
	double count = 0;
	const uchar * grayRow = gray.ptr<uchar>(row);
	for (int c = 0; c < gray.cols; ++c)
	{
		int val = grayRow[c];
		if (val > thresh)
		{
			sum += (c*val*val);
			count += val * val;
		}
	}
	if (count > 0)
	{
		centerX = sum / count;
		return true;
	}
	return false;
}

void grayCenterByRow(cv::Mat& gray, std::vector<cv::Point2f>& centerPoints, int threshFlag)
{
	if (threshFlag == CENTER_TOTALTHRESH)
	{
		threshold(gray, gray, 0, 255, THRESH_TOZERO + THRESH_OTSU);
		//imshow("gray", gray);
		//waitKey(0);
	}
	if (threshFlag == CENTER_TOTALTHRESH || threshFlag == CENTER_NOTHRESH)
	{
		//阈值后的图求重心
		for (int r = 0; r < gray.rows; ++r)
		{
			double x = 0;
			if (getCenterInRow(gray, r, x))
			{
				//求重心Y坐标
				centerPoints.push_back(Point2f(x, r));
			}
		}
	}
	//逐列阈值
	else if (threshFlag == CENTER_COLTHRESH)
	{
		for (int r = 0; r < gray.rows; ++r)
		{
			//求阈值
			int thresh = getThreshInRow(gray, r);
			double x = 0;
			if (getCenterInRow(gray, r, x, thresh))
			{
				//求重心Y坐标
				centerPoints.push_back(Point2f(x, r));
			}
		}
	}
}

void grayCenterByRowSquare(cv::Mat& gray, std::vector<cv::Point2f>& centerPoints, int threshFlag)
{
	if (threshFlag == CENTER_TOTALTHRESH)
	{
		threshold(gray, gray, 0, 255, THRESH_TOZERO + THRESH_OTSU);
		//imshow("gray", gray);
		//waitKey(0);
	}
	if (threshFlag == CENTER_TOTALTHRESH || threshFlag == CENTER_NOTHRESH)
	{
		//阈值后的图求重心
		for (int r = 0; r < gray.rows; ++r)
		{
			double x = 0;
			if (getCenterInRowSquare(gray, r, x))
			{
				//求重心Y坐标
				centerPoints.push_back(Point2f(x, r));
			}
		}
	}
	//逐列阈值
	else if (threshFlag == CENTER_COLTHRESH)
	{
		for (int r = 0; r < gray.rows; ++r)
		{
			//求阈值
			int thresh = getThreshInRow(gray, r);
			double x = 0;
			if (getCenterInRowSquare(gray, r, x, thresh))
			{
				//求重心Y坐标
				centerPoints.push_back(Point2f(x, r));
			}
		}
	}
}

bool detectStripeInRow(const cv::Mat& lightGray, int row, cv::Vec3i& wPos, int thresh)
{
	int w = 0;
	int maxW = 0;
	int startPos = 0;
	int tmpStartPos = 0;
	int endPos = 0;
	for (int i = 0; i < lightGray.cols;)
	{
		while (i < lightGray.cols && lightGray.at<uchar>(row, i)>thresh)//大于阈值
		{
			//int tmp = lightGray.at<uchar>(row, i);
			if (w == 0)
			{
				tmpStartPos = i;
			}
			++w;
			++i;
		}
		//只检测宽度最大的 光条条纹
		if (w > maxW)
		{
			maxW = w;
			startPos = tmpStartPos;
			endPos = i;//endPos 为尾后元素

		}
		w = 0;
		++i;
	}
	if (maxW >= 3)//条纹宽度阈值
	{
		wPos[0] = maxW;
		wPos[1] = startPos;
		wPos[2] = endPos;
		//(maxW, startPos, endPos);
		return true;
	}
	else
	{
		return false;
	}
}

bool getCenterInRow(const cv::Mat& lightGray, int row, cv::Point2f& centerPoint, int thresh)
{
	Vec3i wPos;
	if (detectStripeInRow(lightGray, row, wPos, thresh))
	{
		double sc = 0;
		double sm = 0;
		for (int j = wPos[1]; j < wPos[2]; ++j)
		{
			sc += lightGray.at<uchar>(row, j);
			sm += j * lightGray.at<uchar>(row, j);
		}
		centerPoint.y = row;
		centerPoint.x = sm / static_cast<float>(sc);
		return true;
	}
	else
	{
		return false;
	}
}

void grayCenterWidthRow(cv::Mat& gray, std::vector<cv::Point2f>& centerPoints, int threshFlag)
{
	if (threshFlag == CENTER_TOTALTHRESH)
	{
		threshold(gray, gray, 0, 255, THRESH_TOZERO + THRESH_OTSU);
		//imshow("gray", gray);
		//waitKey(0);
	}
	if (threshFlag == CENTER_TOTALTHRESH || threshFlag == CENTER_NOTHRESH)
	{
		//阈值后的图求重心
		for (int r = 0; r < gray.rows; ++r)
		{
			Point2f cp;
			if (getCenterInRow(gray, r, cp, 0))
			{
				//求重心Y坐标
				centerPoints.push_back(cp);
			}
		}
	}
	//逐行阈值
	else if (threshFlag == CENTER_COLTHRESH)
	{
		for (int r = 0; r < gray.rows; ++r)
		{
			//求阈值
			int thresh0 = getThreshInRow(gray, r);
			Point2f cp;
			if (getCenterInRow(gray, r, cp, thresh0))
			{
				//求重心Y坐标
				centerPoints.push_back(cp);
			}
		}
	}
}

//赵斌 自适应灰度重心
void weightCenter(Mat& srcImage, vector<Point2f>& centerPoints)
{
	vector<Mat> channels;
	split(srcImage, channels);//分离色彩通道
	Mat& imageRedChannel = channels.at(2);
	//阈值化
	Mat threshed;
	double bestThresh = threshold(imageRedChannel, threshed, 0, 255, THRESH_OTSU);
	threshed = threshed - bestThresh;
	imshow("threhed", threshed);
	waitKey(10);
	//平法加权灰度重心
	grayCenterByRowSquare(threshed, centerPoints, CENTER_NOTHRESH);
}
void extractLaserCenterRobust(const cv::Mat& src, std::vector<cv::Point2f>& centerPoints,
	int threshold, int scanWidth, ExtractMode mode)
{
	centerPoints.clear();
	if (src.empty() || src.type() != CV_8UC1) {
		return;
	}

	int rows = src.rows;
	int cols = src.cols;

	// 预留空间，避免push_back多次重新分配
	centerPoints.reserve(rows);

	// 逐行扫描
	for (int i = 0; i < rows; ++i)
	{
		const uchar* ptr = src.ptr<uchar>(i);

		// 1. 寻找当前行的最大值及其位置 (Peak Search)
		// 这一步是为了抗噪，只关注最亮的区域
		int maxVal = -1;
		int maxIdx = -1;

		// 简单的线性搜索找最大值 (也可以用 minMaxIdx，但在循环里手写往往更快)
		for (int j = 0; j < cols; ++j) {
			if (ptr[j] > maxVal) {
				maxVal = ptr[j];
				maxIdx = j;
			}
		}

		// 2. 阈值筛选：如果该行最亮的地方都没超过阈值，说明没有激光
		if (maxVal < threshold) {
			continue;
		}

		// 3. 定义局部窗口 (ROI)
		// 只在 maxIdx - scanWidth 到 maxIdx + scanWidth 范围内计算重心
		int startCol = std::max(0, maxIdx - scanWidth);
		int endCol = std::min(cols - 1, maxIdx + scanWidth);

		double sumVal = 0.0;
		double sumPos = 0.0;

		// 4. 计算局部重心
		if (mode == MODE_NORMAL)
		{
			for (int j = startCol; j <= endCol; ++j)
			{
				int val = ptr[j];
				// 只有大于一定底噪的值才参与计算，进一步提高精度
				// 这里可以简单的再次判断 > threshold 或者 > 0
				if (val > threshold / 2) {
					sumVal += val;
					sumPos += (double)j * val;
				}
			}
		}
		else if (mode == MODE_SQUARED)
		{
			for (int j = startCol; j <= endCol; ++j)
			{
				int val = ptr[j];
				if (val > threshold / 2) {
					double valSq = (double)val * val;
					sumVal += valSq;
					sumPos += (double)j * valSq;
				}
			}
		}

		// 5. 存储结果
		if (sumVal > 1e-5) { // 避免除以0
			float centerX = static_cast<float>(sumPos / sumVal);

			// 简单的亚像素坐标约束：重心不应该跑出窗口太远
			if (abs(centerX - maxIdx) <= scanWidth) {
				centerPoints.push_back(Point2f(centerX, static_cast<float>(i)));
			}
		}
	}
}