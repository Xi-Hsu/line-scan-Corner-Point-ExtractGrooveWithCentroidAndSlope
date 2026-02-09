#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <direct.h>
#include <deque>
#include <limits>
#include <numeric> // for std::accumulate

using namespace std;
using namespace cv;

// ================= [ 1. 全局变量 ] =================

struct HistoryData {
	Point2f weldPoint;
	bool isValid;
};

static deque<HistoryData> g_historyBuffer;
const int HISTORY_SIZE = 5;

// ================= [ 2. 辅助函数 ] =================

double getDist(Point2f a, Point2f b) {
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// 鲁棒的局部灰度重心法 (增加信噪比检查，防止提取背景噪点)
void GetLaserCenterRobust(const Mat& img, vector<Point2f>& points, int thresholdVal) {
	points.clear();
	int scanWidth = 10;

	for (int r = 0; r < img.rows; r++) {
		const uchar* ptr = img.ptr<uchar>(r);

		// 1. 找峰值
		int maxVal = 0;
		int maxIdx = -1;
		int rowSum = 0; // 用于计算行平均值
		for (int c = 0; c < img.cols; c++) {
			int val = ptr[c];
			if (val > maxVal) {
				maxVal = val;
				maxIdx = c;
			}
			rowSum += val;
		}

		// 2. 信噪比/阈值检查
		double rowMean = (double)rowSum / img.cols;
		if (maxVal < thresholdVal || maxIdx == -1 || (maxVal - rowMean) < 10) continue;

		// 3. 局部重心计算
		double sumVal = 0.0;
		double sumPos = 0.0;
		int startCol = max(0, maxIdx - scanWidth);
		int endCol = min(img.cols - 1, maxIdx + scanWidth);

		for (int c = startCol; c <= endCol; c++) {
			int val = ptr[c];
			if (val > thresholdVal / 2) {
				sumVal += val;
				sumPos += val * c;
			}
		}

		if (sumVal > 0) {
			float subPixelX = (float)(sumPos / sumVal);
			if (abs(subPixelX - maxIdx) < 5.0) {
				points.push_back(Point2f(subPixelX, (float)r));
			}
		}
	}
}

// 筛选最佳焊点
Point2f SelectBestWeldPoint(const vector<Point2f>& candidates, Point2f lastWeldPoint) {
	if (candidates.empty()) return Point2f(-1, -1);

	Point2f bestPoint;
	double minDesc = DBL_MAX;
	bool hasPrior = (lastWeldPoint.x >= 0 && lastWeldPoint.y >= 0);

	for (const auto& pt : candidates) {
		double currentDesc = 0;
		if (hasPrior) {
			currentDesc = getDist(pt, lastWeldPoint);
		}
		else {
			currentDesc = 10000.0 - pt.y;
		}

		if (currentDesc < minDesc) {
			minDesc = currentDesc;
			bestPoint = pt;
		}
	}
	return bestPoint;
}

// ================= [ 3. 核心处理 ] =================

void ExtractGrooveStable(Mat& image, string savePath) {
	// [新增] 计时开始
	double t_start = (double)getTickCount();

	vector<Point2f> rawPoints;

	Scalar color_line(0, 255, 0);
	Scalar color_convex(0, 0, 255);
	Scalar color_concave(255, 0, 0);
	Scalar color_refined(0, 255, 255);
	Scalar color_target(0, 255, 0);

	if (image.empty()) return;

	// --- 1. 图像增强 ---
	Mat procImg = image.clone();
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i) {
		p[i] = saturate_cast<uchar>(pow(i / 255.0, 0.6) * 255.0);
	}
	LUT(image, lookUpTable, procImg);
	blur(procImg, procImg, Size(3, 3));

	// --- 2. 提取重心 ---
	GetLaserCenterRobust(procImg, rawPoints, 30);

	if (rawPoints.size() < 5) return;

	// --- 3. 剔除横向跳变点 (去噪) ---
	vector<Point2f> cleanPoints;
	cleanPoints.push_back(rawPoints[0]);
	for (size_t i = 1; i < rawPoints.size(); i++) {
		if (abs(rawPoints[i].x - rawPoints[i - 1].x) < 10.0) {
			cleanPoints.push_back(rawPoints[i]);
		}
	}
	if (cleanPoints.size() < 10) cleanPoints = rawPoints;

	// --- 4. 平滑 ---
	vector<Point2f> centerPoints;
	int win = 2;
	for (int i = win; i < (int)cleanPoints.size() - win; i++) {
		vector<float> vx, vy;
		for (int j = -win; j <= win; j++) {
			vx.push_back(cleanPoints[i + j].x);
			vy.push_back(cleanPoints[i + j].y);
		}
		sort(vx.begin(), vx.end());
		sort(vy.begin(), vy.end());
		centerPoints.push_back(Point2f(vx[win], vy[win]));
	}

	Mat display;
	cvtColor(image, display, COLOR_GRAY2BGR);

	// 绘制绿线
	for (size_t i = 0; i < centerPoints.size() - 1; i++) {
		if (getDist(centerPoints[i], centerPoints[i + 1]) < 10.0) {
			line(display, centerPoints[i], centerPoints[i + 1], color_line, 1);
		}
	}

	// --- 5. 特征提取 ---
	vector<Point2f> allCandidates;
	int step = 13;
	vector<double> curvature(centerPoints.size(), 1.0);
	vector<double> directions(centerPoints.size(), 0.0);

	for (int i = step; i < (int)centerPoints.size() - step; i++) {
		if (norm(centerPoints[i] - centerPoints[i - step]) > step * 3) continue;
		if (norm(centerPoints[i + step] - centerPoints[i]) > step * 3) continue;

		Vec2f v1 = Vec2f(centerPoints[i].x - centerPoints[i - step].x, centerPoints[i].y - centerPoints[i - step].y);
		Vec2f v2 = Vec2f(centerPoints[i + step].x - centerPoints[i].x, centerPoints[i + step].y - centerPoints[i].y);
		double normProd = norm(v1) * norm(v2);
		if (normProd > 0) {
			curvature[i] = v1.dot(v2) / normProd;
			directions[i] = v1[0] * v2[1] - v1[1] * v2[0];
		}
	}

	int nms_win = 10;
	for (int i = nms_win; i < (int)curvature.size() - nms_win; i++) {
		if (curvature[i] < 0.99) {
			bool isMin = true;
			for (int j = i - nms_win; j <= i + nms_win; j++) {
				if (curvature[j] < curvature[i]) { isMin = false; break; }
			}
			if (isMin) {
				bool isConvex = (directions[i] < 0);
				Scalar typeColor = isConvex ? color_convex : color_concave;
				circle(display, centerPoints[i], 3, typeColor, -1);

				Point2f refinedPt = centerPoints[i];
				int skip = 5; int max_fit_len = 40;
				vector<Point2f> leftPts, rightPts;

				for (int k = 1; k <= max_fit_len; k++) {
					int idx = i - skip - k; if (idx < 0) break;
					if (getDist(centerPoints[idx], centerPoints[idx + 1]) > 5.0) break;
					leftPts.push_back(centerPoints[idx]);
				}
				for (int k = 1; k <= max_fit_len; k++) {
					int idx = i + skip + k; if (idx >= centerPoints.size()) break;
					if (getDist(centerPoints[idx], centerPoints[idx - 1]) > 5.0) break;
					rightPts.push_back(centerPoints[idx]);
				}

				if (leftPts.size() >= 3 && rightPts.size() >= 3) {
					Vec4f lineL, lineR;
					fitLine(leftPts, lineL, DIST_L2, 0, 0.01, 0.01);
					fitLine(rightPts, lineR, DIST_L2, 0, 0.01, 0.01);
					double vx1 = lineL[0], vy1 = lineL[1], x1 = lineL[2], y1 = lineL[3];
					double vx2 = lineR[0], vy2 = lineR[1], x2 = lineR[2], y2 = lineR[3];
					double det = vx1 * vy2 - vy1 * vx2;
					if (abs(det) > 1e-5) {
						double t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det;
						Point2f intersectPt(x1 + vx1 * t, y1 + vy1 * t);
						if (norm(intersectPt - centerPoints[i]) < 50.0) refinedPt = intersectPt;
					}
				}
				drawMarker(display, refinedPt, color_refined, MARKER_CROSS, 15, 1);
				allCandidates.push_back(refinedPt);
				i += nms_win;
			}
		}
	}

	// --- 6. 跟踪逻辑 ---
	Point2f priorPt(-1, -1);
	if (!g_historyBuffer.empty() && g_historyBuffer.back().isValid) {
		priorPt = g_historyBuffer.back().weldPoint;
	}

	Point2f finalTarget = SelectBestWeldPoint(allCandidates, priorPt);

	bool isTracking = false;

	if (finalTarget.x >= 0) {
		if (priorPt.x >= 0) {
			double deviation = getDist(finalTarget, priorPt);
			double limit = 100.0;
			if (deviation > limit) {
				cout << "[Info] Deviation (" << deviation << ") > Limit. Resetting tracker to new visual feature." << endl;
				circle(display, finalTarget, 10, Scalar(0, 255, 255), 2);
			}
		}
		isTracking = true;
	}

	HistoryData curData;
	if (isTracking) {
		circle(display, finalTarget, 6, color_target, 2);
		drawMarker(display, finalTarget, color_target, MARKER_TILTED_CROSS, 20, 2);
		putText(display, "TARGET", finalTarget + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.6, color_target, 2);
		curData.weldPoint = finalTarget;
		curData.isValid = true;
	}
	else {
		putText(display, "LOST", Point(30, 80), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 255), 3);
		curData.isValid = false;
	}

	g_historyBuffer.push_back(curData);
	if (g_historyBuffer.size() > HISTORY_SIZE) g_historyBuffer.pop_front();

	if (!display.empty()) {
		// [新增] 计时结束
		double t_end = (double)getTickCount();
		double time_ms = (t_end - t_start) * 1000.0 / getTickFrequency();

		// [新增] 打印和绘制时间
		cout << "[Timer] Cost: " << time_ms << " ms" << endl;
		putText(display, to_string((int)time_ms) + " ms", Point(display.cols - 120, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

		imwrite(savePath, display);
		imshow("Stable Result", display);
	}
}

// Main 函数 (严格原字不动)
int main() {
	_mkdir("./image/result_stable");
	string readpath = "./image/24/*.png";
	vector<String> filenames;
	glob(readpath, filenames, true);

	if (filenames.empty()) {
		cout << "No images found!" << endl;
		return -1;
	}

	// --- [用户交互部分] ---
	Mat firstImg = imread(filenames[0], 0);
	if (!firstImg.empty()) {
		imshow("First Image - Please Check Coordinate", firstImg);
		waitKey(100);
	}

	float startU = 0, startV = 0;
	cout << "========================================" << endl;
	cout << "Please input tracking start point (u v): ";
	cin >> startU >> startV;
	cout << "Tracking initialized at: (" << startU << ", " << startV << ")" << endl;
	cout << "========================================" << endl;

	g_historyBuffer.clear();
	HistoryData initData;
	initData.weldPoint = Point2f(startU, startV);
	initData.isValid = true;
	g_historyBuffer.push_back(initData);

	destroyWindow("First Image - Please Check Coordinate");

	// --- [开始处理循环] ---
	for (size_t i = 0; i < filenames.size(); i++) {
		Mat src = imread(filenames[i], 0);
		if (src.empty()) continue;
		string base_name = filenames[i].substr(filenames[i].find_last_of("\\/") + 1);

		cout << "Processing: " << base_name << " | Prior: " << g_historyBuffer.back().weldPoint << endl;

		ExtractGrooveStable(src, "./image/result_stable/" + base_name);

		if (waitKey(30) == 27) break;
	}
	return 0;
}