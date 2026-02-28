#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <direct.h>
#include <deque>
#include <limits>
#include <numeric>
#include <cmath>

using namespace std;
using namespace cv;

// ================= [ 1. 全局结构与变量 ] =================

struct HistoryData {
	Point2f weldPoint;
	bool isValid;
};

static deque<HistoryData> g_historyBuffer;
const int HISTORY_SIZE = 5;

// 卡尔曼滤波追踪器 LaserTracker (原字不动保留)
class LaserTracker {
private:
	KalmanFilter kf;
	bool is_initialized;
	int miss_count;

public:
	LaserTracker() : kf(4, 2, 0), is_initialized(false), miss_count(0) {
		kf.transitionMatrix = (Mat_<float>(4, 4) <<
			1, 0, 1, 0,
			0, 1, 0, 1,
			0, 0, 1, 0,
			0, 0, 0, 1);
		kf.measurementMatrix = (Mat_<float>(2, 4) <<
			1, 0, 0, 0,
			0, 1, 0, 0);

		setIdentity(kf.processNoiseCov, Scalar::all(0.1));
		setIdentity(kf.measurementNoiseCov, Scalar::all(0.5));
		setIdentity(kf.errorCovPost, Scalar::all(100));
	}

	Point2f update(Point2f measure) {
		if (measure.x < 0 || measure.y < 0) {
			miss_count++;
			if (miss_count > 30) is_initialized = false;
			Mat pred = kf.predict();
			return Point2f(pred.at<float>(0), pred.at<float>(1));
		}

		if (!is_initialized) {
			kf.statePre = (Mat_<float>(4, 1) << measure.x, measure.y, 0, 0);
			kf.statePost = (Mat_<float>(4, 1) << measure.x, measure.y, 0, 0);
			is_initialized = true;
			return measure;
		}

		miss_count = 0;
		kf.predict();
		Mat measurement = (Mat_<float>(2, 1) << measure.x, measure.y);
		Mat estimated = kf.correct(measurement);
		return Point2f(estimated.at<float>(0), estimated.at<float>(1));
	}
};

static LaserTracker g_tracker;
static Rect g_roi;

Rect get_dynamic_roi(Point2f center, int window_w, int window_h, Size img_shape) {
	if (center.x < 0 || center.y < 0) {
		return Rect(0, 0, img_shape.width, img_shape.height);
	}
	int x_start = max(0, (int)center.x - window_w / 2);
	int y_start = max(0, (int)center.y - window_h / 2);
	int x_stop = min(img_shape.width, (int)center.x + window_w / 2);
	int y_stop = min(img_shape.height, (int)center.y + window_h / 2);

	return Rect(x_start, y_start, x_stop - x_start, y_stop - y_start);
}

// ================= [ 2. 数学几何辅助算法 ] =================

double getDist(Point2f a, Point2f b) {
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

Point2f getLinesIntersection(Vec4f line1, Vec4f line2) {
	double vx1 = line1[0], vy1 = line1[1], x1 = line1[2], y1 = line1[3];
	double vx2 = line2[0], vy2 = line2[1], x2 = line2[2], y2 = line2[3];

	double det = vx1 * vy2 - vy1 * vx2;
	if (abs(det) < 1e-6) return Point2f(-1, -1);

	double t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det;
	return Point2f(x1 + vx1 * t, y1 + vy1 * t);
}

// 稳定 Least-Absolute (L1) 拟合，杜绝十字随机跳动
Vec4f StableFitLine(const vector<Point2f>& points) {
	Vec4f bestLine(0, 1, points.empty() ? 0 : points[0].x, points.empty() ? 0 : points[0].y);
	if (points.size() >= 2) {
		fitLine(points, bestLine, DIST_L1, 0, 0.01, 0.01);
	}
	return bestLine;
}

void drawLineSegment(Mat& img, Vec4f lineParams, Scalar color, int thickness = 1) {
	double vx = lineParams[0], vy = lineParams[1];
	double x0 = lineParams[2], y0 = lineParams[3];
	Point p1(cvRound(x0 - 40 * vx), cvRound(y0 - 40 * vy));
	Point p2(cvRound(x0 + 40 * vx), cvRound(y0 + 40 * vy));
	line(img, p1, p2, color, thickness, LINE_AA);
}

// ================= [ 高鲁棒性灰度重心法提取 (绝对原封不动) ] =================
void GetLaserCenterRobust(const Mat& img, vector<Point2f>& points, int thresholdVal) {
	points.clear();
	for (int r = 0; r < img.rows; r++) {
		const uchar* iptr = img.ptr<uchar>(r);
		int maxVal = 0, maxIdx = -1;
		for (int c = 0; c < img.cols; c++) {
			if (iptr[c] > maxVal) { maxVal = iptr[c]; maxIdx = c; }
		}
		if (maxVal > thresholdVal) {
			double fenzi = 0, fenmu = 0;
			int startC = max(0, maxIdx - 15);
			int endC = min(img.cols - 1, maxIdx + 15);
			for (int c = startC; c <= endC; c++) {
				if (iptr[c] > thresholdVal) {
					double weight = static_cast<double>(iptr[c] - thresholdVal);
					weight = weight * weight;
					fenzi += c * weight; fenmu += weight;
				}
			}
			if (fenmu > 0) points.push_back(Point2f(static_cast<float>(fenzi / fenmu), static_cast<float>(r)));
		}
	}

	if (points.size() < 3) return;
	sort(points.begin(), points.end(), [](const Point2f& a, const Point2f& b) { return a.y < b.y; });

	if (points.size() > 10) {
		vector<float> slopes;
		for (size_t i = 1; i < points.size() - 1; i++) {
			float dy = points[i + 1].y - points[i - 1].y;
			float slope = (dy != 0) ? (points[i + 1].x - points[i - 1].x) / dy : 0.0f;
			slopes.push_back(abs(slope));
		}
		Scalar mean_s, std_s;
		meanStdDev(slopes, mean_s, std_s);
		float thresh_slope = mean_s[0] + 2 * std_s[0];

		vector<Point2f> no_outliers;
		no_outliers.push_back(points[0]);
		for (size_t i = 1; i < points.size() - 1; i++) {
			if (slopes[i - 1] <= thresh_slope) no_outliers.push_back(points[i]);
		}
		no_outliers.push_back(points.back());
		points = no_outliers;
	}
}

// 筛选最佳跟踪目标 (绝对原封不动)
Point2f SelectBestWeldPoint(const vector<Point2f>& candidates, Point2f lastWeldPoint) {
	if (candidates.empty()) return Point2f(-1, -1);
	Point2f bestPoint;
	double minDesc = DBL_MAX;
	bool hasPrior = (lastWeldPoint.x >= 0 && lastWeldPoint.y >= 0);

	for (const auto& pt : candidates) {
		double currentDesc = 0;
		if (hasPrior) {
			double dist = getDist(pt, lastWeldPoint);
			if (dist > 100.0) dist += 10000.0;
			currentDesc = dist;
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

// ================= [ 3. 核心提取管线 (修复：直取原始像素，绝不删点内缩) ] =================

void ExtractGrooveStable(Mat& image, string savePath) {
	double t_start = (double)getTickCount();
	vector<Point2f> rawPoints;

	Scalar color_line(0, 255, 0);
	Scalar color_convex(0, 0, 255);
	Scalar color_concave(255, 0, 0);
	Scalar color_refined(0, 255, 255);
	Scalar color_target(0, 255, 0);
	Scalar color_ransac(255, 0, 255);

	if (image.empty()) return;

	if (g_roi.width <= 0) g_roi = Rect(0, 0, image.cols, image.rows);
	g_roi = g_roi & Rect(0, 0, image.cols, image.rows);
	if (g_roi.width <= 0 || g_roi.height <= 0) g_roi = Rect(0, 0, image.cols, image.rows);

	Mat procImg;
	medianBlur(image(g_roi), procImg, 3);
	GaussianBlur(procImg, procImg, Size(5, 5), 0, 0);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
	Mat tophat;
	morphologyEx(procImg, tophat, MORPH_TOPHAT, kernel);
	addWeighted(procImg, 1.0, tophat, 1.0, 0, procImg);

	Mat bw, cleanMask = Mat::zeros(procImg.size(), CV_8UC1);
	threshold(procImg, bw, 35, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	findContours(bw, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++) {
		Rect boundRect = boundingRect(contours[i]);
		if (contourArea(contours[i]) > 20 || boundRect.width > 15 || boundRect.height > 15) {
			drawContours(cleanMask, contours, (int)i, Scalar(255), FILLED);
		}
	}
	Mat cleanedImg;
	bitwise_and(procImg, cleanMask, cleanedImg);

	// 从底层重心法提取原始点云
	GetLaserCenterRobust(cleanedImg, rawPoints, 25);
	for (auto& pt : rawPoints) { pt.x += g_roi.x; pt.y += g_roi.y; }

	if (rawPoints.size() < 5) return;

	Mat display;
	cvtColor(image, display, COLOR_GRAY2BGR);
	rectangle(display, g_roi, Scalar(255, 0, 0), 1);
	vector<Point2f> allCandidates;

	// ================= [ Step 1: 绝对原像素分段 (绝不删点去噪，防止吃掉边缘) ] =================
	vector<vector<Point2f>> segments;
	vector<Point2f> curSeg;
	curSeg.push_back(rawPoints[0]);

	for (size_t i = 1; i < rawPoints.size(); i++) {
		float dx = abs(rawPoints[i].x - rawPoints[i - 1].x);
		float dy = abs(rawPoints[i].y - rawPoints[i - 1].y);

		// 判断两点之间是否物理断裂 (X跳动>12 或者 Y跳动>3)
		if (dx > 12.0 || dy > 3.0) {
			if (curSeg.size() >= 5) segments.push_back(curSeg); // 滤掉极小残渣
			curSeg.clear();
		}
		curSeg.push_back(rawPoints[i]);
	}
	if (curSeg.size() >= 5) segments.push_back(curSeg);

	// 画出完美贴合的绿线
	for (const auto& seg : segments) {
		for (size_t i = 0; i < seg.size() - 1; i++) {
			line(display, seg[i], seg[i + 1], color_line, 1);
		}
	}

	// ================= [ Step 2: 提取绝对断点 (Gap) ] =================
	for (size_t i = 0; i + 1 < segments.size(); i++) {
		// tip_up 和 tip_down 此时 100% 是物理像素的最尽头，绝不内缩！
		Point2f tip_up = segments[i].back();
		Point2f tip_down = segments[i + 1].front();

		// 只有 Y 轴相距在合理范围内，才是我们要找的搭接缝断点
		if (abs(tip_up.y - tip_down.y) < 100.0) {
			Point2f gapMid = (tip_up + tip_down) * 0.5f;
			allCandidates.push_back(gapMid);

			// 黄色十字死死钉在物理边界像素上
			drawMarker(display, tip_up, color_refined, MARKER_CROSS, 10, 1);
			drawMarker(display, tip_down, color_refined, MARKER_CROSS, 10, 1);
			putText(display, "GAP", gapMid + Point2f(10, 0), FONT_HERSHEY_SIMPLEX, 0.5, color_ransac, 1);
		}
	}

	// ================= [ Step 3: 提取真实折点 (Corner) ] =================
	// 折点寻找严格限制在各自的 segment 内部，绝对不跨越断层
	for (const auto& seg : segments) {
		if (seg.size() < 20) continue;

		int step = 12;
		vector<double> cos_vals(seg.size(), -1.0);

		for (int i = step; i < (int)seg.size() - step; i++) {
			Vec2f v1(seg[i - step].x - seg[i].x, seg[i - step].y - seg[i].y);
			Vec2f v2(seg[i + step].x - seg[i].x, seg[i + step].y - seg[i].y);
			double m1 = norm(v1), m2 = norm(v2);
			if (m1 > 0 && m2 > 0) cos_vals[i] = v1.dot(v2) / (m1 * m2);
		}

		int nms = 10;
		for (int i = step + nms; i < (int)cos_vals.size() - nms; i++) {
			if (cos_vals[i] > -0.995) {
				bool isMax = true;
				for (int j = -nms; j <= nms; j++) {
					if (cos_vals[i + j] > cos_vals[i]) { isMax = false; break; }
				}

				if (isMax) {
					Point2f corner_pixel = seg[i]; // 当前真实物理拐点像素
					Point2f final_corner = corner_pixel;
					bool isFitSuccess = false;

					int skip = 3;
					vector<Point2f> leftArm, rightArm;

					// 直接在连续线段内部采样，绝对安全
					for (int k = skip; i - k >= 0 && leftArm.size() < 25; k++) leftArm.push_back(seg[i - k]);
					for (int k = skip; i + k < (int)seg.size() && rightArm.size() < 25; k++) rightArm.push_back(seg[i + k]);

					Vec4f lL, lR;
					if (leftArm.size() >= 8 && rightArm.size() >= 8) {
						lL = StableFitLine(leftArm);
						lR = StableFitLine(rightArm);

						// 两线夹角不过于平行时才求交点
						double dotProd = lL[0] * lR[0] + lL[1] * lR[1];
						if (abs(dotProd) < 0.98) {
							Point2f intersect = getLinesIntersection(lL, lR);

							// 物理容错：如果交点离原像素太远 (>15像素)，说明算废了，丢弃交点，直接返回原像素！
							if (intersect.x > 0 && norm(intersect - corner_pixel) < 15.0) {
								final_corner = intersect;
								isFitSuccess = true;
							}
						}
					}

					allCandidates.push_back(final_corner);

					Vec2f vL(seg[i - 5].x - seg[i].x, seg[i - 5].y - seg[i].y);
					Vec2f vR(seg[i + 5].x - seg[i].x, seg[i + 5].y - seg[i].y);
					if (vL[1] < 0) { vL[0] = -vL[0]; vL[1] = -vL[1]; }
					if (vR[1] < 0) { vR[0] = -vR[0]; vR[1] = -vR[1]; }
					double cross = vL[0] * vR[1] - vL[1] * vR[0];
					bool isConvex = (cross < 0);
					Scalar color = isConvex ? color_convex : color_concave;

					if (isFitSuccess) {
						drawLineSegment(display, lL, color_ransac, 1);
						drawLineSegment(display, lR, color_ransac, 1);
					}
					drawMarker(display, final_corner, color_refined, MARKER_CROSS, 15, 2);
					putText(display, isConvex ? "Convex" : "Concave", final_corner + Point2f(10, -15), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

					i += nms;
				}
			}
		}
	}

	// ================= [ 4. 融合卡尔曼追踪与优选 (包含点焊逻辑，绝对原字不动) ] =================

	Point2f priorPt(-1, -1);
	if (!g_historyBuffer.empty() && g_historyBuffer.back().isValid) {
		priorPt = g_historyBuffer.back().weldPoint;
	}

	Point2f rawFinalTarget = SelectBestWeldPoint(allCandidates, priorPt);

	bool isTackWeld = false;
	if (rawFinalTarget.x >= 0 && priorPt.x >= 0) {
		double deviation = norm(rawFinalTarget - priorPt);

		bool isFirstFrame = (g_historyBuffer.size() <= 1);
		double max_allowed_deviation = isFirstFrame ? 500.0 : 80.0;

		if (deviation > max_allowed_deviation) {
			rawFinalTarget = Point2f(-1, -1);
		}
		else if (!isFirstFrame && deviation > 8.0 && deviation <= max_allowed_deviation) {
			isTackWeld = true;
		}
	}

	Point2f finalTarget;
	if (isTackWeld) {
		finalTarget = g_tracker.update(Point2f(-1, -1));
	}
	else {
		finalTarget = g_tracker.update(rawFinalTarget);
	}

	g_roi = get_dynamic_roi(finalTarget, 400, 300, image.size());

	// ================= [ 5. 渲染输出与历史维护 ] =================
	HistoryData curData;
	if (isTackWeld) {
		putText(display, "TACK WELD", rawFinalTarget + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
		curData.weldPoint = finalTarget;
		curData.isValid = true;
	}
	else if (rawFinalTarget.x >= 0 || finalTarget.x >= 0) {
		circle(display, finalTarget, 4, color_target, 1);
		drawMarker(display, finalTarget, color_target, MARKER_TILTED_CROSS, 15, 2);
		putText(display, "TARGET", finalTarget + Point2f(15, 15), FONT_HERSHEY_SIMPLEX, 0.6, color_target, 2);

		curData.weldPoint = finalTarget;
		curData.isValid = true;
	}
	else {
		curData.isValid = false;
		putText(display, "LOST", Point(30, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
	}

	g_historyBuffer.push_back(curData);
	if (g_historyBuffer.size() > HISTORY_SIZE) g_historyBuffer.pop_front();

	if (!display.empty()) {
		double time_ms = ((double)getTickCount() - t_start) * 1000.0 / getTickFrequency();
		string timeStr = "Cost: " + to_string(time_ms).substr(0, 5) + " ms";
		putText(display, timeStr, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);

		bool saved = imwrite(savePath, display);
		if (!saved) cout << "\n[ERROR] 保存失败! 请确保路径存在: " << savePath << endl;
		imshow("Stable Result", display);
	}
}

// ================= [ 6. Main 函数 (包含启动 UI 交互) ] =================
int main() {
	_mkdir("./image");
	_mkdir("./image/result_stable");

	string readpath = "./image/24/*.png";
	vector<String> filenames;
	glob(readpath, filenames, true);

	if (filenames.empty()) {
		cout << "未找到图片，请检查 ./image/24/ 目录下是否有 .png 文件" << endl;
		return -1;
	}

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
	g_roi = Rect(0, 0, 0, 0);

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