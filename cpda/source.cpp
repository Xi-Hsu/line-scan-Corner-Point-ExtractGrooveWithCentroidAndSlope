#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <direct.h>
#include <deque>    // 用于历史队列
#include <limits>   // 用于 DBL_MAX

// 保留头文件
#include "grayCenterRow.h"    
#include "myExtraceCPDA.h"    

using namespace std;
using namespace cv;

// ================= [ 全局结构与变量 ] =================
struct HistoryData {
	Point2f weldPoint;
	bool isValid;
};

// 历史记录缓冲区
// 策略：如果用户输入了初始点，我们将其作为第0个历史记录存入
static deque<HistoryData> g_historyBuffer;
const int HISTORY_SIZE = 5;

// ================= [ 辅助函数 ] =================

double getDist(Point2f a, Point2f b) {
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

/**
 * @brief 筛选最佳焊点
 * @param candidates 本帧所有候选点
 * @param lastWeldPoint 上一帧点（或者用户输入的先验点）
 */
Point2f SelectBestWeldPoint(const vector<Point2f>& candidates, Point2f lastWeldPoint) {
	if (candidates.empty()) return Point2f(-1, -1);

	Point2f bestPoint;
	double minDesc = DBL_MAX;

	// 只要坐标有效，就视为有先验知识
	bool hasPrior = (lastWeldPoint.x >= 0 && lastWeldPoint.y >= 0);

	for (const auto& pt : candidates) {
		double currentDesc = 0;

		if (hasPrior) {
			// [核心逻辑]：计算与先验点(上一帧 或 用户输入)的欧氏距离
			double dist = getDist(pt, lastWeldPoint);

			// 简单的防突变权重（可选）：如果距离太远(>100像素)，降低其优先级
			if (dist > 100.0) dist += 10000.0;

			currentDesc = dist;
		}
		else {
			// 如果完全没有先验（用户没输入且没历史），只能找X轴中心
			// 但根据你的需求，这里通常不会走到，因为main函数会强制初始化
			currentDesc = pt.y; // 默认备选：找最上方或最下方的点，此处仅作示例
		}

		if (currentDesc < minDesc) {
			minDesc = currentDesc;
			bestPoint = pt;
		}
	}
	return bestPoint;
}

// ================= [ 核心提取流程 ] =================

void ExtractGrooveStable(Mat& image, string savePath) {
	vector<Point2f> rawPoints;
	Scalar color_line(0, 255, 0);

	Scalar color_convex(0, 0, 255);    // 凸 (红色)
	Scalar color_concave(255, 0, 0);   // 凹 (蓝色)
	Scalar color_refined(0, 255, 255); // 精修候选点 (黄色)
	Scalar color_target(0, 255, 0);    // 最终跟踪点 (绿色靶心)

	if (image.empty()) return;

	// 1. 预处理
	Mat blurImg;
	blur(image, blurImg, Size(3, 3), Point(-1, -1));

	// 2. 灰度重心
	grayCenterByRow(blurImg, rawPoints, CENTER_TOTALTHRESH);

	if (rawPoints.size() < 10) return;

	// 3. 中值平滑
	vector<Point2f> centerPoints;
	int win = 1;
	for (int i = win; i < (int)rawPoints.size() - win; i++) {
		vector<float> vx, vy;
		for (int j = -win; j <= win; j++) {
			vx.push_back(rawPoints[i + j].x);
			vy.push_back(rawPoints[i + j].y);
		}
		sort(vx.begin(), vx.end());
		sort(vy.begin(), vy.end());
		centerPoints.push_back(Point2f(vx[win], vy[win]));
	}

	Mat display;
	cvtColor(image, display, COLOR_GRAY2BGR);
	drawCenterPoints(display, centerPoints, color_line);

	// 4. 计算曲率 & 叉乘方向
	int step = 13;
	vector<double> curvature(centerPoints.size(), 1.0);
	vector<double> directions(centerPoints.size(), 0.0);

	for (int i = step; i < (int)centerPoints.size() - step; i++) {
		if (norm(centerPoints[i] - centerPoints[i - step]) > step * 5) continue;
		if (norm(centerPoints[i + step] - centerPoints[i]) > step * 5) continue;

		Vec2f v1 = Vec2f(centerPoints[i].x - centerPoints[i - step].x, centerPoints[i].y - centerPoints[i - step].y);
		Vec2f v2 = Vec2f(centerPoints[i + step].x - centerPoints[i].x, centerPoints[i + step].y - centerPoints[i].y);

		double normProd = norm(v1) * norm(v2);
		if (normProd > 0) {
			curvature[i] = v1.dot(v2) / normProd;
			directions[i] = v1[0] * v2[1] - v1[1] * v2[0];
		}
	}

	// 准备收集所有的精修候选点
	vector<Point2f> allCandidates;

	// 5. NMS & 凹凸判断 & 精修
	int nms_win = 10;
	for (int i = nms_win; i < (int)curvature.size() - nms_win; i++) {
		double cur = curvature[i];

		if (cur < 0.99) {
			bool isMin = true;
			for (int j = i - nms_win; j <= i + nms_win; j++) {
				if (curvature[j] < cur) {
					isMin = false;
					break;
				}
			}
			if (isMin) {
				// [Step 1] 判断凹凸性
				bool isConvex = (directions[i] < 0);
				string typeLabel = isConvex ? "Convex" : "Concave";
				Scalar typeColor = isConvex ? color_convex : color_concave;

				circle(display, centerPoints[i], 4, typeColor, -1);
				putText(display, typeLabel, centerPoints[i] + Point2f(10, 0), FONT_HERSHEY_SIMPLEX, 0.5, typeColor, 1);

				// [Step 2] 拟合精修
				Point2f refinedPt = centerPoints[i];
				int skip = 5;
				int max_fit_len = 40;
				vector<Point2f> leftPts, rightPts;

				for (int k = 1; k <= max_fit_len; k++) {
					int idx = i - skip - k;
					if (idx < 0) break;
					if (k > 1 && norm(centerPoints[idx] - centerPoints[idx + 1]) > 5.0) break;
					leftPts.push_back(centerPoints[idx]);
				}
				for (int k = 1; k <= max_fit_len; k++) {
					int idx = i + skip + k;
					if (idx >= centerPoints.size()) break;
					if (k > 1 && norm(centerPoints[idx] - centerPoints[idx - 1]) > 5.0) break;
					rightPts.push_back(centerPoints[idx]);
				}

				if (leftPts.size() >= 2 && rightPts.size() >= 2) {
					Vec4f lineL, lineR;
					fitLine(leftPts, lineL, DIST_L2, 0, 0.01, 0.01);
					fitLine(rightPts, lineR, DIST_L2, 0, 0.01, 0.01);

					double vx1 = lineL[0], vy1 = lineL[1], x1 = lineL[2], y1 = lineL[3];
					double vx2 = lineR[0], vy2 = lineR[1], x2 = lineR[2], y2 = lineR[3];
					double det = vx1 * vy2 - vy1 * vx2;

					if (abs(det) > 1e-5) {
						double t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det;
						Point2f intersectPt(x1 + vx1 * t, y1 + vy1 * t);
						if (norm(intersectPt - centerPoints[i]) < 50.0) {
							refinedPt = intersectPt;
						}
					}
				}

				drawMarker(display, refinedPt, color_refined, MARKER_CROSS, 20, 2);

				// 将该候选点加入列表
				allCandidates.push_back(refinedPt);

				i += nms_win;
			}
		}
	}

	// ================= [ Step 6: 结合先验/历史筛选 ] =================

	// 6.1 从全局历史中获取参考点
	// 如果是第一帧，这里取出来的就是 Main 函数里用户输入的那个点
	// 如果是后续帧，这里就是上一帧跟踪到的点
	Point2f priorPt(-1, -1);
	if (!g_historyBuffer.empty() && g_historyBuffer.back().isValid) {
		priorPt = g_historyBuffer.back().weldPoint;
	}

	// 6.2 选出最佳点
	Point2f finalTarget = SelectBestWeldPoint(allCandidates, priorPt);
	// -----------------------------------------------------------
	// 【修改后的偏差检查逻辑】
	// -----------------------------------------------------------
	//if (finalTarget.x >= 0 && priorPt.x >= 0) {

	//	// 计算实际偏差
	//	double deviation = norm(finalTarget - priorPt);

	//	// [关键修改]：动态设置阈值
	//	// 如果历史记录只有1个（说明这是刚启动的第一帧），给一个很大的宽容度（例如 500）
	//	// 如果是后续跟踪，才使用严格的阈值（例如 80）
	//	bool isFirstFrame = (g_historyBuffer.size() <= 1);
	//	double max_allowed_deviation = isFirstFrame ? 500.0 : 80.0;

	//	// 打印调试信息 (非常重要，让你看到到底差了多少)
	//	cout << "[Tracking] Frame: " << (isFirstFrame ? "INIT" : "RUN")
	//		<< " | Dist: " << deviation
	//		<< " | Limit: " << max_allowed_deviation << endl;

	//	// 判定
	//	if (deviation > max_allowed_deviation) {
	//		cout << " -> [LOST] Deviation too large!" << endl;
	//		finalTarget = Point2f(-1, -1); // 标记无效
	//	}
	//}
	// 6.3 更新历史并绘图
	if (finalTarget.x >= 0) {
		// 绘制绿色靶心
		circle(display, finalTarget, 4, color_target, 1);
		drawMarker(display, finalTarget, color_target, MARKER_TILTED_CROSS, 10, 1);
		putText(display, "TARGET", finalTarget + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.6, color_target, 2);

		// 更新历史：找到新的位置
		HistoryData curData;
		curData.weldPoint = finalTarget;
		curData.isValid = true;
		g_historyBuffer.push_back(curData);
	}
	else {
		// 没找到点，标记无效，或者保留上一帧的位置(视策略而定)
		HistoryData curData;
		curData.isValid = false;
		g_historyBuffer.push_back(curData); // 推入无效，防止下一帧乱连

		putText(display, "LOST", Point(30, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
	}

	// 保持历史队列长度
	if (g_historyBuffer.size() > HISTORY_SIZE) {
		g_historyBuffer.pop_front();
	}

	if (!display.empty()) {
		imwrite(savePath, display);
		imshow("Stable Result", display);
	}
}

// ================= [ 主程序：包含用户交互 ] =================

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
	 //1. 读取第一张图给用户看（可选，为了让用户知道大概坐标）
	Mat firstImg = imread(filenames[0], 0);
	if (!firstImg.empty()) {
		imshow("First Image - Please Check Coordinate", firstImg);
		waitKey(100); // 刷新一下窗口
	}

	float startU = 0, startV = 0;
	cout << "========================================" << endl;
	cout << "Please input tracking start point (u v): ";
	cin >> startU >> startV;
	cout << "Tracking initialized at: (" << startU << ", " << startV << ")" << endl;
	cout << "========================================" << endl;

	// 2. 初始化全局历史记录
	// 这样 ExtractGrooveStable 处理第一帧时，会把这个点当做“上一帧”的结果
	g_historyBuffer.clear();
	HistoryData initData;
	initData.weldPoint = Point2f(startU, startV);
	initData.isValid = true;
	g_historyBuffer.push_back(initData);

	// 关闭预览窗口
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