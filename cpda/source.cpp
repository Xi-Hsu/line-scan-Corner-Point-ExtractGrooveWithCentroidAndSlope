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

// ================= [ 1. 全局变量与 Python 逻辑组件 ] =================

struct HistoryData {
    Point2f weldPoint;
    bool isValid;
};

static deque<HistoryData> g_historyBuffer;
const int HISTORY_SIZE = 5;

// 卡尔曼滤波追踪器 LaserTracker
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
static Rect g_roi; // 动态 ROI 窗口

// 动态 ROI 边界计算
Rect get_dynamic_roi(Point2f center, int window_w, int window_h, Size img_shape) {
    if (center.x < 0 || center.y < 0) {
        return Rect(0, 0, img_shape.width, img_shape.height); // 丢失时全图搜索
    }
    int x_start = max(0, (int)center.x - window_w / 2);
    int y_start = max(0, (int)center.y - window_h / 2);
    int x_stop = min(img_shape.width, (int)center.x + window_w / 2);
    int y_stop = min(img_shape.height, (int)center.y + window_h / 2);

    return Rect(x_start, y_start, x_stop - x_start, y_stop - y_start);
}

// ================= [ 2. 辅助函数 ] =================

double getDist(Point2f a, Point2f b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// ================= [ RANSAC 直线拟合 ] =================
Vec4f RansacFitLine(const vector<Point2f>& points, int iterations = 100, float threshold = 1.0f) {
    if (points.size() < 2) return Vec4f(0, 1, points.empty() ? 0 : points[0].x, points.empty() ? 0 : points[0].y);
    RNG rng(getTickCount());
    int bestCount = 0;
    Vec4f bestLine(0, 1, points[0].x, points[0].y);
    for (int i = 0; i < iterations; i++) {
        int idx1 = rng.uniform(0, (int)points.size());
        int idx2 = rng.uniform(0, (int)points.size());
        if (idx1 == idx2) continue;
        Point2f p1 = points[idx1];
        Point2f p2 = points[idx2];
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float len = sqrt(dx * dx + dy * dy);
        if (len < 1e-5) continue;
        float nx = -dy / len, ny = dx / len;
        float c = -(nx * p1.x + ny * p1.y);

        int inliers = 0;
        for (const auto& pt : points) {
            if (abs(nx * pt.x + ny * pt.y + c) < threshold) {
                inliers++;
            }
        }
        if (inliers > bestCount) {
            bestCount = inliers;
            bestLine = Vec4f(dx / len, dy / len, p1.x, p1.y);
        }
    }

    // 利用内点进行最小二乘优化，获得最高精度
    vector<Point2f> inlierPts;
    float nx = -bestLine[1], ny = bestLine[0];
    float c = -(nx * bestLine[2] + ny * bestLine[3]);
    for (const auto& pt : points) {
        if (abs(nx * pt.x + ny * pt.y + c) < threshold) inlierPts.push_back(pt);
    }
    if (inlierPts.size() >= 2) {
        fitLine(inlierPts, bestLine, DIST_L2, 0, 0.01, 0.01);
    }
    return bestLine;
}

// ================= [ 辅助函数：在图上画出直线方程 ] =================
void drawLine(Mat& img, Vec4f lineParams, Scalar color, int thickness = 1) {
    double vx = lineParams[0], vy = lineParams[1];
    double x0 = lineParams[2], y0 = lineParams[3];
    // 沿直线方向延伸，画出一条贯穿图像的长线
    Point2f p1(x0 - 1000 * vx, y0 - 1000 * vy);
    Point2f p2(x0 + 1000 * vx, y0 + 1000 * vy);
    line(img, p1, p2, color, thickness);
}

// ================= [ 局部紧身衣顺延追踪尾巴 ] =================
Point2f ExtendEndpoint(const Mat& img, Point2f pt, bool is_downward, float slope_dx_dy) {
    Point2f current_pt = pt;
    int step = is_downward ? 1 : -1;

    for (int i = 0; i < 30; i++) {
        int r = current_pt.y + step;
        if (r < 0 || r >= img.rows) break;

        float pred_x = current_pt.x + slope_dx_dy * step;
        int sc = max(0, (int)pred_x - 4);
        int ec = min(img.cols - 1, (int)pred_x + 4);

        int maxV = 0;
        int maxC = sc;
        const uchar* ptr = img.ptr<uchar>(r);
        for (int c = sc; c <= ec; c++) {
            if (ptr[c] > maxV) { maxV = ptr[c]; maxC = c; }
        }

        if (maxV < 25) break;

        double fz = 0, fm = 0;
        for (int c = max(0, maxC - 2); c <= min(img.cols - 1, maxC + 2); c++) {
            double w = pow(static_cast<double>(ptr[c]), 2.0);
            fz += c * w; fm += w;
        }
        if (fm > 0) {
            current_pt = Point2f(static_cast<float>(fz / fm), static_cast<float>(r));
        }
        else {
            break;
        }
    }
    return current_pt;
}

// ================= [ 改进版激光中心提取函数 ] =================
void GetLaserCenterRobust(const Mat& img, vector<Point2f>& points, int thresholdVal) {
    points.clear();

    for (int r = 0; r < img.rows; r++) {
        const uchar* iptr = img.ptr<uchar>(r);
        int maxVal = 0;
        int maxIdx = -1;
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

    vector<Point2f> interpolated;
    if (!points.empty()) {
        interpolated.push_back(points[0]);
        for (size_t i = 1; i < points.size(); i++) {
            float dy = points[i].y - points[i - 1].y;
            float dx = abs(points[i].x - points[i - 1].x);

            // 【找回丢失的防线 1】：dx 的容忍度必须严格限制为 <= 2.0f
            // 否则在 ROI 里只要 dx 是 3 或 4，它就会把断崖用线连起来！
            if (dy > 1.5f && dy < 15.0f && dx <= 2.0f) {
                int steps = static_cast<int>(dy);
                for (int s = 1; s < steps; s++) {
                    float weight = s / dy;
                    float interp_x = points[i - 1].x + weight * (points[i].x - points[i - 1].x);
                    float interp_y = points[i - 1].y + s;
                    interpolated.push_back(Point2f(interp_x, interp_y));
                }
            }
            interpolated.push_back(points[i]);
        }
    }
    points = interpolated;

    vector<Point2f> filtered;
    for (size_t i = 0; i < points.size(); i++) filtered.push_back(points[i]);

    if (filtered.size() > 5) {
        for (int i = 2; i < (int)filtered.size() - 2; i++) {
            bool continuous = true;
            for (int k = -2; k < 2; k++) {
                // 【找回丢失的防线 2】：禁止跨越断层进行 5 点数组平滑！
                if (abs(filtered[i + k + 1].y - filtered[i + k].y) > 2.0 ||
                    abs(filtered[i + k + 1].x - filtered[i + k].x) > 2.0) {
                    continuous = false; break;
                }
            }
            if (continuous) {
                filtered[i].x = (filtered[i - 2].x + filtered[i - 1].x + filtered[i].x + filtered[i + 1].x + filtered[i + 2].x) / 5.0f;
            }
        }
    }
    points = filtered;
}

Point2f SelectBestWeldPoint(const vector<Point2f>& candidates, Point2f lastWeldPoint) {
    if (candidates.empty()) return Point2f(-1, -1);
    Point2f bestPoint;
    double minDesc = DBL_MAX;
    bool hasPrior = (lastWeldPoint.x >= 0 && lastWeldPoint.y >= 0);

    for (const auto& pt : candidates) {
        double currentDesc = hasPrior ? getDist(pt, lastWeldPoint) : (10000.0 - pt.y);
        if (currentDesc < minDesc) {
            minDesc = currentDesc;
            bestPoint = pt;
        }
    }
    return bestPoint;
}

// ================= [ 3. 核心处理 ] =================

void ExtractGrooveStable(Mat& image, string savePath) {
    double t_start = (double)getTickCount();
    vector<Point2f> rawPoints;

    Scalar color_line(0, 255, 0);
    Scalar color_convex(0, 0, 255);
    Scalar color_concave(255, 0, 0);
    Scalar color_refined(0, 255, 255);
    Scalar color_target(0, 255, 0);
    Scalar color_ransac(255, 0, 255); // 洋红色用于显示 RANSAC 拟合线

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

    GetLaserCenterRobust(procImg, rawPoints, 25);

    for (auto& pt : rawPoints) { pt.x += g_roi.x; pt.y += g_roi.y; }

    if (rawPoints.size() < 5) return;

    vector<Point2f> cleanPoints;
    cleanPoints.push_back(rawPoints[0]);
    for (size_t i = 1; i < rawPoints.size(); i++) {
        if (abs(rawPoints[i].x - rawPoints[i - 1].x) < 15.0) {
            cleanPoints.push_back(rawPoints[i]);
        }
    }
    if (cleanPoints.size() < 10) cleanPoints = rawPoints;

    vector<Point2f> centerPoints;
    int win = 2;
    for (int i = win; i < (int)cleanPoints.size() - win; i++) {
        // 【找回丢失的防线 3】：在生成最终 centerPoints 时，只要窗口内包含物理断层，就严禁平均！
        bool is_continuous = true;
        for (int j = -win; j < win; j++) {
            if (abs(cleanPoints[i + j + 1].y - cleanPoints[i + j].y) > 2.0 ||
                abs(cleanPoints[i + j + 1].x - cleanPoints[i + j].x) > 2.0) {
                is_continuous = false; break;
            }
        }

        if (is_continuous) {
            float sumX = 0;
            for (int j = -win; j <= win; j++) sumX += cleanPoints[i + j].x;
            centerPoints.push_back(Point2f(sumX / (2 * win + 1), cleanPoints[i].y));
        }
        else {
            centerPoints.push_back(cleanPoints[i]);
        }
    }

    Mat display;
    cvtColor(image, display, COLOR_GRAY2BGR);
    rectangle(display, g_roi, Scalar(255, 0, 0), 1);

    for (size_t i = 0; i < centerPoints.size() - 1; i++) {
        if (abs(centerPoints[i].y - centerPoints[i + 1].y) < 5.0 &&
            abs(centerPoints[i].x - centerPoints[i + 1].x) < 5.0) {
            line(display, centerPoints[i], centerPoints[i + 1], color_line, 1);
        }
    }

    vector<Point2f> allCandidates;

    // ================= [ 核心改进：间隙与搭接断层处理 ] =================
    vector<int> gap_indices;
    for (size_t i = 0; i < centerPoints.size() - 1; i++) {
        float dx = abs(centerPoints[i].x - centerPoints[i + 1].x);
        float dy = abs(centerPoints[i].y - centerPoints[i + 1].y);

        bool is_gap = false;
        int lookahead = 1;

        if (dx > 4.0 || dy > 4.0) {
            is_gap = true; // 应对未被平滑的锐利跳变
        }
        else {
            // 【核心修复】对抗“陡坡化”：检测被插值或平滑拉长的错位断层
            lookahead = min(8, (int)centerPoints.size() - 1 - (int)i);
            float dx_accum = abs(centerPoints[i].x - centerPoints[i + lookahead].x);
            float dy_accum = abs(centerPoints[i].y - centerPoints[i + lookahead].y) + 1e-5f;
            float local_slope = dx_accum / dy_accum;

            // 如果局部发生跳跃，且周围的线段仍然是笔直的，100%是断层
            if (dx_accum > 3.0 && i >= 10 && i + lookahead + 10 < centerPoints.size()) {
                float dy_up = abs(centerPoints[i].y - centerPoints[i - 10].y) + 1e-5f;
                float slope_up = abs(centerPoints[i].x - centerPoints[i - 10].x) / dy_up;

                float dy_dn = abs(centerPoints[i + lookahead + 10].y - centerPoints[i + lookahead].y) + 1e-5f;
                float slope_dn = abs(centerPoints[i + lookahead + 10].x - centerPoints[i + lookahead].x) / dy_dn;

                if (local_slope > slope_up + 0.5f && local_slope > slope_dn + 0.5f) {
                    is_gap = true;
                }
            }
        }

        if (is_gap) {
            gap_indices.push_back((int)i);

            vector<Point2f> topPts, bottomPts;
            int fit_len = 40;
            int safe_margin = 6; // 安全隔离带，坚决避开陡坡污染区

            for (int k = max(0, (int)i - fit_len); k <= max(0, (int)i - safe_margin); k++) {
                topPts.push_back(centerPoints[k]);
            }
            for (int k = min((int)centerPoints.size() - 1, (int)i + lookahead + safe_margin); k <= min((int)centerPoints.size() - 1, (int)i + lookahead + fit_len); k++) {
                bottomPts.push_back(centerPoints[k]);
            }

            Point2f true_tip_up = centerPoints[i];
            if (topPts.size() >= 5) {
                Vec4f lineUp = RansacFitLine(topPts);
                //drawLine(display, lineUp, color_ransac, 1);
                if (abs(lineUp[1]) > 1e-5) {
                    float t = (centerPoints[i].y - lineUp[3]) / lineUp[1];
                    true_tip_up.x = lineUp[2] + lineUp[0] * t;
                }
            }

            int bottom_idx = min((int)centerPoints.size() - 1, (int)i + lookahead);
            Point2f true_tip_down = centerPoints[bottom_idx];
            if (bottomPts.size() >= 5) {
                Vec4f lineDown = RansacFitLine(bottomPts);
                //drawLine(display, lineDown, color_ransac, 1);
                if (abs(lineDown[1]) > 1e-5) {
                    float t = (centerPoints[bottom_idx].y - lineDown[3]) / lineDown[1];
                    true_tip_down.x = lineDown[2] + lineDown[0] * t;
                }
            }

            Point2f gapMidPoint((true_tip_up.x + true_tip_down.x) / 2.0f, (true_tip_up.y + true_tip_down.y) / 2.0f);
            allCandidates.push_back(gapMidPoint);

            // 画出精准的黄色十字
            drawMarker(display, true_tip_up, color_refined, MARKER_CROSS, 15, 1);
            drawMarker(display, true_tip_down, color_refined, MARKER_CROSS, 15, 1);

            i += lookahead + 8; // 跳过整个陡坡区域，防止重复触发
        }
    }

    // ================= [ 恢复原始逻辑处理正常 V 型折点 ] =================
    int step = 13;
    vector<double> curvature(centerPoints.size(), 1.0);
    vector<double> directions(centerPoints.size(), 0.0);

    for (int i = step; i < (int)centerPoints.size() - step; i++) {
        bool near_gap = false;
        for (int g : gap_indices) {
            // NMS逻辑会自动避开被我们判断为断层的区域，阻止假十字生成
            if (abs(g - i) <= 70) {
                near_gap = true; break;
            }
        }
        if (near_gap) continue;

        if (abs(centerPoints[i].y - centerPoints[i - step].y) > step * 1.5) continue;
        if (abs(centerPoints[i + step].y - centerPoints[i].y) > step * 1.5) continue;
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
                Point2f refinedPt = centerPoints[i];
                int skip = 12; int max_fit_len = 50;
                vector<Point2f> leftPts, rightPts;
                for (int k = 1; k <= max_fit_len; k++) {
                    int idx = i - skip - k; if (idx < 0) break;
                    leftPts.push_back(centerPoints[idx]);
                }
                for (int k = 1; k <= max_fit_len; k++) {
                    int idx = i + skip + k; if (idx >= centerPoints.size()) break;
                    rightPts.push_back(centerPoints[idx]);
                }
                if (leftPts.size() >= 4 && rightPts.size() >= 4) {
                    Vec4f lineL, lineR;
                    fitLine(leftPts, lineL, DIST_HUBER, 0, 0.01, 0.01);
                    fitLine(rightPts, lineR, DIST_HUBER, 0, 0.01, 0.01);
                    double vx1 = lineL[0], vy1 = lineL[1], vx2 = lineR[0], vy2 = lineR[1], dot = vx1 * vx2 + vy1 * vy2;
                    if (abs(dot) < 0.99) {
                        double x1 = lineL[2], y1 = lineL[3], x2 = lineR[2], y2 = lineR[3], det = vx1 * vy2 - vy1 * vx2;
                        if (abs(det) > 1e-5) {
                            double t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det;
                            Point2f intersectPt(x1 + vx1 * t, y1 + vy1 * t);
                            if (getDist(intersectPt, centerPoints[i]) < 60.0) {
                                float minDist = 20.0f;
                                Point2f snappedPt = intersectPt;
                                for (const auto& cp : centerPoints) {
                                    float d = getDist(cp, intersectPt);
                                    if (d < minDist) { minDist = d; snappedPt = cp; }
                                }
                                refinedPt = snappedPt;
                            }
                        }
                    }
                }
                drawMarker(display, refinedPt, color_refined, MARKER_CROSS, 15, 1);
                allCandidates.push_back(refinedPt);
                i += nms_win;
            }
        }
    }

    Point2f priorPt(-1, -1);
    if (!g_historyBuffer.empty() && g_historyBuffer.back().isValid) priorPt = g_historyBuffer.back().weldPoint;

    Point2f rawFinalTarget = SelectBestWeldPoint(allCandidates, priorPt);

    bool isTracking = false;
    bool isTackWeld = false;

    if (rawFinalTarget.x >= 0) {
        isTracking = true;
        if (priorPt.x >= 0) {
            double deviation = getDist(rawFinalTarget, priorPt);
            if (deviation > 8.0 && deviation < 60.0) isTackWeld = true;
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

    HistoryData curData;
    if (isTracking) {
        if (isTackWeld) {
            putText(display, "TACK WELD", rawFinalTarget + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            curData.weldPoint = finalTarget;
            curData.isValid = true;
        }
        else {
            circle(display, finalTarget, 3, color_target, 1);
            putText(display, "TARGET", finalTarget + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.6, color_target, 1);
            cout << "Detected KeyPoint at: [" << finalTarget.x << ", " << finalTarget.y << "]" << endl;
            curData.weldPoint = finalTarget;
            curData.isValid = true;
        }
    }
    else {
        curData.isValid = false;
    }

    g_historyBuffer.push_back(curData);
    if (g_historyBuffer.size() > HISTORY_SIZE) g_historyBuffer.pop_front();

    if (!display.empty()) {
        double time_ms = ((double)getTickCount() - t_start) * 1000.0 / getTickFrequency();
        string timeStr = "Cost: " + to_string(time_ms).substr(0, 5) + " ms";
        putText(display, timeStr, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);

        imwrite(savePath, display);
        imshow("Stable Result", display);
    }
}

// ================= [ 4. Main 函数 ] =================
int main() {
    _mkdir("./image/result_stable");
    string readpath = "./image/24/*.png";
    vector<String> filenames;
    glob(readpath, filenames, true);
    if (filenames.empty()) return -1;

    float startU = 0, startV = 0;
    cout << "Input tracking start (u v): ";
    cin >> startU >> startV;

    g_historyBuffer.clear();
    HistoryData initData;
    initData.weldPoint = Point2f(startU, startV);
    initData.isValid = true;
    g_historyBuffer.push_back(initData);

    g_roi = Rect(0, 0, 0, 0);

    for (size_t i = 0; i < filenames.size(); i++) {
        Mat src = imread(filenames[i], 0);
        if (src.empty()) continue;
        ExtractGrooveStable(src, "./image/result_stable/" + filenames[i].substr(filenames[i].find_last_of("\\/") + 1));
        if (waitKey(30) == 27) break;
    }
    return 0;
}