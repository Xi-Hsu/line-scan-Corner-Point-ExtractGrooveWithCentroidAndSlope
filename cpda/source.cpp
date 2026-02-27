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

// 求解两条直线(由 fitLine 输出的方向向量和点)的数学绝对交点
Point2f getLinesIntersection(Vec4f line1, Vec4f line2) {
    double vx1 = line1[0], vy1 = line1[1], x1 = line1[2], y1 = line1[3];
    double vx2 = line2[0], vy2 = line2[1], x2 = line2[2], y2 = line2[3];

    double det = vx1 * vy2 - vy1 * vx2;
    if (abs(det) < 1e-6) return Point2f(-1, -1); // 平行线无交点

    double t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det;
    return Point2f(x1 + vx1 * t, y1 + vy1 * t);
}

// RANSAC 直线拟合算法
Vec4f RansacFitLine(const vector<Point2f>& points, int iterations = 100, float threshold = 1.5f) {
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
            if (abs(nx * pt.x + ny * pt.y + c) < threshold) inliers++;
        }

        if (inliers > bestCount) {
            bestCount = inliers;
            bestLine = Vec4f(dx / len, dy / len, p1.x, p1.y);
        }
    }

    vector<Point2f> inlierPts;
    float nx = -bestLine[1], ny = bestLine[0];
    float c = -(nx * bestLine[2] + ny * bestLine[3]);
    for (const auto& pt : points) {
        if (abs(nx * pt.x + ny * pt.y + c) < threshold) inlierPts.push_back(pt);
    }
    if (inlierPts.size() >= 2) fitLine(inlierPts, bestLine, DIST_L2, 0, 0.01, 0.01);

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

// ================= [ 3. 核心提取管线 (绝不消失的折点版) ] =================

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

    GetLaserCenterRobust(cleanedImg, rawPoints, 25);

    for (auto& pt : rawPoints) { pt.x += g_roi.x; pt.y += g_roi.y; }

    if (rawPoints.size() < 5) return;

    // 粗滤离谱噪点
    vector<Point2f> cleanPoints;
    cleanPoints.push_back(rawPoints[0]);
    for (size_t i = 1; i < rawPoints.size(); i++) {
        if (abs(rawPoints[i].y - rawPoints[i - 1].y) < 50.0) {
            cleanPoints.push_back(rawPoints[i]);
        }
    }

    // 中值拔除X方向孤立毛刺
    if (cleanPoints.size() > 10) {
        vector<Point2f> despatteredPoints;
        int win_size = 3;
        for (int i = 0; i < (int)cleanPoints.size(); i++) {
            vector<float> local_x;
            for (int j = max(0, i - win_size); j <= min((int)cleanPoints.size() - 1, i + win_size); j++) {
                local_x.push_back(cleanPoints[j].x);
            }
            sort(local_x.begin(), local_x.end());
            if (abs(cleanPoints[i].x - local_x[local_x.size() / 2]) <= 3.0f) {
                despatteredPoints.push_back(cleanPoints[i]);
            }
        }
        cleanPoints = despatteredPoints;
    }
    if (cleanPoints.size() < 10) cleanPoints = rawPoints;

    Mat display;
    cvtColor(image, display, COLOR_GRAY2BGR);
    rectangle(display, g_roi, Scalar(255, 0, 0), 1);
    vector<Point2f> allCandidates;

    // 画出原始绿线
    for (size_t i = 0; i < cleanPoints.size() - 1; i++) {
        // 断崖处不连线
        if (abs(cleanPoints[i].x - cleanPoints[i + 1].x) < 15.0 && abs(cleanPoints[i].y - cleanPoints[i + 1].y) < 10.0) {
            line(display, cleanPoints[i], cleanPoints[i + 1], color_line, 1);
        }
    }

    // ================= [ Step 1: 提取真实断点 (Gap) ] =================
    // 发现大跳变直接定为断点，利用 RANSAC 往回倒推端点，防止塌陷
    for (size_t i = 0; i < cleanPoints.size() - 1; i++) {
        float dx = abs(cleanPoints[i].x - cleanPoints[i + 1].x);
        float dy = abs(cleanPoints[i].y - cleanPoints[i + 1].y);

        // 如果横向突变(间隙)>10，或纵向突变>5(行扫描中说明跨越了黑带)
        if (dx > 10.0 || dy > 5.0) {
            Point2f tip_up = cleanPoints[i];
            Point2f tip_down = cleanPoints[i + 1];

            // RANSAC 上臂退后拟合
            vector<Point2f> topPts;
            for (int k = max(0, (int)i - 20); k <= max(0, (int)i - 2); k++) topPts.push_back(cleanPoints[k]);
            if (topPts.size() >= 5) {
                Vec4f lineUp = RansacFitLine(topPts);
                if (abs(lineUp[1]) > 1e-5) tip_up.x = lineUp[2] + lineUp[0] * (tip_up.y - lineUp[3]) / lineUp[1];
            }

            // RANSAC 下臂退后拟合
            vector<Point2f> bottomPts;
            for (int k = min((int)cleanPoints.size() - 1, (int)i + 3); k <= min((int)cleanPoints.size() - 1, (int)i + 23); k++) bottomPts.push_back(cleanPoints[k]);
            if (bottomPts.size() >= 5) {
                Vec4f lineDown = RansacFitLine(bottomPts);
                if (abs(lineDown[1]) > 1e-5) tip_down.x = lineDown[2] + lineDown[0] * (tip_down.y - lineDown[3]) / lineDown[1];
            }

            Point2f gapMid = (tip_up + tip_down) * 0.5f;
            allCandidates.push_back(gapMid);

            drawMarker(display, tip_up, color_refined, MARKER_CROSS, 10, 1);
            drawMarker(display, tip_down, color_refined, MARKER_CROSS, 10, 1);
            putText(display, "GAP", gapMid + Point2f(10, 0), FONT_HERSHEY_SIMPLEX, 0.5, color_ransac, 1);
        }
    }

    // ================= [ Step 2: 提取真实折点 (解封版 + 兜底防消失) ] =================
    int step = 12;
    vector<double> cos_vals(cleanPoints.size(), -1.0);

    for (int i = step; i < (int)cleanPoints.size() - step; i++) {
        // 放宽距离限制，只防跨越“万丈深渊”，绝不误伤陡峭斜坡！(阈值放到80)
        if (norm(cleanPoints[i] - cleanPoints[i - step]) > 80.0) continue;
        if (norm(cleanPoints[i + step] - cleanPoints[i]) > 80.0) continue;

        Vec2f v1(cleanPoints[i - step].x - cleanPoints[i].x, cleanPoints[i - step].y - cleanPoints[i].y);
        Vec2f v2(cleanPoints[i + step].x - cleanPoints[i].x, cleanPoints[i + step].y - cleanPoints[i].y);
        double m1 = norm(v1), m2 = norm(v2);
        if (m1 > 0 && m2 > 0) cos_vals[i] = v1.dot(v2) / (m1 * m2);
    }

    int nms = 10;
    for (int i = step + nms; i < (int)cos_vals.size() - nms; i++) {
        // 极度灵敏：cos > -0.995（只要不是绝对直线，稍微有点拐角都抓出来）
        if (cos_vals[i] > -0.995) {
            bool isMax = true;
            for (int j = -nms; j <= nms; j++) {
                if (cos_vals[i + j] > cos_vals[i]) { isMax = false; break; }
            }

            if (isMax) {
                // 找到嫌疑拐点 i。
                Point2f final_corner = cleanPoints[i]; // 默认使用原始极值点作为物理兜底！
                bool isRansacSuccess = false;

                // 尝试用 RANSAC 避开圆滑底部求绝对数学交点
                int skip = 5;
                int arm_len = 25;
                vector<Point2f> leftArm, rightArm;

                for (int k = skip; k < skip + arm_len && i - k >= 0; k++) leftArm.push_back(cleanPoints[i - k]);
                for (int k = skip; k < skip + arm_len && i + k < (int)cleanPoints.size(); k++) rightArm.push_back(cleanPoints[i + k]);

                Vec4f lL, lR;
                if (leftArm.size() >= 8 && rightArm.size() >= 8) {
                    lL = RansacFitLine(leftArm);
                    lR = RansacFitLine(rightArm);

                    Point2f intersect = getLinesIntersection(lL, lR);

                    // 【兜底判断】：如果 RANSAC 算出的交点是正常的（距离原点不超过 30 个像素）
                    // 那就采用 RANSAC 带来的高精度。如果不正常，就废弃 RANSAC，直接输出原始点！
                    if (intersect.x > 0 && norm(intersect - cleanPoints[i]) < 30.0) {
                        final_corner = intersect;
                        isRansacSuccess = true;
                    }
                }

                // 把这个“绝不会消失”的角点放进候选池
                allCandidates.push_back(final_corner);

                // 判断凹凸性 (使用局部相邻点叉乘，避免 RANSAC 失败时无向量可用)
                Vec2f vL(cleanPoints[i - skip].x - cleanPoints[i].x, cleanPoints[i - skip].y - cleanPoints[i].y);
                Vec2f vR(cleanPoints[i + skip].x - cleanPoints[i].x, cleanPoints[i + skip].y - cleanPoints[i].y);
                if (vL[1] < 0) { vL[0] = -vL[0]; vL[1] = -vL[1]; }
                if (vR[1] < 0) { vR[0] = -vR[0]; vR[1] = -vR[1]; }
                double cross = vL[0] * vR[1] - vL[1] * vR[0];
                bool isConvex = (cross < 0);
                Scalar color = isConvex ? color_convex : color_concave;

                // 画图可视化
                if (isRansacSuccess) {
                    drawLineSegment(display, lL, color_ransac, 1);
                    drawLineSegment(display, lR, color_ransac, 1);
                }
                drawMarker(display, final_corner, color_refined, MARKER_CROSS, 15, 2);
                putText(display, isConvex ? "Convex" : "Concave", final_corner + Point2f(10, -15), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

                i += nms; // 找到了就跳过这片区域
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