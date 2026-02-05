#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <direct.h>

// 保留您提供的頭文件邏輯
#include "grayCenterRow.h"    
#include "myExtraceCPDA.h"    

using namespace std;
using namespace cv;

// ================= [ 核心提取流程：灰度重心 + 精確斜率突變檢測 ] =================

void ExtractGrooveWithCentroidAndSlope(Mat& image, string savePath) {
    vector<Point2f> centerPoints;
    Scalar color_line(0, 255, 0); // 綠線表示重心線
    Scalar color_corner(0, 0, 255); // 紅點表示拐點

    if (image.empty()) return;

    // 1. 預處理：Size(3,3) 保持細節，Size(5,5) 更平滑。
    Mat blurImg;
    blur(image, blurImg, Size(3, 3), Point(-1, -1));

    // 2. 【保留核心邏輯】灰度重心法提取亞像素中心點
    grayCenterByRow(blurImg, centerPoints, CENTER_TOTALTHRESH);

    if (centerPoints.size() < 60) return;

    // 準備顯示結果
    Mat display;
    cvtColor(image, display, COLOR_GRAY2BGR);
    drawCenterPoints(display, centerPoints, color_line);

    // 3. 【精度提升邏輯】多點趨勢夾角檢測
    // 相比單純的 step 兩點向量，我們使用滑動視窗內的多點回歸趨勢，防止噪點誘發假拐點
    int step = 10;           // 視窗步長
    int search_dist = 8;     // 局部搜索細化
    double min_cos = 0.985;  // 夾角門檻（約 10 度）

    for (int i = step; i < (int)centerPoints.size() - step; i++) {
        // 取當前點前後各 step 個點，計算趨勢向量
        // 直接使用 p1->p2 和 p2->p3 容易受單點偏差影響
        // 改用 (p[i] - p[i-step]) 與 (p[i+step] - p[i]) 的平均趨勢

        Point2f v1_sum(0, 0), v2_sum(0, 0);
        for (int k = 1; k <= 5; k++) { // 取鄰域 3 個點的平均方向提升魯棒性
            v1_sum += (centerPoints[i] - centerPoints[i - step - k + 1]);
            v2_sum += (centerPoints[i + step + k - 1] - centerPoints[i]);
        }

        double norm1 = norm(v1_sum);
        double norm2 = norm(v2_sum);

        if (norm1 == 0 || norm2 == 0) continue;
        double cosTheta = v1_sum.dot(v2_sum) / (norm1 * norm2);

        // 4. 【局部極值鎖定】在檢測到角度變化的鄰域內尋找餘弦值最小（角度最大）的點
        if (cosTheta < min_cos) {
            double min_local_cos = cosTheta;
            int best_idx = i;

            // 在前後 search_dist 範圍內精確定位轉角峰值
            for (int j = i - search_dist; j <= i + search_dist; j++) {
                if (j < step || j >= (int)centerPoints.size() - step) continue;

                Point2f vt1 = centerPoints[j] - centerPoints[j - step];
                Point2f vt2 = centerPoints[j + step] - centerPoints[j];
                double ct = vt1.dot(vt2) / (norm(vt1) * norm(vt2) + 1e-9);

                if (ct < min_local_cos) {
                    min_local_cos = ct;
                    best_idx = j;
                }
            }

            // 繪製精確識別出的拐點
            circle(display, centerPoints[best_idx], 2, color_corner, -1);

            // 跳過受影響區域，防止重複檢測
            i = best_idx + step;
        }
    }

    // 5. 保存與展示
    if (!display.empty()) {
        imwrite(savePath, display);
        imshow("Enhanced Precision Result", display);
    }
}

// ================= [ 主控制循環 ] =================

int main() {
    _mkdir("./image/result_slope");
    string readpath = "./image/24/*.png";
    vector<String> filenames;
    glob(readpath, filenames, true);

    for (size_t i = 0; i < filenames.size(); i++) {
        Mat src = imread(filenames[i], 0);
        if (src.empty()) continue;

        string base_name = filenames[i].substr(filenames[i].find_last_of("\\/") + 1);
        string save_full_path = "./image/result_slope/" + base_name;

        ExtractGrooveWithCentroidAndSlope(src, save_full_path);

        if (waitKey(30) == 27) break;
    }
    return 0;
}