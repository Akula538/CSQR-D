#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace qrscanner {

using Point = cv::Point2d;
using Points = std::vector<Point>;

double cross(const Point& v1, const Point& v2);
Point center(const Points& v);
double norm(const Point& p);
double dist(const Point& a, const Point& b);
Point normalize(const Point& p);

std::vector<cv::Point> to_cv_points_i(const Points& points);
Points to_points_d(const std::vector<cv::Point>& points);

void show_points(const cv::Mat& img,
                 const Points& points = {},
                 int size = 5,
                 const cv::Scalar& color = cv::Scalar(0, 255, 0),
                 const std::string& path = "marked.png");

struct CutImageResult {
    cv::Mat img;
    Points qr;
    std::vector<Points> pattern_recovered;
};

CutImageResult cut_image(const cv::Mat& img, const Points& qr, const std::vector<Points>& pattern_recovered, int QRsize);

template <class T>
std::vector<T> cyclic_shift(const std::vector<T>& arr, int start_index)
{
    std::vector<T> result;
    if (arr.empty())
        return result;

    const int n = static_cast<int>(arr.size());
    start_index %= n;
    if (start_index < 0)
        start_index += n;

    result.reserve(arr.size());
    for (int i = 0; i < n; ++i)
        result.push_back(arr[(start_index + i) % n]);
    return result;
}

} // namespace qrscanner
