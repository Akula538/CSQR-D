#include "tools.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qrscanner {

double cross(const Point& v1, const Point& v2)
{
    return v1.x * v2.y - v1.y * v2.x;
}

Point center(const Points& v)
{
    if (v.empty())
        return {};

    Point s(0.0, 0.0);
    for (const auto& p : v)
        s += p;
    return s * (1.0 / static_cast<double>(v.size()));
}

double norm(const Point& p)
{
    return std::sqrt(p.x * p.x + p.y * p.y);
}

double dist(const Point& a, const Point& b)
{
    return norm(a - b);
}

Point normalize(const Point& p)
{
    const double n = norm(p);
    if (n == 0.0)
        return {};
    return p * (1.0 / n);
}

std::vector<cv::Point> to_cv_points_i(const Points& points)
{
    std::vector<cv::Point> out;
    out.reserve(points.size());
    for (const auto& p : points)
        out.emplace_back(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)));
    return out;
}

Points to_points_d(const std::vector<cv::Point>& points)
{
    Points out;
    out.reserve(points.size());
    for (const auto& p : points)
        out.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y));
    return out;
}

void show_points(const cv::Mat& img, const Points& points, int size, const cv::Scalar& color, const std::string& path)
{
    cv::Mat paint;
    if (img.channels() == 1)
        cv::cvtColor(img, paint, cv::COLOR_GRAY2BGR);
    else
        paint = img.clone();

    for (const auto& p : points)
        cv::circle(paint, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)), size, color, -1);

    cv::imwrite(path, paint);
}

CutImageResult cut_image(const cv::Mat& img, const Points& qr, const std::vector<Points>& pattern_recovered, int QRsize)
{
    if (qr.empty())
        throw std::runtime_error("cut_image expects non-empty qr points");

    double x1 = qr.front().x;
    double x2 = qr.front().x;
    double y1 = qr.front().y;
    double y2 = qr.front().y;

    for (const auto& p : qr) {
        x1 = std::min(x1, p.x);
        x2 = std::max(x2, p.x);
        y1 = std::min(y1, p.y);
        y2 = std::max(y2, p.y);
    }

    double scale;
    if (QRsize >= 25)
        scale = 0.3;
    else
        scale = 1.6;


    x1 = std::max(x1 - scale * (x2 - x1), 0.0);
    x2 = std::min(x2 + scale * (x2 - x1), static_cast<double>(img.cols));
    y1 = std::max(y1 - scale * (y2 - y1), 0.0);
    y2 = std::min(y2 + scale * (y2 - y1), static_cast<double>(img.rows));

    const Point shift(x1, y1);

    CutImageResult result;
    result.qr.reserve(qr.size());
    for (const auto& p : qr)
        result.qr.push_back(p - shift);

    result.pattern_recovered = pattern_recovered;
    for (auto& pattern : result.pattern_recovered) {
        for (auto& p : pattern)
            p -= shift;
    }

    cv::Rect roi(static_cast<int>(x1), static_cast<int>(y1),
                 std::max(1, static_cast<int>(x2) - static_cast<int>(x1)),
                 std::max(1, static_cast<int>(y2) - static_cast<int>(y1)));
    roi &= cv::Rect(0, 0, img.cols, img.rows);
    result.img = img(roi).clone();
    return result;
}

} // namespace qrscanner
