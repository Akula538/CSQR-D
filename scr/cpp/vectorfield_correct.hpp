#pragma once

#include "TPS_correct.hpp"
#include "tools.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <map>
#include <optional>
#include <tuple>
#include <utility>

namespace qrscanner {

class DirectionField {
public:
    explicit DirectionField(const cv::Mat& img);
    std::pair<std::optional<double>, std::optional<double>> get(double x, double y, int kernel = 35);

private:
    cv::Mat magnitude_;
    cv::Mat orientation_;
    std::map<std::tuple<int, int, int>, std::pair<std::optional<double>, std::optional<double>>> cache_;
};

struct Polyline {
    Points points;
};

Polyline constructPolyline(DirectionField& df, const Point& p, const Point& dir, int QRsize, double line_len, int n, int kernel = 35);
std::vector<Point> PolylineIntersection(const Polyline& cur1, const Polyline& cur2);

std::pair<Points, Points> vectorfield_nodes(const cv::Mat& img,
                                            const Points& points,
                                            const std::vector<Points>& pattern,
                                            int QRsize = 25,
                                            int square_size = 10,
                                            int fract = 1,
                                            int lines_fract = 5,
                                            const TPS* tps = nullptr);

} // namespace qrscanner
