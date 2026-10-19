#pragma once

#include "TPS_correct.hpp"
#include "tools.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <map>
#include <optional>
#include <string>
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

    const std::string& type() const;
    double p2t(const Point& p) const;
    Point t2p(double t) const;
    double dist(const Point& p1, const Point& p2) const;
    Point move(const Point& p, double d) const;
};

Polyline constructPolyline(DirectionField& df, const Point& p, const Point& dir, int QRsize, double line_len, int n, int kernel = 35);
std::vector<Point> PolylineIntersection(const Polyline& cur1, const Polyline& cur2);
std::pair<Polyline, Polyline> constructPolylines4recdist(DirectionField& df, const Points& points, int n);

std::pair<Points, Points> vectorfield_nodes(DirectionField& df,
                                            const Points& points,
                                            const std::vector<Points>& pattern,
                                            int QRsize = 25,
                                            int square_size = 10,
                                            int fract = 1,
                                            int lines_fract = 5,
                                            const TPS* tps = nullptr);

} // namespace qrscanner
