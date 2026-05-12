#pragma once

#include "tools.hpp"

#include <array>
#include <string>
#include <vector>

namespace qrscanner {

class Cubic {
public:
    Cubic() = default;
    Cubic(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);
    double predict(double x) const;

private:
    cv::Vec4d s_ = {};
};

class Spline {
public:
    Spline() = default;
    Spline(double x1, double y1, double d1, double x2, double y2, double d2);
    double predict(double x) const;

private:
    cv::Vec4d s_ = {};
};

class Conic {
public:
    Conic() = default;
    Conic(double x1, double y1, double x2, double y2, double x3, double y3);
    double predict(double x, bool line_con = true) const;

private:
    cv::Vec3d s_ = {};
    double mx_ = 0.0;
};

class Linear {
public:
    Linear() = default;
    Linear(double x1, double y1, double x2, double y2);
    double predict(double x) const;

private:
    cv::Vec2d s_ = {};
};

class Quintic {
public:
    Quintic() = default;
    Quintic(const std::vector<double>& x, const std::vector<double>& y);
    double predict(double x) const;

private:
    std::array<double, 6> s_ = {};
};

class Curve {
public:
    Curve() = default;
    Curve(const Point& p1, const Point& p2, const Point& p3, const Point& p4 = Point(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()));

    const std::string& type() const { return type_; }
    double dist(const Point& p1, const Point& p2) const;
    Point move(const Point& p, double d) const;

    Point P0() const { return P0_; }
    Point N() const { return N_; }
    Point O() const { return O_; }
    double R() const { return R_; }

private:
    bool collinear(const Point& p1, const Point& p2, const Point& p3) const;
    double I(double t) const;
    double p2t(const Point& p) const;

    std::string type_ = "line";
    Point P0_;
    Point N_;
    Point O_;
    double R_ = 0.0;
    Point A_;
    Point B_;
    Point C_;
};

std::vector<Point> CurveIntersection(const Curve& cur1, const Curve& cur2);
std::vector<Points> recover_pattern_if_need(const Points& points,
                                            int QRsize,
                                            const std::vector<Points*>& pattern,
                                            const std::vector<Points*>& search_pattern);

} // namespace qrscanner
