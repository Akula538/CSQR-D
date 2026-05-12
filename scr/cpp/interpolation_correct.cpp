#include "interpolation_correct.hpp"

#include "math_base.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace qrscanner {

static bool solve_mat(const cv::Mat& A, const cv::Mat& b, cv::Mat& x)
{
    return cv::solve(A, b, x, cv::DECOMP_LU) || cv::solve(A, b, x, cv::DECOMP_SVD);
}

Cubic::Cubic(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
{
    cv::Mat A = (cv::Mat_<double>(4, 4) << std::pow(x1, 3), x1 * x1, x1, 1.0,
                                           std::pow(x2, 3), x2 * x2, x2, 1.0,
                                           std::pow(x3, 3), x3 * x3, x3, 1.0,
                                           std::pow(x4, 3), x4 * x4, x4, 1.0);
    cv::Mat b = (cv::Mat_<double>(4, 1) << y1, y2, y3, y4);
    cv::Mat x;
    if (solve_mat(A, b, x))
        s_ = cv::Vec4d(x.at<double>(0), x.at<double>(1), x.at<double>(2), x.at<double>(3));
}

double Cubic::predict(double x) const
{
    return s_[0] * x * x * x + s_[1] * x * x + s_[2] * x + s_[3];
}

Spline::Spline(double x1, double y1, double d1, double x2, double y2, double d2)
{
    cv::Mat A = (cv::Mat_<double>(4, 4) << std::pow(x1, 3), x1 * x1, x1, 1.0,
                                           std::pow(x2, 3), x2 * x2, x2, 1.0,
                                           3.0 * x1 * x1, 2.0 * x1, 1.0, 0.0,
                                           3.0 * x2 * x2, 2.0 * x2, 1.0, 0.0);
    cv::Mat b = (cv::Mat_<double>(4, 1) << y1, y2, d1, d2);
    cv::Mat x;
    if (solve_mat(A, b, x))
        s_ = cv::Vec4d(x.at<double>(0), x.at<double>(1), x.at<double>(2), x.at<double>(3));
}

double Spline::predict(double x) const
{
    return s_[0] * x * x * x + s_[1] * x * x + s_[2] * x + s_[3];
}

Conic::Conic(double x1, double y1, double x2, double y2, double x3, double y3)
    : mx_(std::max({x1, x2, x3}))
{
    cv::Mat A = (cv::Mat_<double>(3, 3) << x1 * x1, x1, 1.0,
                                           x2 * x2, x2, 1.0,
                                           x3 * x3, x3, 1.0);
    cv::Mat b = (cv::Mat_<double>(3, 1) << y1, y2, y3);
    cv::Mat x;
    if (solve_mat(A, b, x))
        s_ = cv::Vec3d(x.at<double>(0), x.at<double>(1), x.at<double>(2));
}

double Conic::predict(double x, bool line_con) const
{
    if (x > mx_ && line_con) {
        const double k = 2.0 * s_[0] * mx_ + s_[1];
        return x * k + predict(mx_, false) - mx_ * k;
    }
    return s_[0] * x * x + s_[1] * x + s_[2];
}

Linear::Linear(double x1, double y1, double x2, double y2)
{
    cv::Mat A = (cv::Mat_<double>(2, 2) << x1, 1.0, x2, 1.0);
    cv::Mat b = (cv::Mat_<double>(2, 1) << y1, y2);
    cv::Mat x;
    if (solve_mat(A, b, x))
        s_ = cv::Vec2d(x.at<double>(0), x.at<double>(1));
}

double Linear::predict(double x) const
{
    return s_[0] * x + s_[1];
}

Quintic::Quintic(const std::vector<double>& x, const std::vector<double>& y)
{
    cv::Mat A = cv::Mat::zeros(6, 6, CV_64F);
    cv::Mat b = cv::Mat::zeros(6, 1, CV_64F);
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c)
            A.at<double>(r, c) = std::pow(x[r], 5 - c);
        b.at<double>(r, 0) = y[r];
    }
    cv::Mat sol;
    if (solve_mat(A, b, sol)) {
        for (int i = 0; i < 6; ++i)
            s_[i] = sol.at<double>(i);
    }
}

double Quintic::predict(double x) const
{
    double y = 0.0;
    for (int i = 0; i < 6; ++i)
        y += s_[i] * std::pow(x, 5 - i);
    return y;
}

static double circle_metric_max(const std::array<Point, 4>& points)
{
    double scale = 0.0;
    int cnt = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j)
                continue;
            scale += norm(points[i] - points[j]);
            ++cnt;
        }
    }
    scale = cnt ? scale / cnt : 1.0;

    cv::Mat A = cv::Mat::zeros(4, 3, CV_64F);
    cv::Mat b = cv::Mat::zeros(4, 1, CV_64F);
    for (int i = 0; i < 4; ++i) {
        A.at<double>(i, 0) = points[i].x;
        A.at<double>(i, 1) = points[i].y;
        A.at<double>(i, 2) = 1.0;
        b.at<double>(i, 0) = -(points[i].x * points[i].x + points[i].y * points[i].y);
    }
    cv::Mat sol;
    cv::solve(A, b, sol, cv::DECOMP_SVD);
    const double a = sol.at<double>(0);
    const double bb = sol.at<double>(1);
    const double c = sol.at<double>(2);
    const double xc = -a / 2.0;
    const double yc = -bb / 2.0;
    const double R = std::sqrt(std::max(0.0, (a * a + bb * bb) / 4.0 - c));

    double max_d = 0.0;
    for (const auto& p : points)
        max_d = std::max(max_d, std::abs(norm(p - Point(xc, yc)) - R));

    return max_d / std::max(scale, 1e-12);
}

Curve::Curve(const Point& p1, const Point& p2, const Point& p3, const Point& p4)
{
    if (!std::isnan(p4.x)) {
        std::array<Point, 4> pts = {p1, p2, p3, p4};
        if (circle_metric_max(pts) > 0.015) {
            try {
                auto fit = fit_parabola_from4(pts);
                type_ = "parabola";
                A_ = fit.A;
                B_ = fit.B;
                C_ = fit.C;
                return;
            } catch (...) {
            }
        }
    }

    if (collinear(p1, p2, p3)) {
        type_ = "line";
        P0_ = p1;
        N_ = normalize(p2 - p1);
    } else {
        type_ = "circle";
        cv::Mat A = (cv::Mat_<double>(2, 2) << 2.0 * (p1.x - p2.x), 2.0 * (p1.y - p2.y),
                                               2.0 * (p1.x - p3.x), 2.0 * (p1.y - p3.y));
        cv::Mat b = (cv::Mat_<double>(2, 1) << p1.x * p1.x - p2.x * p2.x + p1.y * p1.y - p2.y * p2.y,
                                               p1.x * p1.x - p3.x * p3.x + p1.y * p1.y - p3.y * p3.y);
        cv::Mat sol;
        if (!solve_mat(A, b, sol))
            throw std::runtime_error("Failed to construct circle curve");
        O_ = {sol.at<double>(0), sol.at<double>(1)};
        R_ = norm(O_ - p1);
    }
}

bool Curve::collinear(const Point& p1, const Point& p2, const Point& p3) const
{
    const double area = std::abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y));
    return area < 1e-3;
}

double Curve::I(double t) const
{
    const Point r = C_ * (2.0 * t) + B_;
    const double c_norm = norm(C_);
    const double r_norm = norm(r);
    const double cb_cross = cross(C_, B_);
    return (r.dot(C_) * r_norm * c_norm + cb_cross * cb_cross * std::log(r_norm * c_norm + r.dot(C_))) /
           (4.0 * std::pow(c_norm, 3));
}

double Curve::p2t(const Point& p) const
{
    double t = cross(p - A_, C_) / cross(B_, C_);

    auto f = [&](double tt) {
        return norm(A_ + B_ * tt + C_ * tt * tt - p);
    };

    const double eps = 1e-6;
    double derivative = (f(t + eps) - f(t)) / eps;
    int i = 10;
    while (f(t) > 0.3 && i > 0) {
        --i;
        if (std::abs(derivative) < 1e-12)
            break;
        t = t - f(t) / derivative;
        derivative = (f(t + eps) - f(t)) / eps;
    }
    return t;
}

double Curve::dist(const Point& p1, const Point& p2) const
{
    if (type_ == "parabola")
        return I(p2t(p1)) - I(p2t(p2));

    if (type_ == "line")
        return (p1 - P0_).dot(N_) - (p2 - P0_).dot(N_);

    Point q1 = normalize(p1 - O_);
    Point q2 = normalize(p2 - O_);
    double a1 = std::atan2(q1.y, q1.x);
    double a2 = std::atan2(q2.y, q2.x);
    double a = std::fmod(a1 - a2, 2.0 * CV_PI);
    if (a < 0)
        a += 2.0 * CV_PI;
    if (a > CV_PI)
        a -= 2.0 * CV_PI;
    return a * R_;
}

Point Curve::move(const Point& p, double d) const
{
    if (type_ == "parabola") {
        double y = p2t(p);
        const double d0 = I(y);
        const double eps = 1e-3;
        for (int i = 0; i < 100; ++i) {
            const double cur = I(y) - d0 - d;
            if (std::abs(cur) < eps)
                break;
            const double denom = norm(C_ * (2.0 * y) + B_);
            if (denom == 0.0)
                break;
            y -= cur / denom;
        }
        return C_ * y * y + B_ * y + A_;
    }

    if (type_ == "line") {
        const double x = (p - P0_).dot(N_);
        return P0_ + N_ * (x + d);
    }

    double angle = std::atan2((p - O_).y, (p - O_).x);
    angle += d / R_;
    return Point(std::cos(angle), std::sin(angle)) * R_ + O_;
}

std::vector<Point> CurveIntersection(const Curve& cur1, const Curve& cur2)
{
    if (cur1.type() == "circle" && cur2.type() == "circle") {
        const Point p1 = cur1.O();
        const Point p2 = cur2.O();
        const double r1 = cur1.R();
        const double r2 = cur2.R();
        const Point v = p2 - p1;
        const double d = norm(v);
        if (d > r1 + r2 || d < std::abs(r1 - r2) || d == 0.0)
            return {};
        const double a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
        const double h = std::sqrt(std::max(0.0, r1 * r1 - a * a));
        const Point p0 = p1 + v * (a / d);
        const Point v_perp(-v.y * h / d, v.x * h / d);
        return {p0 + v_perp, p0 - v_perp};
    }

    if (cur1.type() == "line" && cur2.type() == "circle") {
        const Point P0 = cur1.P0();
        const Point N = cur1.N();
        const Point O = cur2.O();
        const double R = cur2.R();
        const Point v = O - P0;
        const double proj = v.dot(N);
        const Point perp = v - N * proj;
        const double dist_to_line = norm(perp);
        if (dist_to_line > R)
            return {};
        const double half_chord = std::sqrt(std::max(0.0, R * R - dist_to_line * dist_to_line));
        const Point closest = P0 + N * proj;
        return {closest - N * half_chord, closest + N * half_chord};
    }

    if (cur1.type() == "circle" && cur2.type() == "line")
        return CurveIntersection(cur2, cur1);

    if (cur1.type() == "line" && cur2.type() == "line") {
        if (std::abs(cur1.N().dot(cur2.N())) > 1.0 - 1e-10)
            return {};
        cv::Mat A = (cv::Mat_<double>(2, 2) << cur1.N().x, -cur2.N().x, cur1.N().y, -cur2.N().y);
        cv::Mat b = (cv::Mat_<double>(2, 1) << cur2.P0().x - cur1.P0().x, cur2.P0().y - cur1.P0().y);
        cv::Mat sol;
        cv::solve(A, b, sol, cv::DECOMP_LU);
        return {cur1.P0() + cur1.N() * sol.at<double>(0)};
    }

    throw std::runtime_error("Intersection is not defined for this Curve type");
}

std::vector<Points> recover_pattern_if_need(const Points& points,
                                            int QRsize,
                                            const std::vector<Points*>& pattern_in,
                                            const std::vector<Points*>& search_pattern_in)
{
    std::vector<Points> pattern(2);
    if (pattern_in.size() > 0 && pattern_in[0])
        pattern[0] = *pattern_in[0];
    if (pattern_in.size() > 1 && pattern_in[1])
        pattern[1] = *pattern_in[1];

    if (pattern[0].empty()) {
        Curve cur(points[3], points[2], points[6]);
        Cubic cub(0, cur.dist(points[3], points[3]),
                  7, cur.dist(points[2], points[3]),
                  QRsize - 7, cur.dist(points[7], points[3]),
                  QRsize, cur.dist(points[6], points[3]));
        for (int i = 8; i < QRsize - 7; ++i)
            pattern[0].push_back(cur.move(points[3], cub.predict(i)));
    }

    if (pattern[1].empty()) {
        Curve cur(points[1], points[2], points[14]);
        Cubic cub(0, cur.dist(points[1], points[1]),
                  7, cur.dist(points[2], points[1]),
                  QRsize - 7, cur.dist(points[13], points[1]),
                  QRsize, cur.dist(points[14], points[1]));
        for (int i = 8; i < QRsize - 7; ++i)
            pattern[1].push_back(cur.move(points[1], cub.predict(i)));
    }

    std::vector<Points> search_pattern(4);
    for (int i = 0; i < 4; ++i) {
        if (search_pattern_in.size() > static_cast<size_t>(i) && search_pattern_in[i])
            search_pattern[i] = *search_pattern_in[i];
    }

    if (search_pattern[0].empty())
        for (int i = 0; i < 8; ++i) search_pattern[0].push_back((points[3] * (7 - i) + points[2] * i) * (1.0 / 7.0));
    if (search_pattern[1].empty())
        for (int i = 0; i < 8; ++i) search_pattern[1].push_back((points[7] * (7 - i) + points[6] * i) * (1.0 / 7.0));
    if (search_pattern[2].empty())
        for (int i = 0; i < 8; ++i) search_pattern[2].push_back((points[1] * (7 - i) + points[2] * i) * (1.0 / 7.0));
    if (search_pattern[3].empty())
        for (int i = 0; i < 8; ++i) search_pattern[3].push_back((points[13] * (7 - i) + points[14] * i) * (1.0 / 7.0));

    Points p0;
    p0.reserve(search_pattern[0].size() + pattern[0].size() + search_pattern[1].size());
    p0.insert(p0.end(), search_pattern[0].begin(), search_pattern[0].end());
    p0.insert(p0.end(), pattern[0].begin(), pattern[0].end());
    p0.insert(p0.end(), search_pattern[1].begin(), search_pattern[1].end());
    pattern[0] = std::move(p0);

    Points p1;
    p1.reserve(search_pattern[2].size() + pattern[1].size() + search_pattern[3].size());
    p1.insert(p1.end(), search_pattern[2].begin(), search_pattern[2].end());
    p1.insert(p1.end(), pattern[1].begin(), pattern[1].end());
    p1.insert(p1.end(), search_pattern[3].begin(), search_pattern[3].end());
    pattern[1] = std::move(p1);

    return pattern;
}

} // namespace qrscanner
