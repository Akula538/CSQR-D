#include "dm_correct.hpp"
#include "DMTrace.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace datamatrix::geometry {
using qrscanner::norm;
using qrscanner::normalize;
static std::vector<int> stations(int n, int spacing)
{
    std::vector<int> out;
    int intervals = std::max(1, (n - 1 + spacing - 1) / spacing);
    for (int i = 0; i <= intervals; ++i)
        out.push_back(cvRound(double(i) * (n - 1) / intervals));
    return out;
}
static Point clock_point(const Candidate& c, const Grid& g, int axis, int i)
{
    auto& centers = axis ? g.y.centers : g.x.centers;
    auto& points = axis ? g.y.points : g.x.points;
    if (!points.empty())
        return points[i];
    return inset_point(c, axis, centers[i], int(axis ? g.x.centers.size() : g.y.centers.size()));
}
std::pair<Points, Points> boundary_nodes(const Candidate& c, const Grid& g)
{
    int w = int(g.x.centers.size()), h = int(g.y.centers.size());
    Points ideal, source;
    for (int x : stations(w, 4)) {
        double t = g.x.centers[x];
        ideal.emplace_back(x + .5, .5);
        source.push_back(clock_point(c, g, 0, x));
        ideal.emplace_back(x + .5, h - .5);
        source.push_back(inset_point(c, 2, 1 - t, h));
    }
    for (int y : stations(h, 4)) {
        double t = g.y.centers[y];
        ideal.emplace_back(w - .5, y + .5);
        source.push_back(clock_point(c, g, 1, y));
        ideal.emplace_back(.5, y + .5);
        source.push_back(inset_point(c, 3, 1 - t, w));
    }
    // Remove repeated ideal corner constraints before solving TPS.
    Points a, b;
    for (size_t i = 0; i < ideal.size(); ++i)
        if (std::find(a.begin(), a.end(), ideal[i]) == a.end()) {
            a.push_back(ideal[i]);
            b.push_back(source[i]);
        }
    return {a, b};
}

std::pair<Points, Points> field_nodes(qrscanner::DirectionField& field, const Candidate& c, const Grid& g,
                                      int spacing, int steps)
{
    int w = int(g.x.centers.size()), h = int(g.y.centers.size());
    auto xs = stations(w, spacing), ys = stations(h, spacing);
    std::array<double, 4> spans;
    for (int i = 0; i < 4; ++i)
        spans[i] = c.edges[i].arc_length();
    double pitch = (c.edges[0].length / w + c.edges[1].length / h) * .5;
    int kernel = std::max(35, int(pitch * std::max(3, spacing - 1))) | 1;
    std::vector<qrscanner::Polyline> columns, rows;
    for (int x : xs) {
        double t = g.x.centers[x];
        Point direction = -c.edges[3].direction(1) * (1 - t) + c.edges[1].direction(0) * t;
        columns.push_back(qrscanner::constructPolyline(field, clock_point(c, g, 0, x), direction, h,
                                                       spans[3] * (1 - t) + spans[1] * t, steps, kernel));
    }
    for (int y : ys) {
        double t = g.y.centers[y];
        Point direction = -c.edges[0].direction(1) * (1 - t) + c.edges[2].direction(0) * t;
        rows.push_back(qrscanner::constructPolyline(field, clock_point(c, g, 1, y), direction, w,
                                                    spans[0] * (1 - t) + spans[2] * t, steps, kernel));
    }
#ifdef CSQR_DM_TRACE
    diagnostics::Lines lines;
    for (auto& p : columns)
        lines.push_back(p.points);
    for (auto& p : rows)
        lines.push_back(p.points);
    DM_TRACE("field_lines", {}, {{"w", w}, {"h", h}, {"spacing", spacing}}, {}, lines);
#endif
    auto boundary = boundary_nodes(c, g);
    qrscanner::TPS prior;
    prior.fit(boundary.first, boundary.second);
    Points ideal, source;
    for (size_t y = 0; y < ys.size(); ++y)
        for (size_t x = 0; x < xs.size(); ++x) {
            Point target(xs[x] + .5, ys[y] + .5), predicted = prior.tps_transform({target})[0];
            auto hits = qrscanner::PolylineIntersection(columns[x], rows[y]);
            Point hit = predicted;
            if (!hits.empty())
                hit = *std::min_element(hits.begin(), hits.end(),
                                        [&](auto a, auto b) { return norm(a - predicted) < norm(b - predicted); });
            if (norm(hit - predicted) > pitch * 3)
                hit = predicted;
            ideal.push_back(target);
            source.push_back(hit);
        }
    // A collapsed or inverted cell is evidence of line merging. Never fit a
    // TPS through it: that would silently erase the modules between the lines.
    for (size_t y = 1; y < ys.size(); ++y)
        for (size_t x = 1; x < xs.size(); ++x) {
            auto p = source[(y - 1) * xs.size() + x - 1];
            auto a = source[(y - 1) * xs.size() + x] - p, b = source[y * xs.size() + x - 1] - p;
            if (qrscanner::cross(a, b) < pitch * pitch * (xs[x] - xs[x - 1]) * (ys[y] - ys[y - 1]) * .12) {
                DM_TRACE("field_rejected", {}, {}, {{"reason", "collapsed cell"}});
                return {};
            }
        }
#ifdef CSQR_DM_TRACE
    DM_TRACE("field_nodes", {}, {{"w", w}, {"h", h}, {"spacing", spacing}}, {}, {}, source);
#endif
    return {ideal, source};
}

std::pair<Points, Points> envelope_nodes(const Candidate& c, const Grid& g)
{
    const int w = int(g.x.centers.size()), h = int(g.y.centers.size());
    auto smooth = [](const Clock& clock) {
        int n = int(clock.centers.size());
        cv::Mat a(n, 4, CV_64F), b(n, 1, CV_64F), coefficients;
        for (int i = 0; i < n; ++i) {
            double t = 2 * (i + .5) / n - 1, power = 1;
            for (int j = 0; j < 4; ++j) {
                a.at<double>(i, j) = power;
                power *= t;
            }
            b.at<double>(i) = clock.centers[i];
        }
        cv::solve(a, b, coefficients, cv::DECOMP_SVD);
        return coefficients;
    };
    auto cx = smooth(g.x), cy = smooth(g.y);
    auto phase = [](const cv::Mat& coefficients, double t) {
        double z = 2 * t - 1, value = 0;
        for (int i = 3; i >= 0; --i)
            value = value * z + coefficients.at<double>(i);
        return value;
    };
    Points ideal, source;
    Point tl = c.edges[0].at(0), tr = c.edges[0].at(1), br = c.edges[2].at(0), bl = c.edges[2].at(1);
    for (int y : stations(h, 3))
        for (int x : stations(w, 3)) {
            double u = phase(cx, (x + .5) / w), v = phase(cy, (y + .5) / h);
            Point p = c.edges[0].at(u) * (1 - v) + c.edges[2].at(1 - u) * v + c.edges[3].at(1 - v) * (1 - u) +
                      c.edges[1].at(v) * u -
                      (tl * ((1 - u) * (1 - v)) + tr * (u * (1 - v)) + br * (u * v) + bl * ((1 - u) * v));
            ideal.emplace_back(x + .5, y + .5);
            source.push_back(p);
        }
    return {ideal, source};
}

static cv::Mat source_slopes(qrscanner::DirectionField& field, const qrscanner::TPS& chart, int w, int h,
                             double pitch, int samples)
{
    // Missing directions retain the current lattice as a zero-slope prior.
    // Measurements are taken in source pixels; only the vectors are expressed
    // in the current lattice basis through its inverse Jacobian.
    cv::Mat slopes(samples, samples, CV_64FC2, cv::Scalar(0, 0));
    int kernel = std::max(35, int(4 * pitch)) | 1;
    for (int y = 0; y < samples; ++y)
        for (int x = 0; x < samples; ++x) {
            Point q(double(x) * w / (samples - 1), double(y) * h / (samples - 1));
            auto p = chart.tps_transform({q, q + Point(.05, 0), q + Point(0, .05)});
            Point a = (p[1] - p[0]) / .05, b = (p[2] - p[0]) / .05;
            double det = qrscanner::cross(a, b);
            if (det <= .1 * pitch * pitch)
                return {};
            auto angles = field.get(p[0].x, p[0].y, kernel);
            double horizontal = .65, vertical = .65;
            for (auto angle : {angles.first, angles.second}) {
                if (!angle)
                    continue;
                double theta = (*angle + 90) * CV_PI / 180;
                Point direction(std::cos(theta), std::sin(theta));
                double du = qrscanner::cross(direction, b) / det;
                double dv = qrscanner::cross(a, direction) / det;
                if (std::abs(dv) < horizontal * std::abs(du)) {
                    horizontal = std::abs(dv / du);
                    slopes.at<cv::Vec2d>(y, x)[0] = dv / du;
                }
                if (std::abs(du) < vertical * std::abs(dv)) {
                    vertical = std::abs(du / dv);
                    slopes.at<cv::Vec2d>(y, x)[1] = du / dv;
                }
            }
        }
    return slopes;
}

static std::pair<Points, Points> solve_lattice(const cv::Mat& gray, const Grid& g,
                                               const std::pair<Points, Points>& nodes, int mesh,
                                               double* maximum_shift, const std::pair<Points, Points>* anchors,
                                               qrscanner::DirectionField* supplied_field)
{
    if (nodes.first.empty() || mesh < 2 || mesh > 9)
        return {};
    int w = int(g.x.centers.size()), h = int(g.y.centers.size());
    qrscanner::TPS mapping;
    mapping.fit(nodes.first, nodes.second);
    cv::Mat slopes;
    const bool fit_directions = supplied_field != nullptr;
    if (fit_directions) {
        auto p =
            mapping.tps_transform({Point(w * .5, h * .5), Point(w * .5 + 1, h * .5), Point(w * .5, h * .5 + 1)});
        double pitch = (norm(p[1] - p[0]) + norm(p[2] - p[0])) * .5;
        slopes = source_slopes(*supplied_field, mapping, w, h, pitch, 9);
    }
    int count = mesh * mesh;
    auto basis = [&](Point p) {
        std::array<double, 81> weights{};
        double x = std::clamp(p.x / w * (mesh - 1), 0., mesh - 1.000001);
        double y = std::clamp(p.y / h * (mesh - 1), 0., mesh - 1.000001);
        int ix = int(x), iy = int(y);
        for (int dy = 0; dy <= 1; ++dy)
            for (int dx = 0; dx <= 1; ++dx)
                weights[(iy + dy) * mesh + ix + dx] = (dx ? x - ix : 1 - x + ix) * (dy ? y - iy : 1 - y + iy);
        return weights;
    };
    auto intensity = [&](Point p) {
        double x = std::clamp(p.x, 0., gray.cols - 1.000001), y = std::clamp(p.y, 0., gray.rows - 1.000001);
        int ix = int(x), iy = int(y);
        double dx = x - ix, dy = y - iy;
        return (gray.at<uchar>(iy, ix) * (1 - dx) + gray.at<uchar>(iy, ix + 1) * dx) * (1 - dy) +
               (gray.at<uchar>(iy + 1, ix) * (1 - dx) + gray.at<uchar>(iy + 1, ix + 1) * dx) * dy;
    };
    // Joint least squares: smooth displacement, measured boundary anchors,
    // local source directions and submodule photometric transition positions.
    std::array<cv::Mat, 2> displacements;
    for (int axis = 0; axis < 2; ++axis) {
        cv::Mat normal = cv::Mat::eye(count, count, CV_64F) * .15;
        cv::Mat rhs = cv::Mat::zeros(count, 1, CV_64F);
        auto equation = [&](const std::array<double, 81>& a, double target, double weight) {
            for (int i = 0; i < count; ++i) {
                if (a[i] == 0)
                    continue;
                rhs.at<double>(i) += weight * a[i] * target;
                for (int j = 0; j < count; ++j)
                    if (a[j] != 0)
                        normal.at<double>(i, j) += weight * a[i] * a[j];
            }
        };
        for (int y = 0; y < mesh; ++y)
            for (int x = 0; x < mesh; ++x)
                for (int direction = 0; direction < 2; ++direction) {
                    if ((direction == 0 && x + 1 == mesh) || (direction == 1 && y + 1 == mesh))
                        continue;
                    std::array<double, 81> a{};
                    a[y * mesh + x] = 1;
                    a[(y + direction) * mesh + x + 1 - direction] = -1;
                    equation(a, 0, mesh <= 5 ? 2 : .5);
                }
        if (anchors) {
            for (size_t i = 0; i < anchors->first.size(); ++i) {
                Point q = anchors->first[i];
                auto p = mapping.tps_transform({q, q + Point(.05, 0), q + Point(0, .05)});
                Point u = (p[1] - p[0]) / .05, v = (p[2] - p[0]) / .05;
                double det = qrscanner::cross(u, v);
                if (det <= 1e-6)
                    continue;
                Point error = anchors->second[i] - p[0];
                Point delta(qrscanner::cross(error, v) / det, qrscanner::cross(u, error) / det);
                double weight = (q.y < 1 || q.x > w - 1) ? 8 : 2;
                weight /= std::sqrt(1 + delta.dot(delta));
                equation(basis(q), axis ? delta.y : delta.x, weight);
            }
        }
        if (!slopes.empty()) {
            for (int sy = 0; sy < slopes.rows; ++sy)
                for (int sx = 0; sx < slopes.cols; ++sx) {
                    Point p(double(sx) * (w - 1) / (slopes.cols - 1) + .5,
                            double(sy) * (h - 1) / (slopes.rows - 1) + .5);
                    Point d = axis == 0 ? Point(0, .5) : Point(.5, 0);
                    auto plus = basis(p + d), minus = basis(p - d);
                    for (int i = 0; i < count; ++i)
                        plus[i] -= minus[i];
                    equation(plus, slopes.at<cv::Vec2d>(sy, sx)[1 - axis], 32.);
                }
        }
        for (int y = fit_directions ? (axis ? 1 : 0) : 1; y < (fit_directions ? h : h - 1); ++y)
            for (int x = fit_directions ? (axis ? 0 : 1) : 1; x < (fit_directions ? w : w - 1); ++x) {
                Point q(x + (axis ? .5 : 0), y + (axis ? 0 : .5));
                Point d = axis ? Point(0, 1) : Point(1, 0);
                constexpr int half = 12;
                auto positions = mapping.tps_transform({q - d * .6, q, q + d * .6});
                std::array<double, 2 * half + 1> values{};
                for (int i = 0; i <= 2 * half; ++i) {
                    double t = double(i - half) / half;
                    Point p = t < 0 ? positions[1] * (1 + t) - positions[0] * t
                                    : positions[1] * (1 - t) + positions[2] * t;
                    values[i] = intensity(p);
                }
                double contrast = values.back() - values.front();
                if (std::abs(contrast) < 35)
                    continue;
                double total = 0, moment = 0;
                for (int i = 2; i < 2 * half - 1; ++i) {
                    double gradient = (values[i + 2] - values[i - 2]) * (contrast > 0 ? 1 : -1);
                    gradient = std::max(0., gradient - 4);
                    total += gradient;
                    moment += gradient * (i - half) * .05;
                }
                if (total > 30)
                    equation(basis(q), moment / total, std::min(1., std::abs(contrast) / 128));
            }
        cv::solve(normal, rhs, displacements[axis], cv::DECOMP_CHOLESKY);
    }
    Points ideal, queries;
    double largest_shift = 0;
    for (int y : stations(h, 3))
        for (int x : stations(w, 3)) {
            Point p(x + .5, y + .5), shift;
            auto weights = basis(p);
            for (int i = 0; i < count; ++i) {
                shift.x += weights[i] * displacements[0].at<double>(i);
                shift.y += weights[i] * displacements[1].at<double>(i);
            }
            largest_shift = std::max(largest_shift, cv::norm(shift));
            ideal.push_back(p);
            queries.push_back(p + shift);
        }
    auto source = mapping.tps_transform(queries);
    if (maximum_shift)
        *maximum_shift = largest_shift;
    DM_TRACE("aligned_nodes", {}, {{"w", w}, {"h", h}, {"max_shift", largest_shift}}, {}, {}, source);
    return {ideal, source};
}
std::pair<Points, Points> align_lattice(const cv::Mat& gray, const Grid& grid,
                                        const std::pair<Points, Points>& nodes, int mesh, double* maximum_shift)
{
    return solve_lattice(gray, grid, nodes, mesh, maximum_shift, nullptr, nullptr);
}

std::pair<Points, Points> refine_lattice(const cv::Mat& gray, const Candidate& candidate, const Grid& grid,
                                         const std::pair<Points, Points>& nodes, qrscanner::DirectionField& field)
{
    auto anchors = boundary_nodes(candidate, grid);
    return solve_lattice(gray, grid, nodes, 9, nullptr, &anchors, &field);
}
} // namespace datamatrix::geometry
