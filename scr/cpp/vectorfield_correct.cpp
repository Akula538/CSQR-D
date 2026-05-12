#include "vectorfield_correct.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

namespace qrscanner {

DirectionField::DirectionField(const cv::Mat& img)
{
    cv::Mat gray;
    if (img.channels() == 1)
        gray = img.clone();
    else
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(gray, gray, cv::Size(1, 1), 0);
    cv::Mat sobel_x, sobel_y;
    cv::Sobel(gray, sobel_x, CV_64F, 1, 0, 7);
    cv::Sobel(gray, sobel_y, CV_64F, 0, 1, 7);
    cv::magnitude(sobel_x, sobel_y, magnitude_);
    cv::phase(sobel_x, sobel_y, orientation_, false);
}

static double percentile85(const std::vector<double>& values)
{
    if (values.empty())
        return 0.0;
    std::vector<double> v = values;
    size_t idx = static_cast<size_t>(std::floor(0.85 * (v.size() - 1)));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

static std::vector<double> gaussian_kernel(double sigma)
{
    const int kernel_size = static_cast<int>(2 * 3 * sigma) + 1;
    std::vector<double> kernel;
    double sum = 0.0;
    for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i) {
        double val = std::exp(-(i * i) / (2.0 * sigma * sigma));
        kernel.push_back(val);
        sum += val;
    }
    for (double& v : kernel)
        v /= sum;
    return kernel;
}

static std::vector<double> convolve_wrap(const std::vector<double>& src, const std::vector<double>& kernel)
{
    const int n = static_cast<int>(src.size());
    const int k = static_cast<int>(kernel.size());
    const int half = k / 2;
    std::vector<double> out(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < k; ++j) {
            int idx = (i + j - half) % n;
            if (idx < 0)
                idx += n;
            s += src[idx] * kernel[j];
        }
        out[i] = s;
    }
    return out;
}

static double circ_dist(double a, double b)
{
    double d = std::fmod(std::abs(a - b), 180.0);
    return std::min(d, 180.0 - d);
}

std::pair<std::optional<double>, std::optional<double>> DirectionField::get(double x_, double y_, int kernel)
{
    const int x = static_cast<int>(x_);
    const int y = static_cast<int>(y_);
    const auto key = std::make_tuple(x, y, kernel);
    auto it = cache_.find(key);
    if (it != cache_.end())
        return it->second;

    const int xm = magnitude_.cols;
    const int ym = magnitude_.rows;
    const int x1 = std::max(0, x - kernel / 2);
    const int y1 = std::max(0, y - kernel / 2);
    const int x2 = std::min(xm - 1, x - kernel / 2 + kernel);
    const int y2 = std::min(ym - 1, y - kernel / 2 + kernel);

    std::vector<double> mags;
    for (int yy = y1; yy < y2; ++yy) {
        const double* row = magnitude_.ptr<double>(yy);
        for (int xx = x1; xx < x2; ++xx)
            mags.push_back(row[xx]);
    }
    const double threshold = percentile85(mags);

    std::vector<double> hist(180, 0.0);
    for (int yy = y1; yy < y2; ++yy) {
        const double* mag = magnitude_.ptr<double>(yy);
        const double* ori = orientation_.ptr<double>(yy);
        for (int xx = x1; xx < x2; ++xx) {
            if (mag[xx] <= threshold)
                continue;
            double deg = std::fmod(ori[xx] * 180.0 / CV_PI, 180.0);
            if (deg < 0)
                deg += 180.0;
            int bin = std::min(179, std::max(0, static_cast<int>(std::floor(deg))));
            hist[bin] += mag[xx];
        }
    }

    auto smooth = convolve_wrap(hist, gaussian_kernel(2.0));
    const double max_val = *std::max_element(smooth.begin(), smooth.end());
    if (max_val <= 0.0) {
        cache_[key] = {std::nullopt, std::nullopt};
        return cache_[key];
    }

    std::vector<std::pair<double, int>> peaks;
    for (int i = 0; i < 180; ++i) {
        double prev = smooth[(i + 179) % 180];
        double next = smooth[(i + 1) % 180];
        if (smooth[i] >= 0.1 * max_val && smooth[i] >= prev && smooth[i] >= next)
            peaks.push_back({smooth[i], i});
    }

    std::sort(peaks.begin(), peaks.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::pair<double, int>> filtered;
    for (const auto& p : peaks) {
        bool ok = true;
        for (const auto& q : filtered) {
            if (circ_dist(p.second, q.second) < 25.0) {
                ok = false;
                break;
            }
        }
        if (ok)
            filtered.push_back(p);
    }

    if (filtered.empty()) {
        cache_[key] = {std::nullopt, std::nullopt};
    } else if (filtered.size() == 1) {
        cache_[key] = {static_cast<double>(filtered[0].second), std::nullopt};
    } else {
        cache_[key] = {static_cast<double>(filtered[0].second), static_cast<double>(filtered[1].second)};
    }
    return cache_[key];
}

static Points calc_integral_line(DirectionField& DF, Point p, Point dir0, double line_len, int n, int kernel = 35, int RK = 1)
{
    const double h = line_len / n;
    dir0 = normalize(dir0);

    auto get_vec = [&](const Point& pp, Point v) {
        v = normalize(v);
        auto [dir1opt, dir2opt] = DF.get(pp.x, pp.y, kernel);
        if (!dir1opt)
            return v;
        if (!dir2opt) {
            double dir1 = (*dir1opt + 90.0) * CV_PI / 180.0;
            Point v1(std::cos(dir1), std::sin(dir1));
            if (std::abs(v1.dot(v)) >= 0.85)
                return v1 * (v1.dot(v) >= 0.0 ? 1.0 : -1.0);
            return v;
        }

        double dir1 = (*dir1opt + 90.0) * CV_PI / 180.0;
        double dir2 = (*dir2opt + 90.0) * CV_PI / 180.0;
        Point v1(std::cos(dir1), std::sin(dir1));
        Point v2(std::cos(dir2), std::sin(dir2));
        if (std::abs(v1.dot(v)) > std::abs(v2.dot(v)))
            return v1 * (v1.dot(v) >= 0.0 ? 1.0 : -1.0);
        return v2 * (v2.dot(v) >= 0.0 ? 1.0 : -1.0);
    };

    Points points = {p};
    Point v = dir0;
    Point prev_v = v;
    for (int i = 0; i < n; ++i) {
        try {
            if (v.dot(dir0) < -0.5)
                break;
            Point predicted_v = (v.dot(dir0) > -0.26 && std::abs(prev_v.dot(dir0)) < 0.99) ? v * 2.0 - prev_v : v;
            Point k5;
            if (RK == 4) {
                Point k1 = get_vec(p, predicted_v);
                Point k2 = get_vec(p + k1 * (h / 2.0), predicted_v);
                Point k3 = get_vec(p + k2 * (h / 2.0), predicted_v);
                Point k4 = get_vec(p + k3 * h, predicted_v);
                k5 = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (1.0 / 6.0);
            } else if (RK == 3) {
                Point k1 = get_vec(p, predicted_v);
                Point k2 = get_vec(p + k1 * (h / 2.0), predicted_v);
                Point k3 = get_vec(p + k2 * h, predicted_v);
                k5 = (k1 + k2 * 4.0 + k3) * (1.0 / 6.0);
            } else if (RK == 2) {
                Point k1 = get_vec(p, predicted_v);
                k5 = get_vec(p + k1 * (h / 2.0), predicted_v);
            } else {
                k5 = get_vec(p, predicted_v);
            }
            Point p_next = p + normalize(k5) * h;
            points.push_back(p_next);
            prev_v = v;
            v = normalize(p_next - p);
            p = p_next;
        } catch (...) {
            points.push_back(p + normalize(dir0) * h);
            break;
        }
    }
    return points;
}

Polyline constructPolyline(DirectionField& df, const Point& p, const Point& dir, int QRsize, double line_len, int n, int kernel)
{
    Points pol1 = calc_integral_line(df, p, -dir, line_len * (6.0 / QRsize + 0.5), n * 6 / QRsize + n / 2, kernel);
    Points pol2 = calc_integral_line(df, p, dir, line_len * ((QRsize - 6.0) / QRsize + 0.5), n * (QRsize - 6) / QRsize + n / 2, kernel);
    std::reverse(pol1.begin(), pol1.end());
    if (!pol1.empty())
        pol1.pop_back();
    pol1.insert(pol1.end(), pol2.begin(), pol2.end());
    return {pol1};
}

static std::optional<Point> seg_intersection(Point p1, Point p2, Point p3, Point p4, double eps = 1e-9)
{
    Point ab = p2 - p1;
    Point cd = p4 - p3;
    const double denom = cross(ab, cd);
    if (std::abs(denom) > eps) {
        const double t = cross(p3 - p1, cd) / denom;
        const double u = cross(p3 - p1, ab) / denom;
        if (-eps <= t && t <= 1.0 + eps && -eps <= u && u <= 1.0 + eps)
            return p1 + ab * t;
        return std::nullopt;
    }
    if (std::abs(cross(p3 - p1, ab)) > eps)
        return std::nullopt;

    const int axis = std::abs(ab.x) >= std::abs(ab.y) ? 0 : 1;
    const double a0 = axis == 0 ? p1.x : p1.y;
    const double a1 = axis == 0 ? p2.x : p2.y;
    const double b0 = axis == 0 ? p3.x : p3.y;
    const double b1 = axis == 0 ? p4.x : p4.y;
    const double lo = std::max(std::min(a0, a1), std::min(b0, b1));
    const double hi = std::min(std::max(a0, a1), std::max(b0, b1));
    if (lo > hi + eps)
        return std::nullopt;
    return (p1 + p2 + p3 + p4) * 0.25;
}

std::vector<Point> PolylineIntersection(const Polyline& cur1, const Polyline& cur2)
{
    std::vector<Point> ans;
    for (int i = 0; i + 1 < static_cast<int>(cur1.points.size()); ++i) {
        for (int j = 0; j + 1 < static_cast<int>(cur2.points.size()); ++j) {
            auto inter = seg_intersection(cur1.points[i], cur1.points[i + 1], cur2.points[j], cur2.points[j + 1]);
            if (inter)
                ans.push_back(*inter);
        }
    }
    return ans;
}

static std::vector<int> range_(int c, int f)
{
    std::vector<int> a;
    for (int i = 0; i < c - 1; i += f)
        a.push_back(i);
    a.push_back(c - 1);
    return a;
}

std::pair<Points, Points> vectorfield_nodes(const cv::Mat& img,
                                            const Points& points,
                                            const std::vector<Points>& pattern,
                                            int QRsize,
                                            int,
                                            int fract,
                                            int lines_fract,
                                            const TPS* tps)
{
    const auto rg = range_(QRsize + 1, fract);
    const double characteristic_size = std::max(norm(points[6] - points[3]), norm(points[14] - points[1]));
    int kernel_size = std::max(static_cast<int>(characteristic_size * 30.0 / 300.0), 35);
    if (kernel_size % 2 == 0)
        ++kernel_size;

    DirectionField df(img);
    std::array<std::vector<Polyline>, 2> grade;

    for (int i : rg) {
        Point dist1 = norm(pattern[1][i] - points[2]) < norm(pattern[1][i] - points[13]) ? points[2] - points[3] : points[13] - points[12];
        grade[1].push_back(constructPolyline(df, pattern[1][i], dist1, QRsize, norm(points[6] - points[3]), lines_fract, kernel_size));

        Point dist2 = norm(pattern[0][i] - points[2]) < norm(pattern[0][i] - points[7]) ? points[2] - points[1] : points[7] - points[4];
        grade[0].push_back(constructPolyline(df, pattern[0][i], dist2, QRsize, norm(points[14] - points[1]), lines_fract, kernel_size));
    }

    std::vector<std::vector<Point>> nodes(rg.size(), std::vector<Point>(rg.size()));
    std::vector<std::vector<std::optional<Point>>> predicted_last(rg.size(), std::vector<std::optional<Point>>(rg.size()));

    if (tps) {
        Points pts;
        for (int x : rg)
            for (int y : rg)
                pts.emplace_back(x, y);
        pts = tps->tps_transform(pts);
        for (int x = 0; x < static_cast<int>(rg.size()); ++x)
            for (int y = 0; y < static_cast<int>(rg.size()); ++y)
                predicted_last[x][y] = pts[x * rg.size() + y];
    }

    for (int x = 0; x < static_cast<int>(rg.size()); ++x) {
        for (int y = 0; y < static_cast<int>(rg.size()); ++y) {
            Point last;
            if (predicted_last[x][y])
                last = *predicted_last[x][y];
            else if (x == 0 && y == 0)
                last = points[0];
            else if (x == 0)
                last = nodes[x][y - 1];
            else
                last = nodes[x - 1][y];

            auto p = PolylineIntersection(grade[0][x], grade[1][y]);
            if (p.empty()) {
                nodes[x][y] = last;
                continue;
            }
            nodes[x][y] = p[0];
            for (const auto& i : p) {
                if (norm(last - i) < norm(last - nodes[x][y]))
                    nodes[x][y] = i;
            }
        }
    }

    Points target;
    for (int x : rg)
        for (int y : rg)
            target.emplace_back(x, y);

    Points flat_nodes;
    for (const auto& row : nodes)
        flat_nodes.insert(flat_nodes.end(), row.begin(), row.end());

    return {target, flat_nodes};
}

} // namespace qrscanner
