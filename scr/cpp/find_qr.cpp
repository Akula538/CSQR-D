#include "find_qr.hpp"

#include "BitMatrix.h"
#include "ConcentricFinder.h"
#include "HybridBinarizer.h"
#include "ImageView.h"
#include "binarizer.hpp"
#include "interpolation_correct.hpp"
#include "qrcode/QRDetector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

namespace qrscanner {

static double sign(double x)
{
    if (x > 0.0)
        return 1.0;
    if (x < 0.0)
        return -1.0;
    return 0.0;
}

static cv::Mat to_gray_continuous(const cv::Mat& img)
{
    cv::Mat gray;
    if (img.channels() == 1)
        gray = img.clone();
    else if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4)
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    else
        gray = img.clone();
    if (!gray.isContinuous())
        gray = gray.clone();
    return gray;
}

static cv::Mat bitmatrix_to_bgr(const ZXing::BitMatrix& matrix)
{
    cv::Mat binary(matrix.height(), matrix.width(), CV_8UC1);
    for (int y = 0; y < matrix.height(); ++y) {
        uchar* row = binary.ptr<uchar>(y);
        for (int x = 0; x < matrix.width(); ++x)
            row[x] = matrix.get(x, y) ? 0 : 255;
    }
    cv::Mat bgr;
    cv::cvtColor(binary, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

std::vector<cv::Point> approxQuadr(const std::vector<cv::Point>& contour)
{
    double l = 0.1;
    double r = cv::arcLength(contour, true);
    while (r - l > 0.1) {
        const double m = (l + r) / 2.0;
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, m, true);
        if (approx.size() <= 4)
            r = m;
        else
            l = m;
    }

    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, r, true);
    return approx;
}

std::pair<std::vector<cv::Point>, double> approxedQuadrR(const std::vector<cv::Point>& contour)
{
    double l = 0.1;
    double r = cv::arcLength(contour, true) * 0.1;
    bool flag = false;
    while (r - l > 0.1) {
        const double m = (l + r) / 2.0;
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, m, true);
        if (approx.size() <= 4) {
            r = m;
            flag = true;
        } else {
            l = m;
        }
    }
    if (!flag)
        r = 1000000000.0;

    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, r, true);
    return {approx, r};
}

static bool isQuadr(const std::vector<cv::Point>& contour)
{
    if (cv::contourArea(contour) == 0.0)
        return false;
    auto [app, d] = approxedQuadrR(contour);
    return d <= 0.1 * cv::arcLength(app, true);
}

std::vector<int> get_colors(const std::vector<std::vector<cv::Point>>&, const std::vector<cv::Vec4i>& hierarchy)
{
    if (hierarchy.empty())
        return {};

    const int n = static_cast<int>(hierarchy.size());
    std::vector<int> parents(n);
    std::vector<int> depths(n, -1);
    for (int i = 0; i < n; ++i) {
        parents[i] = hierarchy[i][3];
        if (parents[i] == -1)
            depths[i] = 0;
    }

    bool unknown = true;
    while (unknown) {
        unknown = false;
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            if (depths[i] != -1)
                continue;
            unknown = true;
            const int parent = parents[i];
            if (parent != -1 && depths[parent] != -1) {
                depths[i] = depths[parent] + 1;
                changed = true;
            }
        }
        if (unknown && !changed) {
            for (int& d : depths)
                if (d == -1)
                    d = 0;
            break;
        }
    }

    std::vector<int> colors(n);
    for (int i = 0; i < n; ++i)
        colors[i] = depths[i] % 2;
    return colors;
}

std::vector<int> find_search(const std::vector<std::vector<cv::Point>>& contours,
                             const std::vector<cv::Vec4i>& hierarchy,
                             const std::vector<int>& colors,
                             int lvl)
{
    std::vector<int> ans;
    if (hierarchy.empty())
        return ans;

    if (lvl == 0) {
        for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
            if (colors[i] != 1)
                continue;
            int j = hierarchy[i][2];
            while (j != -1) {
                int k = hierarchy[j][2];
                while (k != -1) {
                    if (isQuadr(contours[k]) && isQuadr(contours[j]) && isQuadr(contours[i])) {
                        ans.push_back(i);
                        break;
                    }
                    k = hierarchy[k][0];
                }
                if (k != -1)
                    break;
                j = hierarchy[j][0];
            }
        }
        return ans;
    }

    if (lvl == 1) {
        for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
            if (colors[i] != 0)
                continue;
            int j = hierarchy[i][2];
            while (j != -1) {
                if (isQuadr(contours[j]) && isQuadr(contours[i])) {
                    ans.push_back(i);
                    break;
                }
                j = hierarchy[j][0];
            }
        }
        return ans;
    }

    if (lvl == 2) {
        for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
            if (colors[i] == 1 && isQuadr(contours[i]))
                ans.push_back(i);
        }
    }
    return ans;
}

std::vector<int> find_corrective(const std::vector<std::vector<cv::Point>>& contours,
                                 const std::vector<cv::Vec4i>& hierarchy,
                                 const std::vector<int>& colors,
                                 double search_area)
{
    std::vector<int> ans;
    if (hierarchy.empty())
        return ans;

    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        if (colors[i] != 0 || !isQuadr(contours[i]))
            continue;
        if (hierarchy[i][2] != -1 && cv::contourArea(contours[i]) < search_area)
            ans.push_back(i);
    }
    return ans;
}

Points clockwise(const Points& contour)
{
    auto cvp = to_cv_points_i(contour);
    if (cvp.size() < 3)
        return contour;
    Points out = contour;
    if (cv::contourArea(cvp, true) < 0.0)
        std::reverse(out.begin(), out.end());
    return out;
}

static double dist_center(const Points& a, const Points& b)
{
    return norm(center(a) - center(b));
}

static bool same_int_point(const Point& p, const Point& q)
{
    return static_cast<int>(p.x) == static_cast<int>(q.x) && static_cast<int>(p.y) == static_cast<int>(q.y);
}

static bool contains_point(const Points& points, const Point& p, int* idx = nullptr)
{
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        if (same_int_point(points[i], p)) {
            if (idx)
                *idx = i;
            return true;
        }
    }
    return false;
}

std::vector<Points> sort_search(const std::vector<Points>& search_in)
{
    std::vector<Points> search = search_in;
    std::vector<cv::Point> all_vertices;
    for (const auto& s : search) {
        for (const auto& p : s)
            all_vertices.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
    }

    std::vector<cv::Point> hull_cv;
    cv::convexHull(all_vertices, hull_cv, false, true);
    Points hull_vertices = to_points_d(hull_cv);

    auto calculate_angle = [](const Point& p1, const Point& p2, const Point& p3) {
        Point v1 = p1 - p2;
        Point v2 = p3 - p2;
        const double mag1 = norm(v1);
        const double mag2 = norm(v2);
        if (mag1 == 0.0 || mag2 == 0.0)
            return 180.0;
        double c = v1.dot(v2) / (mag1 * mag2);
        c = std::max(-1.0, std::min(1.0, c));
        return std::acos(c) * 180.0 / CV_PI;
    };

    std::vector<std::pair<double, int>> angle_indices;
    for (int i = 0; i < static_cast<int>(hull_vertices.size()); ++i) {
        angle_indices.push_back({calculate_angle(hull_vertices[(i - 1 + hull_vertices.size()) % hull_vertices.size()],
                                                 hull_vertices[i],
                                                 hull_vertices[(i + 1) % hull_vertices.size()]),
                                 i});
    }
    std::sort(angle_indices.begin(), angle_indices.end());

    std::vector<int> selected;
    for (int i = 0; i < std::min(5, static_cast<int>(angle_indices.size())); ++i)
        selected.push_back(angle_indices[i].second);
    std::sort(selected.begin(), selected.end());

    Points pentagon;
    for (int idx : selected)
        pentagon.push_back(hull_vertices[idx]);
    std::reverse(pentagon.begin(), pentagon.end());

    std::vector<Points> new_search;
    int zero_point = -1;
    for (int i = 0; i < 3; ++i) {
        std::vector<Point> points_in;
        for (const auto& p : search[i]) {
            for (const auto& hp : pentagon) {
                if (same_int_point(p, hp))
                    points_in.push_back(hp);
            }
        }
        if (points_in.size() == 1) {
            for (int k = 0; k < static_cast<int>(pentagon.size()); ++k) {
                if (same_int_point(pentagon[k], points_in[0])) {
                    zero_point = k;
                    break;
                }
            }
            int shift = 0;
            contains_point(search[i], points_in[0], &shift);
            search[i] = cyclic_shift(search[i], shift);
            new_search.push_back(search[i]);
            break;
        }
    }
    if (zero_point == -1)
        throw std::runtime_error("Failed to sort finder patterns by convex hull");

    Point point = pentagon[(zero_point + 1) % 5];
    for (int i = 0; i < 3; ++i) {
        int idx = 0;
        if (contains_point(search[i], point, &idx)) {
            search[i] = cyclic_shift(search[i], (idx - 1) % 4);
            new_search.push_back(search[i]);
            break;
        }
    }

    point = pentagon[(zero_point - 1 + 5) % 5];
    for (int i = 0; i < 3; ++i) {
        int idx = 0;
        if (contains_point(search[i], point, &idx)) {
            search[i] = cyclic_shift(search[i], (idx + 1) % 4);
            new_search.push_back(search[i]);
            break;
        }
    }
    return new_search;
}

std::vector<Points> sort_search_dist(const std::vector<Points>& search)
{
    std::vector<Points> new_search;
    double mx = 0.0;
    int best = 0;
    for (int i = 0; i < 3; ++i) {
        const double d = dist_center(search[(i + 1) % 3], search[(i + 2) % 3]);
        if (d > mx) {
            mx = d;
            best = i;
        }
    }
    new_search = cyclic_shift(search, best);

    if (cross(center(new_search[2]) - center(new_search[0]), center(new_search[1]) - center(new_search[0])) > 0.0)
        std::swap(new_search[1], new_search[2]);

    Point c = (center(new_search[1]) + center(new_search[2])) * 0.5;

    auto shift_search = [&](int ind, int target) {
        int best_idx = 0;
        for (int i = 0; i < 4; ++i) {
            if (norm(new_search[ind][best_idx] - c) > norm(new_search[ind][i] - c))
                best_idx = i;
        }
        new_search[ind] = cyclic_shift(new_search[ind], (best_idx - target) % 4);
    };

    shift_search(0, 2);
    shift_search(1, 3);
    shift_search(2, 1);
    return new_search;
}

std::vector<Points> sort_search_dir(const std::vector<Points>& search)
{
    auto abs_mod = [](double x) {
        x = std::fmod(x, CV_PI);
        if (x < 0)
            x += CV_PI;
        return std::min(x, CV_PI - x);
    };

    auto mean_mod = [&](double a, double b) {
        double x = (a + b) / 2.0;
        double y = std::fmod(x + CV_PI / 2.0, CV_PI);
        if (abs_mod(x - a) < abs_mod(y - a))
            return x;
        return y;
    };

    std::vector<std::array<double, 2>> dirs(3);
    for (int s = 0; s < 3; ++s) {
        double d[4];
        for (int i = 0; i < 4; ++i) {
            Point v = search[s][(i + 1) % 4] - search[s][i];
            double a = std::fmod(std::atan2(v.y, v.x), CV_PI);
            if (a < 0)
                a += CV_PI;
            d[i] = a;
        }
        dirs[s] = {mean_mod(d[0], d[2]), mean_mod(d[1], d[3])};
    }

    double mx = 0.0;
    int best = 0;
    for (int i = 0; i < 3; ++i) {
        const double angle = abs_mod(dirs[i][0] - dirs[i][1]);
        if (angle > mx) {
            mx = angle;
            best = i;
        }
    }

    double dir1 = dirs[best][0];
    double dir2 = dirs[best][1];
    for (int off : {1, 2}) {
        int idx = (best + off) % 3;
        if (abs_mod(dir1 - dirs[idx][0]) > abs_mod(dir1 - dirs[idx][1]))
            std::swap(dirs[idx][0], dirs[idx][1]);
    }

    dir1 = mean_mod(dir1, mean_mod(dirs[(best + 1) % 3][0], dirs[(best + 2) % 3][0]));
    dir2 = mean_mod(dir2, mean_mod(dirs[(best + 1) % 3][1], dirs[(best + 2) % 3][1]));

    Point v1(std::cos(dir1), std::sin(dir1));
    Point v2(std::cos(dir2), std::sin(dir2));
    cv::Matx22d M(v1.x, v2.x, v1.y, v2.y);
    cv::Matx22d Minv = M.inv();

    auto transform_center = [&](const Points& s) {
        cv::Vec2d r = Minv * cv::Vec2d(center(s).x, center(s).y);
        return r;
    };

    auto worst_by_dir = [&](int d) {
        double mx_local = 0.0;
        int idx = 0;
        for (int i = 0; i < 3; ++i) {
            const double xi = transform_center(search[i])[d];
            const double x1 = transform_center(search[(i + 1) % 3])[d];
            const double x2 = transform_center(search[(i + 2) % 3])[d];
            const double x = std::abs(xi - x1) + std::abs(xi - x2);
            if (x > mx_local) {
                mx_local = x;
                idx = i;
            }
        }
        return idx;
    };

    std::vector<int> left_top = {0, 1, 2};
    int worst0 = worst_by_dir(0);
    int worst1 = worst_by_dir(1);
    if (worst0 == worst1)
        return sort_search(search);
    left_top.erase(std::remove(left_top.begin(), left_top.end(), worst0), left_top.end());
    left_top.erase(std::remove(left_top.begin(), left_top.end(), worst1), left_top.end());

    std::vector<Points> new_search = cyclic_shift(search, left_top[0]);
    if (cross(center(new_search[2]) - center(new_search[0]), center(new_search[1]) - center(new_search[0])) > 0.0)
        std::swap(new_search[1], new_search[2]);

    Point c = (center(new_search[1]) + center(new_search[2])) * 0.5;
    auto shift_search = [&](int ind, int target) {
        int best_idx = 0;
        for (int i = 0; i < 4; ++i) {
            if (norm(new_search[ind][best_idx] - c) > norm(new_search[ind][i] - c))
                best_idx = i;
        }
        new_search[ind] = cyclic_shift(new_search[ind], (best_idx - target) % 4);
    };

    shift_search(0, 2);
    shift_search(1, 3);
    shift_search(2, 1);
    return new_search;
}

Points sort_corrective(const std::vector<Points>& correctives, const std::vector<Points>& search)
{
    if (correctives.empty())
        throw std::runtime_error("No corrective pattern candidates");

    Points corrective = correctives[0];
    Point predicted = search[1][3] + search[2][1] - search[0][2];
    for (const auto& c : correctives) {
        if (norm(predicted - center(c)) < norm(predicted - center(corrective)))
            corrective = c;
    }

    int zero = 0;
    for (int i = 0; i < 4; ++i) {
        if (norm(center(search[0]) - corrective[i]) < norm(center(search[0]) - corrective[zero]))
            zero = i;
    }
    return cyclic_shift(corrective, zero);
}

static std::vector<std::array<int, 3>> valid_combinations(const std::vector<Points>& search)
{
    std::vector<std::array<int, 3>> ans;
    double threshold = search.size() <= 3 ? 1.5 : 1.0;

    std::vector<double> mods;
    for (const auto& s : search) {
        double m = 0.0;
        for (int i = 0; i < static_cast<int>(s.size()); ++i)
            m = std::max(m, norm(s[i] - s[(i + 1) % s.size()]));
        mods.push_back(m);
    }

    for (int i = 0; i < static_cast<int>(search.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(search.size()); ++j) {
            for (int k = j + 1; k < static_cast<int>(search.size()); ++k) {
                std::array<double, 3> md = {mods[i], mods[j], mods[k]};
                const double mn = *std::min_element(md.begin(), md.end());
                const double mx = *std::max_element(md.begin(), md.end());
                if (mn > 0.0 && (mx - mn) / mn < threshold)
                    ans.push_back({i, j, k});
            }
        }
    }
    return ans;
}

static std::pair<std::vector<std::vector<Points>>, std::vector<std::array<int, 3>>> disp_evristic_sort(const std::vector<Points>& search,
                                                                                                      const std::vector<double>& search_areas)
{
    auto F_area = [](double S1, double S2, double S3) {
        const double mean_area = (S1 + S2 + S3) / 3.0;
        const double variance = ((S1 - mean_area) * (S1 - mean_area) +
                                 (S2 - mean_area) * (S2 - mean_area) +
                                 (S3 - mean_area) * (S3 - mean_area)) /
                                3.0;
        return 1.0 / (1.0 + variance / (mean_area + 1e-6));
    };

    auto F_shape = [&](const std::array<double, 3>& sides_in) {
        auto sides = sides_in;
        std::sort(sides.begin(), sides.end());
        const double leg1 = sides[0];
        const double leg2 = sides[1];
        const double hypo = sides[2];
        const double actual_leg_ratio = leg2 > 0.0 ? leg1 / leg2 : 0.0;
        const double actual_hypo_ratio = leg1 > 0.0 ? hypo / leg1 : 0.0;
        const double leg_similarity = 1.0 - std::min(1.0, std::abs(actual_leg_ratio - 1.0));
        const double hypo_similarity = 1.0 - std::min(1.0, std::abs(actual_hypo_ratio - std::sqrt(2.0)) / std::sqrt(2.0));
        double angle_score = 0.0;
        if (leg1 > 0.0 && leg2 > 0.0) {
            const double cos_angle = (leg1 * leg1 + leg2 * leg2 - hypo * hypo) / (2.0 * leg1 * leg2);
            angle_score = 1.0 - std::min(1.0, std::abs(cos_angle));
        }
        return (leg_similarity + hypo_similarity + angle_score) / 3.0;
    };

    std::vector<std::pair<double, std::array<int, 3>>> scored;
    for (const auto& qr : valid_combinations(search)) {
        const double evr = -(0.4 * F_area(search_areas[qr[0]], search_areas[qr[1]], search_areas[qr[2]]) +
                             0.6 * F_shape({dist_center(search[qr[0]], search[qr[1]]),
                                            dist_center(search[qr[1]], search[qr[2]]),
                                            dist_center(search[qr[2]], search[qr[0]])}));
        scored.push_back({evr, qr});
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::vector<Points>> all_search;
    std::vector<std::array<int, 3>> indexes;
    for (const auto& s : scored) {
        all_search.push_back({search[s.second[0]], search[s.second[1]], search[s.second[2]]});
        indexes.push_back(s.second);
    }
    return {all_search, indexes};
}

Points find(const cv::Mat& img)
{
    cv::Mat gray;
    if (img.channels() == 1)
        gray = img;
    else
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    auto colors = get_colors(contours, hierarchy);
    std::vector<Points> search_contours;
    for (int idx : find_search(contours, hierarchy, colors))
        search_contours.push_back(clockwise(to_points_d(approxQuadr(contours[idx]))));

    std::vector<double> search_areas;
    for (const auto& c : search_contours)
        search_areas.push_back(cv::contourArea(to_cv_points_i(c)));

    auto [all_search, all_indexes] = disp_evristic_sort(search_contours, search_areas);
    std::vector<Points> search = all_search.at(0);

    const double search_area = std::accumulate(search_areas.begin(), search_areas.end(), 0.0) / std::max<size_t>(search_contours.size(), 1);
    search = sort_search_dir(search);

    std::vector<Points> corrective;
    for (int idx : find_corrective(contours, hierarchy, colors, search_area))
        corrective.push_back(clockwise(to_points_d(approxQuadr(contours[idx]))));

    Points corr = sort_corrective(corrective, search);
    search.insert(search.begin() + 2, corr);

    Points flat;
    for (const auto& s : search)
        flat.insert(flat.end(), s.begin(), s.end());
    return flat;
}

static int get_pixel_binary(const cv::Mat& img, Point p, const HomographyMatrix* H, bool tolerate_outside)
{
    if (H)
        p = H->predict(p);
    const int x = static_cast<int>(p.x);
    const int y = static_cast<int>(p.y);
    if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) {
        if (tolerate_outside)
            return 0;
        throw std::out_of_range("pixel outside image");
    }
    if (img.channels() == 1)
        return img.at<uchar>(y, x) > 127 ? 1 : 0;
    const cv::Vec3b pix = img.at<cv::Vec3b>(y, x);
    return (static_cast<int>(pix[0]) + pix[1] + pix[2]) / 3 > 127 ? 1 : 0;
}

static Points recognize_distribution_(const cv::Mat& img, Point p1, Point p2, Point p3, Point p4, const HomographyMatrix* H)
{
    Curve cur(p1, p2, p3, p4);
    const double dx = sign(cur.dist(p3, p2));
    const int count = static_cast<int>(std::abs(cur.dist(p2, p3)));
    Points points;
    for (int i = 0; i < count; ++i) {
        Point p = cur.move(p2, dx * i);
        points.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
    }
    if (points.empty())
        throw std::runtime_error("No distribution scan points");

    while (!points.empty() && !get_pixel_binary(img, points.back(), H, false))
        points.pop_back();
    while (!points.empty() && !get_pixel_binary(img, points.front(), H, false))
        points.erase(points.begin());
    if (points.size() < 2)
        throw std::runtime_error("Distribution scan is empty");

    Points bord;
    for (int i = 0; i + 1 < static_cast<int>(points.size()); ++i) {
        if (get_pixel_binary(img, points[i], H, false) != get_pixel_binary(img, points[i + 1], H, false))
            bord.push_back(points[i]);
    }
    if (bord.size() < 2)
        throw std::runtime_error("Distribution has too few borders");

    double max_step = 0.0;
    for (int i = 0; i + 1 < static_cast<int>(bord.size()); ++i)
        max_step = std::max(max_step, std::abs(cur.dist(bord[i + 1], bord[i])));
    const double trashold = max_step * 0.3;

    int i = 0;
    while (i + 1 < static_cast<int>(bord.size())) {
        if (std::abs(cur.dist(bord[i + 1], bord[i])) < trashold) {
            bord.erase(bord.begin() + i);
            bord.erase(bord.begin() + i);
        } else {
            ++i;
        }
    }
    return bord;
}

static Points recognize_search_distribution_(const cv::Mat& img, Point p1, Point p2, const HomographyMatrix* H)
{
    Curve cur(p1, (p1 + p2) * 0.5, p2);
    const double dx = 0.5 * sign(cur.dist(p2, p1));
    const int all_dist = 2 * static_cast<int>(std::abs(cur.dist(p1, p2)));

    Points points;
    const int start = -((all_dist + 6) / 7);
    const int end = (all_dist * 8) / 7;
    for (int i = start; i < end; ++i) {
        Point p = cur.move(p1, dx * i);
        points.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
    }
    if (points.empty())
        throw std::runtime_error("No search distribution scan points");

    while (!points.empty() && !get_pixel_binary(img, points.back(), H, true))
        points.pop_back();
    while (!points.empty() && !get_pixel_binary(img, points.front(), H, true))
        points.erase(points.begin());
    if (points.size() < 2)
        throw std::runtime_error("Search distribution scan is empty");

    Points bord;
    for (int i = 0; i + 1 < static_cast<int>(points.size()); ++i) {
        if (get_pixel_binary(img, points[i], H, true) != get_pixel_binary(img, points[i + 1], H, true))
            bord.push_back(points[i]);
    }
    if (bord.size() < 2)
        throw std::runtime_error("Search distribution has too few borders");

    double max_step = 0.0;
    for (int i = 0; i + 1 < static_cast<int>(bord.size()); ++i)
        max_step = std::max(max_step, std::abs(cur.dist(bord[i + 1], bord[i])));
    const double trashold = max_step * 0.1;

    int i = 0;
    while (i + 1 < static_cast<int>(bord.size())) {
        if (std::abs(cur.dist(bord[i + 1], bord[i])) < trashold) {
            bord.erase(bord.begin() + i);
            bord.erase(bord.begin() + i);
        } else {
            ++i;
        }
    }
    return bord;
}

DistributionResult recognize_distribution(const cv::Mat& img, const Points& search, const HomographyMatrix* H)
{
    auto pattern1 = recognize_distribution_(img, (search[0] + search[3] * 13.0) * (1.0 / 14.0),
                                            (search[1] + search[2] * 13.0) * (1.0 / 14.0),
                                            (search[4] + search[7] * 13.0) * (1.0 / 14.0),
                                            (search[5] + search[6] * 13.0) * (1.0 / 14.0), H);

    auto pattern2 = recognize_distribution_(img, (search[0] + search[1] * 13.0) * (1.0 / 14.0),
                                            (search[3] + search[2] * 13.0) * (1.0 / 14.0),
                                            (search[12] + search[13] * 13.0) * (1.0 / 14.0),
                                            (search[15] + search[14] * 13.0) * (1.0 / 14.0), H);

    int sz1 = (static_cast<int>(pattern1.size()) / 4) * 4 + 17;
    int sz2 = (static_cast<int>(pattern2.size()) / 4) * 4 + 17;
    int size = std::max(sz1, sz2);

    std::optional<Points> p1 = pattern1;
    std::optional<Points> p2 = pattern2;
    if (pattern1.size() % 4 != 2)
        p1.reset();
    if (pattern2.size() % 4 != 2)
        p2.reset();
    if (!p1 && p2)
        size = sz2;
    if (p1 && !p2)
        size = sz1;
    if (size != sz1)
        p1.reset();
    if (size != sz2)
        p2.reset();

    return {size, {p1, p2}};
}

static std::optional<Points> comp_pattern(const Points& pattern, Point p1, Point p2)
{
    if (pattern.size() != 6)
        return std::nullopt;
    Curve cur(p1, (p1 + p2) * 0.5, p2);
    std::vector<double> x = {0, 1, 2, 5, 6, 7};
    std::vector<double> y;
    for (const auto& p : pattern)
        y.push_back(cur.dist(p, p1));
    Quintic qu(x, y);

    Points out;
    out.insert(out.end(), pattern.begin(), pattern.begin() + 3);
    out.push_back(cur.move(p1, qu.predict(3)));
    out.push_back(cur.move(p1, qu.predict(4)));
    out.insert(out.end(), pattern.begin() + 3, pattern.end());
    return out;
}

SearchDistributionResult recognize_search_distribution(const cv::Mat& img, const Points& search, const HomographyMatrix* H)
{
    Points pattern1 = recognize_search_distribution_(img, (search[0] + search[3]) * 0.5, (search[1] + search[2]) * 0.5, H);
    Points pattern2 = recognize_search_distribution_(img, (search[4] + search[7]) * 0.5, (search[5] + search[6]) * 0.5, H);
    Points pattern3 = recognize_search_distribution_(img, (search[0] + search[1]) * 0.5, (search[3] + search[2]) * 0.5, H);
    Points pattern4 = recognize_search_distribution_(img, (search[12] + search[13]) * 0.5, (search[15] + search[14]) * 0.5, H);

    return {{
        comp_pattern(pattern1, (search[0] + search[3]) * 0.5, (search[1] + search[2]) * 0.5),
        comp_pattern(pattern2, (search[4] + search[7]) * 0.5, (search[5] + search[6]) * 0.5),
        comp_pattern(pattern3, (search[0] + search[1]) * 0.5, (search[3] + search[2]) * 0.5),
        comp_pattern(pattern4, (search[12] + search[13]) * 0.5, (search[15] + search[14]) * 0.5),
    }};
}

ZxingFindResult zxing_find_with_binary(const cv::Mat& img)
{
    cv::Mat gray = to_gray_continuous(img);
    ZXing::ImageView view(gray.data,
                          static_cast<int>(gray.total()),
                          gray.cols,
                          gray.rows,
                          ZXing::ImageFormat::Lum,
                          gray.cols,
                          1);
    ZXing::HybridBinarizer bin(view);
    auto matrixPtr = bin.getBlackMatrix();
    const ZXing::BitMatrix& matrix = *matrixPtr;

    ZxingFindResult result;
    result.binary = bitmatrix_to_bgr(matrix);

    auto patterns = ZXing::QRCode::FindFinderPatterns(matrix, true);
    for (const auto& cp : patterns) {
        ZxingPattern pattern;
        pattern.center = {cp.x, cp.y};
        pattern.size = cp.size;
        auto quad = ZXing::FindConcentricPatternCorners(matrix, cp, cp.size, 2);
        if (quad) {
            for (const auto& p : *quad)
                pattern.corners.push_back({p.x, p.y});
        }
        result.patterns.push_back(pattern);
    }

    int i = 0;
    while (i < static_cast<int>(result.patterns.size()) - 1) {
        if (result.patterns[i].center.x == result.patterns[i + 1].center.x &&
            result.patterns[i].center.y == result.patterns[i + 1].center.y) {
            result.patterns.erase(result.patterns.begin() + i + 1);
        } else {
            ++i;
        }
    }

    return result;
}

std::vector<ZxingPattern> zxing_find(const cv::Mat& img)
{
    return zxing_find_with_binary(img).patterns;
}

std::optional<std::string> zxing_check(const cv::Mat& img)
{
    auto res = zxing_find(img);
    if (res.size() < 3)
        return std::string("CTFall");
    int cnt = 0;
    for (const auto& i : res)
        cnt += static_cast<int>(i.corners.size() == 4);
    if (cnt < 3)
        return std::string("CTF");
    return std::nullopt;
}

bool zxing_detect(const cv::Mat& img)
{
    return zxing_find(img).size() >= 3;
}

struct ContoursWindowResult {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
};

static ContoursWindowResult findContoursWindow(const cv::Mat& binary, Point c_, int d_)
{
    auto relux = [&](int x) {
        if (x < 0)
            return 0;
        if (x >= binary.cols)
            return binary.cols - 1;
        return x;
    };
    auto reluy = [&](int y) {
        if (y < 0)
            return 0;
        if (y >= binary.rows)
            return binary.rows - 1;
        return y;
    };

    const int d = d_ + 1;
    const int cx = static_cast<int>(c_.x);
    const int cy = static_cast<int>(c_.y);
    const int x1 = relux(cx - d);
    const int x2 = relux(cx + d);
    const int y1 = reluy(cy - d);
    const int y2 = reluy(cy + d);

    cv::Mat roi = binary(cv::Range(y1, std::max(y1 + 1, y2)), cv::Range(x1, std::max(x1 + 1, x2))).clone();
    ContoursWindowResult result;
    cv::findContours(roi, result.contours, result.hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    for (auto& cnt : result.contours) {
        bool touches = false;
        for (const auto& p : cnt) {
            if (p.x <= 1 || p.x >= x2 - x1 - 1 || p.y <= 1 || p.y >= y2 - y1 - 1) {
                touches = true;
                break;
            }
        }
        if (touches)
            cnt = {cv::Point(0, 0)};
    }

    for (auto& cnt : result.contours) {
        for (auto& p : cnt)
            p += cv::Point(x1, y1);
    }
    return result;
}

std::pair<Points, cv::Mat> find_qr_zxing(const cv::Mat& orig, const ZxingFindResult* zxing_qr)
{
    ZxingFindResult local;
    if (!zxing_qr) {
        local = zxing_find_with_binary(orig);
        zxing_qr = &local;
    }

    const auto& prets = zxing_qr->patterns;
    cv::Mat img = zxing_qr->binary.clone();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);

    std::vector<Points> search_contours;
    for (const auto& pret : prets) {
        Point c(static_cast<int>(pret.center.x), static_cast<int>(pret.center.y));
        if (!pret.corners.empty()) {
            Points cont;
            for (const auto& p : pret.corners)
                cont.push_back((p - c) * (7.0 / 6.0) + c);
            search_contours.push_back(cont);
        } else {
            auto window = findContoursWindow(binary, c, pret.size);
            auto colors = get_colors(window.contours, window.hierarchy);
            for (int lvl = 0; lvl < 3; ++lvl) {
                auto search_prets = find_search(window.contours, window.hierarchy, colors, lvl);
                if (!search_prets.empty()) {
                    auto best = approxQuadr(window.contours[search_prets[0]]);
                    for (int contour : search_prets) {
                        auto cont = approxQuadr(window.contours[contour]);
                        if (norm(center(to_points_d(best)) - c) > norm(center(to_points_d(cont)) - c))
                            best = cont;
                    }
                    Points cont;
                    for (const auto& p : to_points_d(best))
                        cont.push_back((p - c) * (7.0 / (7.0 - lvl * 2.0)) + c);
                    search_contours.push_back(cont);
                    break;
                }
            }
        }
    }

    for (auto& s : search_contours) {
        for (auto& p : s)
            p = Point(static_cast<int>(p.x), static_cast<int>(p.y));
        s = clockwise(s);
    }

    std::vector<double> search_areas;
    for (const auto& s : search_contours)
        search_areas.push_back(cv::contourArea(to_cv_points_i(s)));

    std::vector<double> pret_sizes;
    for (const auto& p : prets)
        pret_sizes.push_back(static_cast<double>(p.size));

    auto [all_search, all_indexes] = disp_evristic_sort(search_contours, pret_sizes);
    auto search = all_search.at(0);
    auto indexes = all_indexes.at(0);

    const double search_area = (search_areas[indexes[0]] + search_areas[indexes[1]] + search_areas[indexes[2]]) / 3.0;
    int search_module = std::max({prets[indexes[0]].size, prets[indexes[1]].size, prets[indexes[2]].size});

    search = sort_search_dir(search);

    Point c = search[1][3] + search[2][1] - search[0][2];
    auto window = findContoursWindow(binary, c, 2 * search_module);
    auto colors = get_colors(window.contours, window.hierarchy);

    std::vector<Points> corrective;
    for (int idx : find_corrective(window.contours, window.hierarchy, colors, search_area))
        corrective.push_back(clockwise(to_points_d(approxQuadr(window.contours[idx]))));

    Points corr = sort_corrective(corrective, search);
    search.insert(search.begin() + 2, corr);

    Points flat;
    for (const auto& s : search)
        flat.insert(flat.end(), s.begin(), s.end());
    return {flat, img};
}

} // namespace qrscanner
