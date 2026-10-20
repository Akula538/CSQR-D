#include "TPS_correct.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace qrscanner {

double TPS::tps_kernel(double r)
{
    if (r == 0.0)
        return 0.0;
    return r * r * std::log(r);
}

void TPS::fit(const Points& from_pts, const Points& to_pts, double reg)
{
    if (from_pts.size() != to_pts.size() || from_pts.empty())
        throw std::runtime_error("TPS fit expects matching non-empty point arrays");

    ctrl_pts_ = from_pts;
    scale_ = 1.0;
    shift_ = {0.0, 0.0};
    const int N = static_cast<int>(from_pts.size());
    cv::Mat K = cv::Mat::zeros(N, N, CV_64F);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            K.at<double>(i, j) = tps_kernel(norm(from_pts[i] - from_pts[j]));
    }

    cv::Mat P = cv::Mat::ones(N, 3, CV_64F);
    for (int i = 0; i < N; ++i) {
        P.at<double>(i, 1) = from_pts[i].x;
        P.at<double>(i, 2) = from_pts[i].y;
    }

    cv::Mat L = cv::Mat::zeros(N + 3, N + 3, CV_64F);
    cv::Mat Kreg = K + reg * cv::Mat::eye(N, N, CV_64F);
    Kreg.copyTo(L(cv::Rect(0, 0, N, N)));
    P.copyTo(L(cv::Rect(N, 0, 3, N)));
    cv::Mat Pt;
    cv::transpose(P, Pt);
    Pt.copyTo(L(cv::Rect(0, N, N, 3)));

    cv::Mat V = cv::Mat::zeros(N + 3, 2, CV_64F);
    for (int i = 0; i < N; ++i) {
        V.at<double>(i, 0) = to_pts[i].x;
        V.at<double>(i, 1) = to_pts[i].y;
    }

    if (!cv::solve(L, V, params_, cv::DECOMP_LU))
        cv::solve(L, V, params_, cv::DECOMP_SVD);
}

Points TPS::tps_transform(const Points& points) const
{
    const int N = static_cast<int>(ctrl_pts_.size());
    Points mapped;
    mapped.reserve(points.size());
    for (const auto& p : points) {
        double x = params_.at<double>(N, 0) + params_.at<double>(N + 1, 0) * p.x + params_.at<double>(N + 2, 0) * p.y;
        double y = params_.at<double>(N, 1) + params_.at<double>(N + 1, 1) * p.x + params_.at<double>(N + 2, 1) * p.y;
        for (int i = 0; i < N; ++i) {
            const double U = tps_kernel(norm(p - ctrl_pts_[i]));
            x += U * params_.at<double>(i, 0);
            y += U * params_.at<double>(i, 1);
        }
        mapped.emplace_back(x * scale_ + shift_.x, y * scale_ + shift_.y);
    }
    return mapped;
}

void TPS::affine(double scale, Point shift)
{
    scale_ *= scale;
    shift_ = shift_ * scale + shift;
}

cv::Mat TPS::warp_image_tps(const cv::Mat& src_img, cv::Size out_shape, double scale, double margins, int order, double cval) const
{
    cv::Mat map_x(out_shape, CV_32F);
    cv::Mat map_y(out_shape, CV_32F);

    // QR landmarks lie on an integer/half-integer module lattice. On the
    // output pixel lattice many control-point distances are identical.
    // Evaluate the kernel once per distance, rather than once per landmark
    // and output pixel. Keep the general path for arbitrary TPS landmarks.
    std::vector<cv::Point> centers;
    int max_dx = 0, max_dy = 0;
    bool lattice = std::isfinite(scale) && scale > 0 && std::isfinite(margins);
    for (const auto& p : ctrl_pts_) {
        const double x = p.x * scale + margins;
        const double y = p.y * scale + margins;
        if (!lattice || !std::isfinite(x) || !std::isfinite(y) ||
            std::abs(x) > 8192 || std::abs(y) > 8192 ||
            std::abs(x - std::round(x)) > 1e-9 || std::abs(y - std::round(y)) > 1e-9) {
            lattice = false;
            break;
        }
        centers.emplace_back(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
        max_dx = std::max({max_dx, std::abs(centers.back().x), std::abs(out_shape.width - 1 - centers.back().x)});
        max_dy = std::max({max_dy, std::abs(centers.back().y), std::abs(out_shape.height - 1 - centers.back().y)});
    }
    if (lattice && !centers.empty() &&
        static_cast<int64_t>(max_dx + 1) * (max_dy + 1) <= 4 * 1024 * 1024) {
        cv::Mat kernels(max_dy + 1, max_dx + 1, CV_64F);
        for (int dy = 0; dy <= max_dy; ++dy) {
            double* row = kernels.ptr<double>(dy);
            for (int dx = 0; dx <= max_dx; ++dx) {
                const double x = dx / scale, y = dy / scale;
                row[dx] = tps_kernel(std::sqrt(x * x + y * y));
            }
        }
        const int n = static_cast<int>(centers.size());
        const auto* weights = params_.ptr<cv::Vec2d>();
        std::vector<const double*> kernel_rows(n);
        for (int y = 0; y < out_shape.height; ++y) {
            for (int i = 0; i < n; ++i)
                kernel_rows[i] = kernels.ptr<double>(std::abs(y - centers[i].y));
            float* mx = map_x.ptr<float>(y);
            float* my = map_y.ptr<float>(y);
            const double py = (y - margins) / scale;
            for (int x = 0; x < out_shape.width; ++x) {
                const double px = (x - margins) / scale;
                double sx = weights[n][0] + weights[n + 1][0] * px + weights[n + 2][0] * py;
                double sy = weights[n][1] + weights[n + 1][1] * px + weights[n + 2][1] * py;
                for (int i = 0; i < n; ++i) {
                    const double u = kernel_rows[i][std::abs(x - centers[i].x)];
                    sx += u * weights[i][0];
                    sy += u * weights[i][1];
                }
                mx[x] = static_cast<float>(sx * scale_ + shift_.x);
                my[x] = static_cast<float>(sy * scale_ + shift_.y);
            }
        }
    } else {
        Points grid;
        grid.reserve(static_cast<size_t>(out_shape.area()));
        for (int y = 0; y < out_shape.height; ++y) {
            for (int x = 0; x < out_shape.width; ++x)
                grid.emplace_back((x - margins) / scale, (y - margins) / scale);
        }

        Points src = tps_transform(grid);
        size_t k = 0;
        for (int y = 0; y < out_shape.height; ++y) {
            float* mx = map_x.ptr<float>(y);
            float* my = map_y.ptr<float>(y);
            for (int x = 0; x < out_shape.width; ++x, ++k) {
                mx[x] = static_cast<float>(src[k].x);
                my[x] = static_cast<float>(src[k].y);
            }
        }
    }

    cv::Mat warped;
    cv::remap(src_img, warped, map_x, map_y, order == 0 ? cv::INTER_NEAREST : cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(cval));
    return warped;
}

cv::Mat tps_correct(const cv::Mat& img, const TPS& tps, int QRsize, int square_size)
{
    const int margins = square_size * 2;
    const int img_size = QRsize * square_size + margins * 2;
    return tps.warp_image_tps(img, cv::Size(img_size, img_size), square_size, margins);
}

TPS fit_tps_full_qr(const Points& qr, int QRsize)
{
    Points search = {{0, 0}, {7, 0}, {7, 7}, {0, 7}};
    Points alligment = {{0, 0}, {3, 0}, {3, 3}, {0, 3}};

    Points to;
    to.insert(to.end(), search.begin(), search.end());
    for (auto p : search) to.push_back(p + Point(QRsize - 7, 0));
    for (auto p : alligment) to.push_back(p + Point(QRsize - 8, QRsize - 8));
    for (auto p : search) to.push_back(p + Point(0, QRsize - 7));

    TPS tps;
    tps.fit(to, qr, 1e-3);
    return tps;
}

TPS fit_tps_alligment_center(const Points& qr, int QRsize)
{
    Points search = {{0, 0}, {7, 0}, {7, 7}, {0, 7}};
    Points alligment = {{1.5, 1.5}};

    Points to;
    to.insert(to.end(), search.begin(), search.end());
    for (auto p : search) to.push_back(p + Point(QRsize - 7, 0));
    for (auto p : alligment) to.push_back(p + Point(QRsize - 8, QRsize - 8));
    for (auto p : search) to.push_back(p + Point(0, QRsize - 7));

    Points qr2;
    qr2.insert(qr2.end(), qr.begin(), qr.begin() + 8);
    qr2.push_back(center(Points(qr.begin() + 8, qr.begin() + 12)));
    qr2.insert(qr2.end(), qr.end() - 4, qr.end());

    TPS tps;
    tps.fit(to, qr2, 1e-3);
    return tps;
}

TPS fit_tps_no_alligment(const Points& qr, int QRsize)
{
    Points search = {{0, 0}, {7, 0}, {7, 7}, {0, 7}};

    Points to;
    to.insert(to.end(), search.begin(), search.end());
    for (auto p : search) to.push_back(p + Point(QRsize - 7, 0));
    for (auto p : search) to.push_back(p + Point(0, QRsize - 7));

    Points qr2;
    qr2.insert(qr2.end(), qr.begin(), qr.begin() + 8);
    qr2.insert(qr2.end(), qr.end() - 4, qr.end());

    TPS tps;
    tps.fit(to, qr2, 1e-3);
    return tps;
}

} // namespace qrscanner
