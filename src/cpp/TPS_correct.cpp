#include "TPS_correct.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
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
        mapped.emplace_back(x * scale_ + shift_, y * scale_ + shift_);
    }
    return mapped;
}

void TPS::affine(double scale, double shift)
{
    scale_ *= scale;
    shift_ += shift;
}

cv::Mat TPS::warp_image_tps(const cv::Mat& src_img, cv::Size out_shape, double scale, double margins, int order, double cval) const
{
    cv::Mat map_x(out_shape, CV_32F);
    cv::Mat map_y(out_shape, CV_32F);

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
