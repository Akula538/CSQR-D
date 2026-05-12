#pragma once

#include "tools.hpp"

#include <opencv2/core.hpp>

namespace qrscanner {

class TPS {
public:
    void fit(const Points& from_pts, const Points& to_pts, double reg = 1e-3);
    Points tps_transform(const Points& points) const;
    void affine(double scale, double shift);
    cv::Mat warp_image_tps(const cv::Mat& src_img,
                           cv::Size out_shape,
                           double scale = 1.0,
                           double margins = 0.0,
                           int order = 1,
                           double cval = 255.0) const;

private:
    static double tps_kernel(double r);

    cv::Mat params_;
    Points ctrl_pts_;
    double scale_ = 1.0;
    double shift_ = 0.0;
};

cv::Mat tps_correct(const cv::Mat& img, const TPS& tps, int QRsize, int square_size = 10);
TPS fit_tps_full_qr(const Points& qr, int QRsize);
TPS fit_tps_alligment_center(const Points& qr, int QRsize);
TPS fit_tps_no_alligment(const Points& qr, int QRsize);

} // namespace qrscanner
