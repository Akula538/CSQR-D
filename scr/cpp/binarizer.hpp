#pragma once

#include <opencv2/core.hpp>

namespace qrscanner {

cv::Mat binarize(const cv::Mat& orig);
cv::Mat binarize_mean(const cv::Mat& orig, int offset = 10, int window_size = 0);
cv::Mat binarize_smart(const cv::Mat& orig, int offset = 10, int window_size = 0);
cv::Mat binarize_sauvola(const cv::Mat& image, int window = 0, double k = 0.3, double R = 128.0);
cv::Mat binarize_zxing_cpp(const cv::Mat& img);

} // namespace qrscanner
