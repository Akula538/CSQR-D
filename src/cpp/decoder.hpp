#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <string>

namespace qrscanner {

std::optional<std::string> decode_cv2(const cv::Mat& img);
std::optional<std::string> decode_qreader(const cv::Mat& img);
std::optional<std::string> decode_zxingcpp(const cv::Mat& img);
// Rectified image with two quiet-zone modules on each side.
std::optional<std::string> decode_qr_grid(const cv::Mat& img, int dimension, int module_size = 6);
std::optional<std::string> decode(const cv::Mat& img);

} // namespace qrscanner
