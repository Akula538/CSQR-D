#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <string>

namespace qrscanner {

std::optional<std::string> decode_cv2(const cv::Mat& img);
std::optional<std::string> decode_qreader(const cv::Mat& img);
std::optional<std::string> decode_zxingcpp(const cv::Mat& img);
std::optional<std::string> decode(const cv::Mat& img);

} // namespace qrscanner
