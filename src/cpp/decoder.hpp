#pragma once

#include "BarcodeFormat.h"
#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <unordered_set>

namespace qrscanner
{

// Per scan, never global. Keys contain dimensions and every sampled bit;
// only failed matrices are cached, so successful reads remain repeatable.
struct GridDecodeCache {
    std::unordered_set<std::string> failed;
};

std::optional<std::string> decode_cv2(const cv::Mat &img);
std::optional<std::string> decode_qreader(const cv::Mat &img,
                                          ZXing::BarcodeFormat format = ZXing::BarcodeFormat::QRCode,
                                          cv::Size grid = {}, int module_size = 6,
                                          GridDecodeCache *cache = nullptr);
std::optional<std::string> decode_zxingcpp(const cv::Mat &img,
                                           ZXing::BarcodeFormat format = ZXing::BarcodeFormat::QRCode);
// Rectified image with two quiet-zone modules on each side.
std::optional<std::string> decode_qr_grid(const cv::Mat &img, int dimension, int module_size = 6);
std::optional<std::string> decode_matrix_grid(const cv::Mat &img, int columns, int rows,
                                              ZXing::BarcodeFormat format, int module_size = 6,
                                              GridDecodeCache *cache = nullptr);
std::optional<std::string> decode(const cv::Mat &img);

} // namespace qrscanner
