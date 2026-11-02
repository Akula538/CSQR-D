#pragma once

#include "QRscanner.hpp"

namespace datamatrix
{

struct ScanOptions {
    // Preserve the fast result; on failure expand the search and use optical flow.
    bool thorough = false;
    std::optional<std::string> debug_output_dir;
};

// Accepts 8-bit gray, BGR or BGRA images. Returns the same status/text contract
// as the QR scanner. Data Matrix decoding is deliberately format-restricted.
qrscanner::ScanResult detectAndDecode(const cv::Mat &image, const ScanOptions &options = {});
qrscanner::ScanResult decodeBaseline(const cv::Mat &image);

} // namespace datamatrix
