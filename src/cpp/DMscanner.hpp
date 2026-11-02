#pragma once

#include "scan_result.hpp"
#include <opencv2/core.hpp>
#include <optional>
#include <string>

namespace datamatrix {

using qrscanner::ScanResult;
using qrscanner::ScanStatus;

struct ScanOptions {
    // Off: bounded camera-frame work. On: more candidates, clock-size hypotheses and
    // refinement iterations, intended for a still image rather than a stream.
    bool thorough = false;
    std::optional<std::string> debug_output_dir;
};

// Accepts 8-bit gray, BGR or BGRA images. Returns the same status/text contract
// as the QR scanner. Data Matrix decoding is deliberately format-restricted.
ScanResult detectAndDecode(const cv::Mat& image, const ScanOptions& options = {});
ScanResult decodeBaseline(const cv::Mat& image);

} // namespace datamatrix
