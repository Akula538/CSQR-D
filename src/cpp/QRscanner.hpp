#pragma once

#include "scan_result.hpp"
#include <opencv2/core.hpp>

#include <optional>
#include <string>

namespace qrscanner {

struct ScanOptions {
    bool can_homo = false;
    std::optional<unsigned int> random_seed;
    std::optional<std::string> debug_output_dir;
};

ScanResult detectAndDecode(const cv::Mat& img, const ScanOptions& options = {});

} // namespace qrscanner
