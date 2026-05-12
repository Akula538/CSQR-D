#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <string>

namespace qrscanner {

enum class ScanStatus {
    Decoded,
    Ctf,
    None
};

struct ScanOptions {
    bool can_homo = false;
    std::optional<unsigned int> random_seed;
    std::optional<std::string> debug_output_dir;
};

struct ScanResult {
    ScanStatus status = ScanStatus::None;
    std::string text;
};

ScanResult detectAndDecode(const cv::Mat& img, const ScanOptions& options = {});

} // namespace qrscanner
