#pragma once
#include "DMscanner.hpp"
#include "decoder.hpp"
#include "dm_correct.hpp"

namespace datamatrix {
struct GridCandidate {
    geometry::Candidate candidate;
    geometry::Grid grid;
    cv::Mat crop;
    cv::Rect bounds;
    std::pair<qrscanner::Points, qrscanner::Points> nodes;
    int rank = 0, detection_side = 0;
    double quality = 0, payload_contrast = 0;
};

cv::Mat prepare_image(const cv::Mat& image);
ScanResult decode_candidate(const cv::Mat& gray, geometry::Candidate candidate, double detection_scale,
                            int detection_side, int rank, const ScanOptions& options,
                            qrscanner::GridDecodeCache& decode_cache, bool& crop_filtered,
                            std::vector<GridCandidate>& pending);
ScanResult refine_and_decode(GridCandidate& work, const ScanOptions& options,
                             qrscanner::GridDecodeCache& decode_cache);
} // namespace datamatrix
