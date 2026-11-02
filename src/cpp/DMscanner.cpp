#include "DMscanner.hpp"
#include "DMTrace.hpp"
#include "dm_decode.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace datamatrix {

ScanResult detectAndDecode(const cv::Mat& image, const ScanOptions& options)
{
    auto gray = prepare_image(image);
    if (gray.empty())
        return {};
    qrscanner::GridDecodeCache decode_cache;
    const int candidate_limit = options.thorough ? 12 : 2;
    const int field_limit = options.thorough ? 12 : 1;
    int field_attempts = 0;
    bool crop_filtered = false;
    ScanResult result;
    std::vector<GridCandidate> pending;
    for (int detection_side : {1000, 1500}) {
        double detection_scale = std::min(1., double(detection_side) / std::max(gray.cols, gray.rows));
        cv::Mat detection_image;
        cv::resize(gray, detection_image,
                   {cvRound(gray.cols * detection_scale), cvRound(gray.rows * detection_scale)}, 0, 0,
                   cv::INTER_AREA);
        DM_SCOPE(search_trace, "source_search", detection_image,
                 {{"scale", detection_scale},
                  {"side", detection_side},
                  {"thorough", options.thorough},
                  {"candidate_limit", candidate_limit}});
        result = decodeBaseline(detection_image);
        if (result.status == ScanStatus::Decoded)
            return result;
        auto candidates = geometry::locate(detection_image);
        for (auto& candidate : candidates)
            candidate = geometry::to_source(candidate, detection_scale);
        for (int rank = 0; rank < std::min(candidate_limit, int(candidates.size())); ++rank) {
            result = decode_candidate(gray, candidates[rank], detection_scale, detection_side, rank, options,
                                      decode_cache, crop_filtered, pending);
            if (result.status == ScanStatus::Decoded)
                return result;
        }
        if (detection_scale == 1)
            break;
    }
    std::stable_sort(pending.begin(), pending.end(),
                     [](const auto& a, const auto& b) { return a.quality > b.quality; });
    for (auto& work : pending) {
        if (field_attempts++ >= field_limit)
            break;
        result = refine_and_decode(work, options, decode_cache);
        if (result.status == ScanStatus::Decoded)
            return result;
    }
    return options.thorough ? decodeBaseline(image) : ScanResult{};
}
} // namespace datamatrix
