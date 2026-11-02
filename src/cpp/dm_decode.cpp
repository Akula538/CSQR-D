#include "dm_decode.hpp"
#include "Barcode.h"
#include "BarcodeFormat.h"
#include "DMTrace.hpp"
#include "ImageView.h"
#include "ReadBarcode.h"
#include "ReaderOptions.h"
#include "decoder.hpp"
#include "find_dm.hpp"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace datamatrix {
ScanResult decodeBaseline(const cv::Mat& image)
{
    if (image.empty() || image.depth() != CV_8U ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
        return {};
    auto text = qrscanner::decode_zxingcpp(image, ZXing::BarcodeFormat::DataMatrix);
    return text ? ScanResult{ScanStatus::Decoded, *text} : ScanResult{};
}

static ScanResult read_rectified(const cv::Mat& gray, cv::Size grid = {},
                                 qrscanner::GridDecodeCache* cache = nullptr, bool enhanced = true)
{
    const auto& image = gray;
    auto result = decodeBaseline(image);
    if (result.status == ScanStatus::Decoded)
        return result;
    if (!grid.empty())
        if (auto text = qrscanner::decode_matrix_grid(image, grid.width, grid.height,
                                                      ZXing::BarcodeFormat::DataMatrix, 6, cache))
            return {ScanStatus::Decoded, *text};
    if (enhanced)
        if (auto text = qrscanner::decode_qreader(image, ZXing::BarcodeFormat::DataMatrix, grid, 6, cache)) {
            return {ScanStatus::Decoded, *text};
        }
    ZXing::ReaderOptions options;
    options.setFormats(ZXing::BarcodeFormat::DataMatrix)
        .setIsPure(true)
        .setTryInvert(true)
        .setTextMode(ZXing::TextMode::HRI);
    auto decoded = ZXing::ReadBarcode(
        ZXing::ImageView(image.data, image.cols, image.rows, ZXing::ImageFormat::Lum, int(image.step)), options);
    return decoded.isValid() ? ScanResult{ScanStatus::Decoded, decoded.text()} : ScanResult{};
}

// A correctly phased grid concentrates image transitions at module boundaries,
// rather than at module centres. This scores geometry without knowing the data.
static double lattice_quality(const cv::Mat& image, int w, int h)
{
    double edges = 0, centres = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 1; x < w; ++x) {
            int py = 12 + y * 6 + 3, px = 12 + x * 6;
            edges += std::abs(int(image.at<uchar>(py, px + 1)) - image.at<uchar>(py, px - 1));
            centres += std::abs(int(image.at<uchar>(py, px + 4)) - image.at<uchar>(py, px + 2));
        }
    for (int y = 1; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int py = 12 + y * 6, px = 12 + x * 6 + 3;
            edges += std::abs(int(image.at<uchar>(py + 1, px)) - image.at<uchar>(py - 1, px));
            centres += std::abs(int(image.at<uchar>(py + 4, px)) - image.at<uchar>(py + 2, px));
        }
    return (edges - centres) / (edges + centres + 1);
}

static ScanResult read_nodes(GridCandidate& work, int iteration, const ScanOptions& options,
                      qrscanner::GridDecodeCache& decode_cache)
{
    int w = int(work.grid.x.centers.size()), h = int(work.grid.y.centers.size());
    DM_SCOPE(grid_trace, "grid", {},
             {{"w", w}, {"h", h}, {"field", 8}, {"iteration", iteration}, {"clock_cost", work.grid.score()}});
    qrscanner::TPS tps;
    tps.fit(work.nodes.first, work.nodes.second);
    auto corrected = tps.warp_image_tps(work.crop, {(w + 4) * 6, (h + 4) * 6}, 6, 12);
    corrected.rowRange(0, 12).setTo(255);
    corrected.rowRange(corrected.rows - 12, corrected.rows).setTo(255);
    corrected.colRange(0, 12).setTo(255);
    corrected.colRange(corrected.cols - 12, corrected.cols).setTo(255);
    work.quality = lattice_quality(corrected, w, h) - .5 * work.grid.score();
    cv::Scalar mean, deviation;
    cv::meanStdDev(corrected(cv::Rect(24, 24, (w - 4) * 6, (h - 4) * 6)), mean, deviation);
    work.payload_contrast = deviation[0];
    auto decoded = read_rectified(corrected, {w, h}, &decode_cache, true);
    DM_TRACE("rectified", corrected,
             {{"valid", decoded.status == ScanStatus::Decoded},
              {"field", 8},
              {"iteration", iteration},
              {"quality", work.quality}});
    if (options.debug_output_dir) {
        auto dir = std::filesystem::path(*options.debug_output_dir);
        std::filesystem::create_directories(dir);
        cv::imwrite(
            (dir / (std::to_string(work.detection_side) + "_" + std::to_string(work.rank) + "_" +
                    std::to_string(w) + "x" + std::to_string(h) + "_8_" + std::to_string(iteration) + ".png"))
                .string(),
            corrected);
    }
    return decoded;
}
cv::Mat prepare_image(const cv::Mat& image)
{
    if (image.empty() || image.depth() != CV_8U ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
        return {};
    cv::Mat gray;
    if (image.channels() == 1)
        gray = image;
    else
        cv::cvtColor(image, gray, image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    if (std::min(gray.cols, gray.rows) < 24)
        return {};
    bool dark_frame = true;
    for (const auto& edge : {gray.row(0), gray.row(gray.rows - 1), gray.col(0), gray.col(gray.cols - 1)}) {
        cv::Scalar mean, deviation;
        cv::meanStdDev(edge, mean, deviation);
        dark_frame &= mean[0] < 80 && deviation[0] < 8;
    }
    if (dark_frame) {
        cv::Mat inverted;
        cv::bitwise_not(gray, inverted);
        gray = inverted;
    }
    return gray;
}
ScanResult decode_candidate(const cv::Mat& gray, geometry::Candidate candidate, double detection_scale,
                            int detection_side, int rank, const ScanOptions& options,
                            qrscanner::GridDecodeCache& decode_cache, bool& crop_filtered,
                            std::vector<GridCandidate>& pending)
{
    ScanResult result;
    std::vector<cv::Point2f> outline;
    for (const auto& edge : candidate.edges)
        for (int i = 0; i <= 32; ++i)
            outline.push_back(edge.at(i / 32.));
    candidate.bounds |= cv::boundingRect(outline);
    int padding = int(std::max(candidate.bounds.width, candidate.bounds.height) * .3) + 8;
    auto bounds = (candidate.bounds + cv::Size(2 * padding, 2 * padding) - cv::Point(padding, padding)) &
                  cv::Rect(0, 0, gray.cols, gray.rows);
    cv::Mat crop = gray(bounds);
    result = decodeBaseline(crop);
    if (result.status == ScanStatus::Decoded)
        return result;
    candidate = geometry::translated(candidate, {-double(bounds.x), -double(bounds.y)});
    DM_SCOPE(trace, "candidate", crop,
             {{"rank", rank},
              {"score", candidate.score},
              {"crop_x", bounds.x},
              {"crop_y", bounds.y},
              {"proposal_kind", candidate.proposal_kind}});
    if (detection_scale < 1)
        candidate = geometry::refine(crop, candidate);
#ifdef CSQR_DM_TRACE
    diagnostics::Lines edges(4);
    for (int e = 0; e < 4; ++e)
        for (int t = 0; t <= 100; ++t)
            edges[e].push_back(candidate.edges[e].at(t / 100.));
    DM_TRACE("boundaries", crop, {}, {}, edges);
#endif
    // As in QR: try a boundary TPS once, then build the field on the
    // untouched crop if decoding fails. Never feed this render to a field.
    qrscanner::Points boundary_ideal, boundary_source;
    for (int edge = 0; edge < 4; ++edge)
        for (int i = 0; i < 4; ++i) {
            double t = i / 4.;
            boundary_ideal.emplace_back(edge == 0   ? t * 32
                                        : edge == 1 ? 32
                                        : edge == 2 ? (1 - t) * 32
                                                    : 0,
                                        edge == 0   ? 0
                                        : edge == 1 ? t * 32
                                        : edge == 2 ? 32
                                                    : (1 - t) * 32);
            boundary_source.push_back(candidate.edges[edge].at(t));
        }
    qrscanner::TPS boundary_tps;
    boundary_tps.fit(boundary_ideal, boundary_source);
    auto boundary_image = boundary_tps.warp_image_tps(crop, {216, 216}, 6, 12);
    result = read_rectified(boundary_image, {}, nullptr, options.thorough);
    DM_TRACE("boundary_render", boundary_image, {{"valid", result.status == ScanStatus::Decoded}});
    if (result.status == ScanStatus::Decoded)
        return result;
    auto grids = geometry::read_clocks(crop, candidate);
    if (!options.thorough && grids.size() > 1)
        grids.resize(1);
    for (auto& grid : grids) {
        GridCandidate work{candidate, grid,          crop, bounds, geometry::envelope_nodes(candidate, grid),
                         rank,      detection_side};
        result = read_nodes(work, 0, options, decode_cache);
        if (result.status == ScanStatus::Decoded)
            return result;
        // Decode low-contrast images first. If that failed and the
        // interior is almost uniform, tracing geometry has no reliable
        // payload transitions to recover. This is not an ECC verdict.
        if (work.payload_contrast < 8) {
            DM_TRACE("recovery_rejected", {}, {{"payload_contrast", work.payload_contrast}});
            continue;
        }
        // Unconstrained filtering of the large source crop is useful
        // only after clocks support a lattice. Do not spend seven
        // native finder calls on candidates whose clocks already failed.
        if (!crop_filtered) {
            crop_filtered = true;
            if (auto text = qrscanner::decode_qreader(crop, ZXing::BarcodeFormat::DataMatrix))
                return {ScanStatus::Decoded, *text};
        }
        work.nodes = geometry::align_lattice(crop, grid, work.nodes, 5);
        result = read_nodes(work, 1, options, decode_cache);
        if (result.status == ScanStatus::Decoded)
            return result;
        pending.push_back(std::move(work));
    }
    return {};
}
ScanResult refine_and_decode(GridCandidate& work, const ScanOptions& options,
                             qrscanner::GridDecodeCache& decode_cache)
{
    ScanResult result;
    DM_SCOPE(recovery_pass, "source_recovery", {}, {{"side", work.detection_side}});
    DM_SCOPE(
        recovery_trace, "recovery_candidate", work.crop,
        {{"rank", work.rank}, {"crop_x", work.bounds.x}, {"crop_y", work.bounds.y}, {"quality", work.quality}});
    qrscanner::DirectionField field(work.crop);
    work.grid = geometry::refine_clocks(field, work.crop, work.candidate, std::move(work.grid));
    auto traced = geometry::field_nodes(field, work.candidate, work.grid, 5, 20);
    if (!traced.first.empty())
        work.nodes = std::move(traced);
    result = read_nodes(work, 2, options, decode_cache);
    if (result.status == ScanStatus::Decoded)
        return result;
    for (int i = 0; i < (options.thorough ? 5 : 3); ++i) {
        DM_SCOPE(refinement_trace, "refinement", {},
                 {{"w", double(work.grid.x.centers.size())},
                  {"h", double(work.grid.y.centers.size())},
                  {"field", 8},
                  {"iteration", i + 3}});
        work.nodes = geometry::refine_lattice(work.crop, work.candidate, work.grid, work.nodes, field);
        result = read_nodes(work, i + 3, options, decode_cache);
        if (result.status == ScanStatus::Decoded)
            return result;
    }
    return {};
}
} // namespace datamatrix
