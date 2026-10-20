#include "QRscanner.hpp"

#include "TPS_correct.hpp"
#include "binarizer.hpp"
#include "decoder.hpp"
#include "find_qr.hpp"
#include "interpolation_correct.hpp"
#include "tools.hpp"
#include "vectorfield_correct.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace qrscanner {

static cv::Mat to_gray(const cv::Mat& img)
{
    if (img.channels() == 1)
        return img.clone();
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4)
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    else
        gray = img.clone();
    return gray;
}

static void write_debug(const ScanOptions& options, const std::string& name, const cv::Mat& img)
{
    if (!options.debug_output_dir)
        return;
    std::filesystem::create_directories(*options.debug_output_dir);
    cv::imwrite((std::filesystem::path(*options.debug_output_dir) / name).string(), img);
}

static void write_debug_text(const ScanOptions& options, const std::string& name, const std::string& text)
{
    if (!options.debug_output_dir)
        return;
    std::filesystem::create_directories(*options.debug_output_dir);
    std::ofstream file(std::filesystem::path(*options.debug_output_dir) / name, std::ios::app);
    file << text;
}

static ScanResult decoded_result(const std::string& text)
{
    return {ScanStatus::Decoded, text};
}

static ScanResult scan_once(const cv::Mat& input, const ScanOptions& options, int detection_side)
{
    if (input.empty())
        return {ScanStatus::Ctf, "CTF"};

    write_debug_text(options, "trace.txt", "detection side: " + std::to_string(detection_side) + "\n");

    const int width = input.cols;
    const int height = input.rows;
    const double scale = std::min(double(detection_side) / width, double(detection_side) / height);

    cv::Mat img = to_gray(input);
    cv::Mat img_downscaled;
    cv::resize(img,
               img_downscaled,
               cv::Size(std::max(1, static_cast<int>(width * scale)), std::max(1, static_cast<int>(height * scale))),
               0,
               0,
               scale < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC);

    auto result = decode_zxingcpp(img_downscaled);
    if (result)
        return decoded_result(*result);

    ZxingFindResult zxing_qr = zxing_find_with_binary(img_downscaled);
    write_debug(options, "zxing_binary.png", zxing_qr.binary);
    if (options.debug_output_dir) {
        write_debug_text(options, "trace.txt", "zxing patterns: " + std::to_string(zxing_qr.patterns.size()) + "\n");
        for (const auto& p : zxing_qr.patterns) {
            write_debug_text(options, "trace.txt",
                             std::to_string(p.center.x) + "," + std::to_string(p.center.y) + " " +
                             std::to_string(p.size) + " corners=" + std::to_string(p.corners.size()) + "\n");
        }
    }

    auto decode_original_variants = [&]() -> std::optional<std::string> {
        return zxing_qr.patterns.size() >= 2 ? decode_qreader(img_downscaled) : std::nullopt;
    };

    Points qr;
    cv::Mat img_binarized;
    try {
        auto found = find_qr_zxing(img_downscaled, &zxing_qr);
        qr = found.first;
        img_binarized = found.second;
    } catch (...) {
        cv::Mat inverted_downscaled;
        cv::bitwise_not(img_downscaled, inverted_downscaled);
        try {
            auto found = find_qr_zxing(inverted_downscaled);
            qr = found.first;
            img_binarized = found.second;
            cv::Mat inverted;
            cv::bitwise_not(img, inverted);
            img = inverted;
        } catch(...){
            try {
                img_binarized = binarize_zxing_cpp(img_downscaled);
                qr = find(img_binarized);
            } catch (...) {
                if (auto decoded = decode_original_variants())
                    return decoded_result(*decoded);
                write_debug_text(options, "trace.txt", "find_qr fallback failed\n");
                return {ScanStatus::Ctf, "CTF"};
            }
        }
    }

    DistributionResult distribution;
    SearchDistributionResult search_distribution;
    std::vector<Points> pattern_recovered;
    try {
        distribution = recognize_distribution(img_binarized, qr);
        if (distribution.size < 21 || distribution.size > 177)
            throw std::runtime_error("Invalid QR dimension");
        search_distribution = recognize_search_distribution(img_binarized, qr);

        std::vector<Points*> pattern_ptrs(2, nullptr);
        std::vector<Points*> search_ptrs(4, nullptr);
        for (int i = 0; i < 2; ++i)
            if (distribution.pattern[i])
                pattern_ptrs[i] = &*distribution.pattern[i];
        for (int i = 0; i < 4; ++i)
            if (search_distribution.pattern[i])
                search_ptrs[i] = &*search_distribution.pattern[i];
        pattern_recovered = recover_pattern_if_need(qr, distribution.size, pattern_ptrs, search_ptrs);
    } catch (...) {
        if (auto decoded = decode_original_variants())
            return decoded_result(*decoded);
        write_debug_text(options, "trace.txt", "recognize_distribution failed\n");
        return {ScanStatus::Ctf, "CTF"};
    }

    TPS tps;
    write_debug_text(options, "trace.txt", "grid size: " + std::to_string(distribution.size) +
                     " landmarks: " + std::to_string(qr.size()) + "\n");
    if (distribution.size <= 21){
        tps = fit_tps_no_alligment(qr, distribution.size);
    }
    if ((int)distribution.size >= 25) {
        tps = fit_tps_full_qr(qr, distribution.size);
    }

    cv::Mat corrected = tps_correct(img_downscaled, tps, distribution.size, 6);
    write_debug(options, "corrected_full_tps.png", corrected);

    result = decode_qr_grid(corrected, distribution.size);
    if (result)
        return decoded_result(*result);
    result = decode_zxingcpp(corrected);
    if (result)
        return decoded_result(*result);
    result = decode_qreader(corrected);
    if (result)
        return decoded_result(*result);

    // High-resolution preprocessing is useful for some noisy images but
    // wasteful when the cheap first geometric correction already decodes.
    result = decode_original_variants();
    if (result)
        return decoded_result(*result);

    TPS tps1 = fit_tps_alligment_center(qr, distribution.size);
    corrected = tps_correct(img_downscaled, tps1, distribution.size, 6);
    write_debug(options, "corrected_alignment_tps.png", corrected);

    result = decode_qr_grid(corrected, distribution.size);
    if (result)
        return decoded_result(*result);
    result = decode_zxingcpp(corrected);
    if (result)
        return decoded_result(*result);
    result = decode_qreader(corrected);
    if (result)
        return decoded_result(*result);

    for (auto& p : qr)
        p *= (1.0 / scale);
    for (auto& pattern : pattern_recovered)
        for (auto& p : pattern)
            p *= (1.0 / scale);
            
    auto cut = cut_image(img, qr, pattern_recovered, distribution.size);
    tps.affine(1.0 / scale, -cut.shift);
    DirectionField df(cut.img);

    try {
        HomographyMatrix cut_to_binary(scale, cut.shift * scale);
        auto pattern_polylines = constructPolylines4recdist(df, cut.qr, 30);
        const int previous_size = distribution.size;
        distribution = recognize_distribution_polyline(img_binarized, cut.qr, &cut_to_binary, pattern_polylines);
        if (distribution.size < 21 || distribution.size > 177)
            throw std::runtime_error("Invalid QR dimension");

        for (auto& pattern : search_distribution.pattern)
            if (pattern)
                for (auto& p : *pattern)
                    p = p * (1.0 / scale) - cut.shift;

        std::vector<Points*> pattern_ptrs(2, nullptr);
        std::vector<Points*> search_ptrs(4, nullptr);
        for (int i = 0; i < 2; ++i)
            if (distribution.pattern[i])
                pattern_ptrs[i] = &*distribution.pattern[i];
        for (int i = 0; i < 4; ++i)
            if (search_distribution.pattern[i])
                search_ptrs[i] = &*search_distribution.pattern[i];
        cut.pattern_recovered = recover_pattern_if_need(cut.qr, distribution.size, pattern_ptrs, search_ptrs);
        if (distribution.size != previous_size)
            tps = distribution.size <= 21 ? fit_tps_no_alligment(cut.qr, distribution.size)
                                          : fit_tps_full_qr(cut.qr, distribution.size);
    } catch (...) {
        write_debug_text(options, "trace.txt", "recognize_distribution failed\n");
        return {ScanStatus::Ctf, "CTF"};
    }

    for (const auto& settings : {std::pair<int, int>{5, 20}, {3, std::max(40, 2 * distribution.size)}}) {
        auto nodes = vectorfield_nodes(df, cut.qr, cut.pattern_recovered, distribution.size, 6,
                                      settings.first, settings.second, &tps);

        TPS vector_tps;
        vector_tps.fit(nodes.first, nodes.second);
        corrected = tps_correct(cut.img, vector_tps, distribution.size, 6);
        write_debug(options, settings.first == 5 ? "corrected_vector_tps.png" : "corrected_vector_fine_tps.png", corrected);

        result = decode_qr_grid(corrected, distribution.size);
        if (result)
            return decoded_result(*result);
        result = decode_zxingcpp(corrected);
        if (result)
            return decoded_result(*result);
        result = decode_qreader(corrected);
        if (result)
            return decoded_result(*result);
    }

    return {ScanStatus::None, ""};
}

ScanResult detectAndDecode(const cv::Mat& input, const ScanOptions& options)
{
    if (options.debug_output_dir) {
        std::filesystem::create_directories(*options.debug_output_dir);
        std::ofstream(std::filesystem::path(*options.debug_output_dir) / "trace.txt", std::ios::trunc);
    }
    auto run = [&](int side) {
        try {
            return scan_once(input, options, side);
        } catch (const std::exception& e) {
            write_debug_text(options, "trace.txt", std::string("geometry error: ") + e.what() + "\n");
            return ScanResult{ScanStatus::Ctf, "CTF"};
        }
    };
    auto result = run(1000);
    // A failed localization/timing scan is often a sampling-resolution issue.
    // Keep the ordinary path cheap; retry only when no usable grid was found.
    if (!input.empty() && result.status == ScanStatus::Ctf)
        result = run(1500);
    return result;
}

} // namespace qrscanner
