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
#include <random>
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

static cv::Mat gray_to_bgr_if_needed(const cv::Mat& img)
{
    if (img.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    return img;
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

ScanResult detectAndDecode(const cv::Mat& input, const ScanOptions& options)
{
    if (input.empty())
        return {ScanStatus::Ctf, "CTF"};

    if (options.debug_output_dir) {
        std::filesystem::create_directories(*options.debug_output_dir);
        std::ofstream(std::filesystem::path(*options.debug_output_dir) / "trace.txt", std::ios::trunc);
    }

    // std::mt19937 rng(options.random_seed.value_or(std::random_device{}()));
    // std::uniform_real_distribution<double> rnd(0.0, 1.0);

    const int width = input.cols;
    const int height = input.rows;
    const double scale = std::min(1000.0 / width, 1000.0 / height);

    cv::Mat img = to_gray(input);
    cv::Mat img_downscaled;
    cv::resize(img,
               img_downscaled,
               cv::Size(static_cast<int>(width * scale), static_cast<int>(height * scale)),
               0,
               0,
               scale < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC);

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

    // if (rnd(rng) > 0.05 && zxing_qr.patterns.size() < 2) {
    //     if (rnd(rng) < 0.08) {
    //         auto result = decode(img_downscaled);
    //         if (result)
    //             return decoded_result(*result);
    //     }
    //     write_debug_text(options, "trace.txt", "early CTF: zxing patterns < 3\n");
    //     return {ScanStatus::Ctf, "CTF"};
    // }

    // return {ScanStatus::None, ""};

    auto result = decode_zxingcpp(img_downscaled);
    if (result)
        return decoded_result(*result);

    if (zxing_qr.patterns.size() >= 2) {
        result = decode_qreader(img_downscaled);
        if (result)
            return decoded_result(*result);
    }

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
        write_debug_text(options, "trace.txt", "recognize_distribution failed\n");
        return {ScanStatus::Ctf, "CTF"};
    }

    TPS tps;
    if (distribution.size <= 21){
        tps = fit_tps_no_alligment(qr, distribution.size);
    }
    if ((int)distribution.size >= 25) {
        tps = fit_tps_full_qr(qr, distribution.size);
    }

    cv::Mat corrected = tps_correct(img_downscaled, tps, distribution.size, 6);
    write_debug(options, "corrected_full_tps.png", corrected);

    result = decode_zxingcpp(corrected);
    if (result)
        return decoded_result(*result);
    result = decode_qreader(corrected);
    if (result)
        return decoded_result(*result);

    TPS tps1 = fit_tps_alligment_center(qr, distribution.size);
    corrected = tps_correct(img_downscaled, tps1, distribution.size, 6);
    write_debug(options, "corrected_alignment_tps.png", corrected);

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
    tps.affine(1.0 / scale, 0.0);

    auto cut = cut_image(img, qr, pattern_recovered, distribution.size);
    auto nodes = vectorfield_nodes(cut.img, cut.qr, cut.pattern_recovered, distribution.size, 6, 5, 20, &tps);

    TPS vector_tps;
    vector_tps.fit(nodes.first, nodes.second);
    corrected = tps_correct(cut.img, vector_tps, distribution.size, 6);
    write_debug(options, "corrected_vector_tps.png", corrected);

    result = decode_zxingcpp(corrected);
    if (result)
        return decoded_result(*result);
    result = decode_qreader(corrected);
    if (result)
        return decoded_result(*result);

    return {ScanStatus::None, ""};
}

} // namespace qrscanner
