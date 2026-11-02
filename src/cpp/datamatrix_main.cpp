#include "DMscanner.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef CSQR_DM_COMPARE_LEGACY
qrscanner::ScanResult decodeLegacy(const cv::Mat &image);
#endif

static std::string json_string(const std::string &text)
{
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else
            out << c;
    }
    return out.str() + '"';
}

int main(int argc, char **argv)
{
    try {
        std::string path;
        bool baseline = false, json = false, compare = false;
        int repeats = 1;
        datamatrix::ScanOptions options;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--thorough")
                options.thorough = true;
            else if (arg == "--baseline")
                baseline = true;
            else if (arg == "--compare")
                compare = true;
            else if (arg == "--json")
                json = true;
            else if (arg == "--repeat" && i + 1 < argc)
                repeats = std::stoi(argv[++i]);
            else if (arg == "--debug-output" && i + 1 < argc)
                options.debug_output_dir = argv[++i];
            else if (path.empty() && arg.rfind("--", 0) != 0)
                path = arg;
            else
                throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
        if (path.empty() || repeats < 1 || (baseline && compare)) {
            std::cerr << "Usage: datamatrix_cli IMAGE_OR_DIRECTORY [--baseline|--compare] [--json] [--repeat N] "
                         "[--thorough] [--debug-output DIR]\n";
            return 1;
        }
        cv::setNumThreads(1);
        int scannerCount = compare ? 2 : 1;
#ifdef CSQR_DM_COMPARE_LEGACY
        if (!compare || options.thorough)
            throw std::runtime_error("Legacy comparison requires --compare without --thorough");
        scannerCount = 3;
#endif
        std::vector<std::filesystem::path> files;
        if (std::filesystem::is_directory(path)) {
            for (const auto &entry : std::filesystem::recursive_directory_iterator(path)) {
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp"))
                    files.push_back(entry.path());
            }
        } else
            files.emplace_back(path);
        std::sort(files.begin(), files.end());
        if (files.empty())
            throw std::runtime_error("No supported images in " + path);
        bool warmed = false;
        for (const auto &file : files) {
            const auto image = cv::imread(file.string(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
            if (image.empty())
                throw std::runtime_error("Cannot read " + file.string());
            auto localOptions = options;
            if (options.debug_output_dir && files.size() > 1)
                localOptions.debug_output_dir =
                    (std::filesystem::path(*options.debug_output_dir) / file.stem()).string();
            if (compare && !warmed) {
                datamatrix::decodeBaseline(image);
                auto warmOptions = options;
                warmOptions.debug_output_dir.reset();
                datamatrix::detectAndDecode(image, warmOptions);
#ifdef CSQR_DM_COMPARE_LEGACY
                decodeLegacy(image);
#endif
                warmed = true;
            }
            for (int i = 0; i < repeats; ++i) {
                for (int pass = 0; pass < scannerCount; ++pass) {
                    const int scanner = compare ? (i + pass) % scannerCount : (baseline ? 0 : 1);
                    const char *scannerName = scanner == 0 ? "ZXing-C++" : "CSQR-D-DataMatrix";
                    const auto start = std::chrono::steady_clock::now();
                    qrscanner::ScanResult result;
#ifdef CSQR_DM_COMPARE_LEGACY
                    if (scanner == 2) {
                        scannerName = "Legacy-DataMatrix";
                        result = decodeLegacy(image);
                    } else
#endif
                        result = scanner == 0 ? datamatrix::decodeBaseline(image)
                                              : datamatrix::detectAndDecode(image, localOptions);
                    const double ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                            .count();
                    const bool ok = result.status == qrscanner::ScanStatus::Decoded;
                    if (json)
                        std::cout << "{\"file\":"
                                  << json_string(std::filesystem::is_directory(path)
                                                     ? file.lexically_relative(std::filesystem::path(path))
                                                           .generic_string()
                                                     : file.filename().string())
                                  << ",\"scanner\":" << json_string(scannerName) << ",\"repeat\":" << i
                                  << ",\"width\":" << image.cols << ",\"height\":" << image.rows
                                  << ",\"decoded\":" << (ok ? "true" : "false")
                                  << ",\"text\":" << json_string(result.text) << ",\"ms\":" << ms << "}\n";
                    else
                        std::cout << (ok ? result.text : "None") << '\n';
                    std::cout.flush();
                }
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
