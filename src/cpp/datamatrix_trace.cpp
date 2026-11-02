#include "BitMatrix.h"
#include "DMTrace.hpp"
#include "DMscanner.hpp"
#include "DecoderResult.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace datamatrix::diagnostics
{
static std::filesystem::path output;
static std::ofstream records;
static std::vector<int> parents;
static std::unordered_map<std::string, std::string> images;
static int sequence = 0;

static std::string quoted(const std::string &s)
{
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else
            out << c;
    }
    return out.str() + '"';
}

static std::string saveImage(const cv::Mat &image)
{
    // Use the complete pixels as a deduplication key: many hypotheses share
    // exactly the same mask. No images or evidence are dropped on a hash match.
    std::string key =
        std::to_string(image.rows) + "/" + std::to_string(image.cols) + "/" + std::to_string(image.type()) + ":";
    for (int y = 0; y < image.rows; ++y)
        key.append(reinterpret_cast<const char *>(image.ptr(y)), image.cols * image.elemSize());
    auto found = images.find(key);
    if (found != images.end())
        return found->second;
    auto name = "images/" + std::to_string(images.size()) + ".png";
    if (!cv::imwrite((output / name).string(), image, {cv::IMWRITE_PNG_COMPRESSION, 3}))
        throw std::runtime_error("Cannot write diagnostic image");
    images.emplace(std::move(key), name);
    return name;
}

static void number(double n)
{
    if (std::isfinite(n))
        records << n;
    else
        records << "null";
}
static void point(const cv::Point2d &p)
{
    records << '[';
    number(p.x);
    records << ',';
    number(p.y);
    records << ']';
}

int emit(const std::string &kind, const cv::Mat &image, const Numbers &numbers, const Strings &strings,
         const Lines &lines, const std::vector<cv::Point2d> &points)
{
    int id = ++sequence;
    records << "{\"id\":" << id << ",\"parent\":" << (parents.empty() ? 0 : parents.back())
            << ",\"kind\":" << quoted(kind) << ",\"numbers\":{";
    bool comma = false;
    for (const auto &[k, v] : numbers) {
        if (comma)
            records << ',';
        comma = true;
        records << quoted(k) << ':';
        number(v);
    }
    records << "},\"strings\":{";
    comma = false;
    for (const auto &[k, v] : strings) {
        if (comma)
            records << ',';
        comma = true;
        records << quoted(k) << ':' << quoted(v);
    }
    records << "},\"lines\":[";
    comma = false;
    for (const auto &line : lines) {
        if (comma)
            records << ',';
        comma = true;
        records << '[';
        for (size_t i = 0; i < line.size(); ++i) {
            if (i)
                records << ',';
            point(line[i]);
        }
        records << ']';
    }
    records << "],\"points\":[";
    for (size_t i = 0; i < points.size(); ++i) {
        if (i)
            records << ',';
        point(points[i]);
    }
    records << ']';
    if (!image.empty())
        records << ",\"image\":" << quoted(saveImage(image)) << ",\"width\":" << image.cols
                << ",\"height\":" << image.rows;
    records << "}\n";
    return id;
}

Scope::Scope(const std::string &kind, const cv::Mat &image, const Numbers &numbers, const Strings &strings,
             const Lines &lines, const std::vector<cv::Point2d> &points)
{
    int id = emit(kind, image, numbers, strings, lines, points);
    parents.push_back(id);
}
Scope::~Scope() { parents.pop_back(); }

void recordDecode(const ZXing::BitMatrix &bits, const ZXing::DecoderResult &decoded, double dx, double dy)
{
    cv::Mat modules(bits.height(), bits.width(), CV_8U);
    for (int y = 0; y < bits.height(); ++y)
        for (int x = 0; x < bits.width(); ++x)
            modules.at<uchar>(y, x) = bits.get(x, y) ? 0 : 255;
    emit("decode", (dx == 0 && dy == 0) || decoded.isValid() ? modules : cv::Mat(),
         {{"valid", decoded.isValid()}, {"error", int(decoded.error().type())}, {"dx", dx}, {"dy", dy}},
         {{"message", decoded.error().msg()},
          {"location", decoded.error().location()},
          {"text", decoded.isValid() ? decoded.content().text(ZXing::TextMode::HRI) : ""}});
}
} // namespace datamatrix::diagnostics

int main(int argc, char **argv)
{
    try {
        if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--thorough")) {
            std::cerr << "Usage: datamatrix_trace IMAGE OUTPUT_DIRECTORY [--thorough]\n";
            return 1;
        }
        namespace diag = datamatrix::diagnostics;
        diag::output = argv[2];
        std::filesystem::create_directories(diag::output / "images");
        diag::records.open(diag::output / "events.jsonl", std::ios::binary);
        diag::records << std::setprecision(7);
        if (!diag::records)
            throw std::runtime_error("Cannot open events.jsonl");
        cv::setNumThreads(1);
        auto image = cv::imread(argv[1], cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        if (image.empty())
            throw std::runtime_error("Cannot read input image");
        datamatrix::ScanOptions options;
        options.thorough = argc == 4;
        auto result = datamatrix::detectAndDecode(image, options);
        diag::emit("result", {},
                   {{"decoded", result.status == qrscanner::ScanStatus::Decoded},
                    {"thorough", options.thorough},
                    {"width", image.cols},
                    {"height", image.rows}},
                   {{"text", result.text}});
        std::cout << (result.status == qrscanner::ScanStatus::Decoded ? "Decoded" : "None") << ": "
                  << diag::sequence << " events, " << diag::images.size() << " images\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
