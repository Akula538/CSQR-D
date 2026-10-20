#include "TPS_correct.hpp"
#include "decoder.hpp"
#include "tools.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <stdexcept>

using namespace qrscanner;

static void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void check_tps(bool lattice)
{
    Points from = {{0, 0}, {25, 0}, {25, 25}, {0, 25}, {12, 12}, {7, 18}};
    if (!lattice)
        from[4].x += 0.123;
    Points to = {{20, 20}, {170, 25}, {165, 170}, {25, 165}, {90, 105}, {65, 135}};
    TPS tps;
    tps.fit(from, to);
    cv::Mat source(200, 200, CV_32FC2);
    for (int y = 0; y < source.rows; ++y)
        for (int x = 0; x < source.cols; ++x)
            source.at<cv::Vec2f>(y, x) = {float(x), float(y)};

    Points grid;
    const int size = 174;
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            grid.emplace_back((x - 12.) / 6, (y - 12.) / 6);
    const auto reference = tps.tps_transform(grid);
    cv::Mat mx(size, size, CV_32F), my(size, size, CV_32F);
    for (int i = 0; i < size * size; ++i) {
        mx.ptr<float>()[i] = float(reference[i].x);
        my.ptr<float>()[i] = float(reference[i].y);
    }
    cv::Mat expected;
    cv::remap(source, expected, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(255));
    auto actual = tps.warp_image_tps(source, {size, size}, 6, 12);
    require(cv::norm(actual, expected, cv::NORM_INF) <= 0.032, "TPS warp differs from dense reference");

    const auto before = tps.tps_transform({{9, 11}})[0];
    tps.affine(2, {3, 4});
    tps.affine(0.5, {-2, 1});
    const auto after = tps.tps_transform({{9, 11}})[0];
    require(norm(after - (before + Point(-0.5, 3))) < 1e-8, "TPS affine composition");
    tps.fit(from, to);
    require(norm(tps.tps_transform({{9, 11}})[0] - before) < 1e-8, "TPS fit must reset affine");
}

static void check_grid_decoder()
{
    const std::string text = "CSQR-D grid check";
    for (int version : {1, 2, 5}) {
        cv::QRCodeEncoder::Params params;
        params.version = version;
        params.correction_level = cv::QRCodeEncoder::CORRECT_LEVEL_L;
        cv::Mat symbol;
        const std::string payload = version == 1 ? "CSQR-D" : text;
        cv::QRCodeEncoder::create(params)->encode(payload, symbol);
        // OpenCV includes a two-module quiet zone.
        const int dimension = 17 + 4 * version;
        require(symbol.rows == dimension + 4, "Unexpected QR encoder border");
        cv::Mat image;
        cv::resize(symbol, image, {}, 6, 6, cv::INTER_NEAREST);
        auto decoded = decode_qr_grid(image, dimension);
        require(decoded && *decoded == payload, "Known QR grid payload");
        cv::GaussianBlur(image, image, {3, 3}, 0.7);
        decoded = decode_qr_grid(image, dimension);
        require(decoded && *decoded == payload, "Blurred QR grid payload");
    }
    require(!decode_qr_grid(cv::Mat::zeros(174, 174, CV_8U), 25), "Blank grid must not decode");
    require(!decode_qr_grid(cv::Mat::zeros(174, 174, CV_8U), 26), "Invalid QR dimension");
}

static void check_crop()
{
    cv::Mat image(100, 100, CV_8U, cv::Scalar(255));
    Points qr = {{20.7, 22.4}, {70.3, 22.4}, {70.3, 75.2}, {20.7, 75.2}};
    const auto result = cut_image(image, qr, {}, 25);
    require(result.shift.x == int(result.shift.x) && result.shift.y == int(result.shift.y),
            "Crop shift must match integer ROI");
    require(norm(result.qr[0] + result.shift - qr[0]) < 1e-10, "Crop landmark translation");
}

int main()
{
    try {
        check_tps(true);
        check_tps(false);
        check_grid_decoder();
        check_crop();
        std::cout << "Scanner checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
