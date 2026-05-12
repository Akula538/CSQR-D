#include "decoder.hpp"

#include "Barcode.h"
#include "BarcodeFormat.h"
#include "ImageView.h"
#include "ReadBarcode.h"
#include "ReaderOptions.h"

#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace qrscanner {


cv::Mat croplocate_cv2(const cv::Mat& image)
{
    if (image.empty())
        return image;

    cv::QRCodeDetector detector;

    // Немного помогает на сложных QR
    detector.setUseAlignmentMarkers(true);

    std::vector<cv::Point> points;
    bool found = detector.detect(image, points);

    // Не нашли QR -> возвращаем исходник
    if (!found || points.size() != 4)
        return image;

    // Bounding rect вокруг найденного quad
    cv::Rect bbox = cv::boundingRect(points);

    // Добавляем padding
    int pad_x = static_cast<int>(bbox.width * 0.15);
    int pad_y = static_cast<int>(bbox.height * 0.15);

    bbox.x -= pad_x;
    bbox.y -= pad_y;
    bbox.width += pad_x * 2;
    bbox.height += pad_y * 2;

    // Клип к границам изображения
    bbox &= cv::Rect(0, 0, image.cols, image.rows);

    if (bbox.width <= 0 || bbox.height <= 0)
        return image;

    return image(bbox).clone();
}

static cv::Mat continuous_for_zxing(const cv::Mat& img, ZXing::ImageFormat& format)
{
    cv::Mat view = img;
    if (img.channels() == 1) {
        format = ZXing::ImageFormat::Lum;
    } else if (img.channels() == 3) {
        format = ZXing::ImageFormat::BGR;
    } else if (img.channels() == 4) {
        format = ZXing::ImageFormat::BGRA;
    } else {
        cv::cvtColor(img, view, cv::COLOR_BGR2GRAY);
        format = ZXing::ImageFormat::Lum;
    }
    if (!view.isContinuous())
        view = view.clone();
    return view;
}

std::optional<std::string> decode_cv2(const cv::Mat& img)
{
    cv::Mat gray;
    if (img.channels() == 1)
        gray = img;
    else
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::QRCodeDetector detector;
    std::string data = detector.detectAndDecode(gray);
    if (!data.empty())
        return data;
    return std::nullopt;
}

std::optional<std::string> decode_zxingcpp(const cv::Mat& img)
{
    ZXing::ImageFormat format = ZXing::ImageFormat::None;
    cv::Mat buffer = continuous_for_zxing(img, format);
    ZXing::ImageView view(buffer.data,
                          static_cast<int>(buffer.total() * buffer.elemSize()),
                          buffer.cols,
                          buffer.rows,
                          format,
                          static_cast<int>(buffer.step),
                          static_cast<int>(buffer.elemSize()));

    ZXing::ReaderOptions options;
    options.setFormats(ZXing::BarcodeFormat::QRCode)
           .setTryHarder(true)
           .setTryRotate(true)
           .setTryInvert(true)
           .setTryDownscale(true)
           .setTextMode(ZXing::TextMode::HRI);

    auto result = ZXing::ReadBarcode(view, options);
    if (result.isValid())
        return result.text();
    return std::nullopt;
}

static std::optional<std::string> try_decode(const cv::Mat& img)
{
    auto result = decode_zxingcpp(img);
    if (result && !result->empty())
        return result;
    return std::nullopt;
}

static std::optional<std::string> try_gray_variants(const cv::Mat& gray)
{
    // if (auto r = try_decode(gray); r)
    //     return r;

    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (auto r = try_decode(binary); r)
        return r;

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);
    if (auto r = try_decode(blurred); r)
        return r;

    cv::GaussianBlur(gray, blurred, cv::Size(7, 7), 0.0);
    if (auto r = try_decode(blurred); r)
        return r;

    cv::Mat sharpened;
    static const cv::Mat sharpen_kernel = (cv::Mat_<float>(3, 3) <<
        -1.f, -1.f, -1.f,
        -1.f,  9.f, -1.f,
        -1.f, -1.f, -1.f
    );

    cv::filter2D(gray, sharpened, -1, sharpen_kernel);
    if (auto r = try_decode(sharpened); r)
        return r;

    cv::threshold(sharpened, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (auto r = try_decode(binary); r)
        return r;

    cv::GaussianBlur(sharpened, blurred, cv::Size(5, 5), 0.0);
    if (auto r = try_decode(blurred); r)
        return r;

    cv::GaussianBlur(sharpened, blurred, cv::Size(7, 7), 0.0);
    if (auto r = try_decode(blurred); r)
        return r;

    return std::nullopt;
}

std::optional<std::string> decode_qreader(const cv::Mat& img)
{
    // return std::nullopt;
    if (img.empty() || img.depth() != CV_8U)
        return std::nullopt;

    // if (auto r = try_decode(img); r)
    //     return r;

    // cv::Mat cropped = croplocate_cv2(img);
    // cv::imwrite("debug/cropped.png", cropped);

    cv::Mat inverted;
    cv::bitwise_not(img, inverted);
    if (auto r = try_decode(inverted); r)
        return r;

    // cv::Mat gray;
    // if (img.channels() == 1) {
    //     gray = img;
    // } else if (img.channels() == 3) {
    //     cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // } else if (img.channels() == 4) {
    //     cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    // } else {
    //     cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // }

    return try_gray_variants(img);
}

std::optional<std::string> decode(const cv::Mat& img)
{
    auto result = decode_zxingcpp(img);
    if (result)
        return result;
    return std::nullopt;
}

} // namespace qrscanner
