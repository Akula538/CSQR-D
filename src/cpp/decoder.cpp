#include "decoder.hpp"

#include "Barcode.h"
#include "BarcodeFormat.h"
#include "BitMatrix.h"
#include "DecoderResult.h"
#include "ImageView.h"
#include "ReadBarcode.h"
#include "ReaderOptions.h"
#include "datamatrix/DMDecoder.h"
#include "datamatrix/DMVersion.h"
#include "qrcode/QRDecoder.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

namespace qrscanner
{

cv::Mat croplocate_cv2(const cv::Mat &image)
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

static cv::Mat continuous_for_zxing(const cv::Mat &img, ZXing::ImageFormat &format)
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

std::optional<std::string> decode_cv2(const cv::Mat &img)
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

std::optional<std::string> decode_zxingcpp(const cv::Mat &img, ZXing::BarcodeFormat barcodeFormat)
{
    ZXing::ImageFormat format = ZXing::ImageFormat::None;
    cv::Mat buffer = continuous_for_zxing(img, format);
    ZXing::ImageView view(buffer.data, static_cast<int>(buffer.total() * buffer.elemSize()), buffer.cols,
                          buffer.rows, format, static_cast<int>(buffer.step), static_cast<int>(buffer.elemSize()));

    ZXing::ReaderOptions options;
    options.setFormats(barcodeFormat)
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

std::optional<std::string> decode_qr_grid(const cv::Mat &img, int dimension, int module_size)
{
    if (img.type() != CV_8UC1 || dimension < 21 || dimension > 177 || (dimension - 21) % 4 || module_size < 2 ||
        img.rows != (dimension + 4) * module_size || img.cols != img.rows)
        return std::nullopt;

    return decode_matrix_grid(img, dimension, dimension, ZXing::BarcodeFormat::QRCode, module_size);
}

static std::optional<std::string> decode_sampled_grid(const cv::Mat &img, int columns, int rows,
                                                      ZXing::BarcodeFormat format, int module_size,
                                                      GridDecodeCache &sampled)
{
    if (img.type() != CV_8UC1 || module_size < 2 || columns < 8 || rows < 8 || columns > 177 || rows > 177 ||
        img.cols != (columns + 4) * module_size || img.rows != (rows + 4) * module_size)
        return std::nullopt;
    if (format == ZXing::BarcodeFormat::DataMatrix) {
        if (!ZXing::DataMatrix::VersionForDimensions(rows, columns))
            return std::nullopt;
    } else if (format != ZXing::BarcodeFormat::QRCode || columns != rows || columns < 21 || (columns - 21) % 4)
        return std::nullopt;
    // Geometry has already been recovered by TPS. Sample the known module
    // centers directly so residual finder distortion cannot prevent decoding.
    cv::Mat smooth;
    cv::blur(img, smooth, cv::Size(3, 3));
    for (int smoothing = format == ZXing::BarcodeFormat::DataMatrix ? 0 : 1; smoothing < 2; ++smoothing) {
        const auto &sampleImage = smoothing ? smooth : img;
        for (int method = 0; method < 2; ++method) {
            cv::Mat binary;
            if (method == 0)
                cv::adaptiveThreshold(sampleImage, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                                      module_size * 5 + 1, 5);
            else
                cv::threshold(sampleImage, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            int offset = format == ZXing::BarcodeFormat::DataMatrix ? std::max(1, cvRound(module_size / 6.)) : 1;
            for (int dy : {0, -offset, offset}) {
                for (int dx : {0, -offset, offset}) {
                    ZXing::BitMatrix bits(columns, rows);
                    for (int y = 0; y < rows; ++y) {
                        const auto *row = binary.ptr<uchar>((y + 2) * module_size + module_size / 2 + dy);
                        for (int x = 0; x < columns; ++x)
                            bits.set(x, y, row[(x + 2) * module_size + module_size / 2 + dx] == 0);
                    }
                    std::string key;
                    if (format == ZXing::BarcodeFormat::DataMatrix) {
                        int errors = 0;
                        for (int x = 0; x < columns; ++x) {
                            errors += !bits.get(x, rows - 1);
                            errors += bits.get(x, 0) != (x % 2 == 0);
                        }
                        for (int y = 0; y < rows; ++y) {
                            errors += !bits.get(0, y);
                            errors += bits.get(columns - 1, y) != (y % 2 == 1);
                        }
                        if (errors > .3 * (2 * rows + 2 * columns))
                            continue;
                        key.push_back(char(columns));
                        key.push_back(char(rows));
                        key.append(reinterpret_cast<const char *>(bits.row(0).begin()), columns * rows);
                        if (sampled.failed.count(key))
                            continue;
                    }
                    auto decoded = format == ZXing::BarcodeFormat::QRCode ? ZXing::QRCode::Decode(bits)
                                                                          : ZXing::DataMatrix::Decode(bits);
                    if (decoded.isValid())
                        return decoded.content().text(ZXing::TextMode::HRI);
                    if (!key.empty() && sampled.failed.size() < 4096)
                        sampled.failed.insert(std::move(key));
                }
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> decode_matrix_grid(const cv::Mat &img, int columns, int rows,
                                              ZXing::BarcodeFormat format, int module_size, GridDecodeCache *cache)
{
    GridDecodeCache local;
    return decode_sampled_grid(img, columns, rows, format, module_size, cache ? *cache : local);
}

static std::optional<std::string> try_decode(const cv::Mat &img, ZXing::BarcodeFormat format, cv::Size grid,
                                             int module_size, GridDecodeCache &sampled)
{
    // The caller has already run the native finder on this rectified image.
    // Filtered variants of a known DM lattice only need module sampling/ECC.
    if (!grid.empty() && format == ZXing::BarcodeFormat::DataMatrix)
        return decode_sampled_grid(img, grid.width, grid.height, format, module_size, sampled);
    auto result = decode_zxingcpp(img, format);
    if (result && !result->empty())
        return result;
    if (!grid.empty()) {
        return decode_sampled_grid(img, grid.width, grid.height, format, module_size, sampled);
    }
    return std::nullopt;
}

static std::optional<std::string> try_gray_variants(const cv::Mat &gray, ZXing::BarcodeFormat format,
                                                    cv::Size grid, int module_size, GridDecodeCache *cache)
{
    GridDecodeCache local;
    auto &sampled = cache ? *cache : local;
    // A known DM lattice is already sampled with adaptive/Otsu thresholds
    // and a 3x3 box filter. Retain the two complementary wide Gaussian reads;
    // the unconstrained image/QR path keeps its established seven variants.
    bool knownDM = !grid.empty() && format == ZXing::BarcodeFormat::DataMatrix;
    cv::Mat sharpened;
    static const cv::Mat sharpenKernel =
        (cv::Mat_<float>(3, 3) << -1.f, -1.f, -1.f, -1.f, 9.f, -1.f, -1.f, -1.f, -1.f);
    for (int variant = 0; variant < 7; ++variant) {
        if (knownDM && variant != 2 && variant != 6)
            continue;
        if (variant >= 3 && sharpened.empty())
            cv::filter2D(gray, sharpened, -1, sharpenKernel);
        cv::Mat prepared;
        switch (variant) {
        case 0:
            cv::threshold(gray, prepared, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            break;
        case 1:
            cv::GaussianBlur(gray, prepared, {5, 5}, 0);
            break;
        case 2:
            cv::GaussianBlur(gray, prepared, {7, 7}, 0);
            break;
        case 3:
            prepared = sharpened;
            break;
        case 4:
            cv::threshold(sharpened, prepared, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            break;
        case 5:
            cv::GaussianBlur(sharpened, prepared, {5, 5}, 0);
            break;
        case 6:
            cv::GaussianBlur(sharpened, prepared, {7, 7}, 0);
            break;
        }
        if (auto result = try_decode(prepared, format, grid, module_size, sampled))
            return result;
    }

    return std::nullopt;
}

std::optional<std::string> decode_qreader(const cv::Mat &img, ZXing::BarcodeFormat format, cv::Size grid,
                                          int module_size, GridDecodeCache *cache)
{
    if (img.empty() || img.depth() != CV_8U)
        return std::nullopt;

    return try_gray_variants(img, format, grid, module_size, cache);
}

std::optional<std::string> decode(const cv::Mat &img)
{
    auto result = decode_zxingcpp(img);
    if (result)
        return result;
    return std::nullopt;
}

} // namespace qrscanner
