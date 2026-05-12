#include "binarizer.hpp"

#include "BitMatrix.h"
#include "HybridBinarizer.h"
#include "ImageView.h"

#include <opencv2/imgproc.hpp>

#include <memory>

namespace qrscanner {

static cv::Mat to_gray_continuous(const cv::Mat& img)
{
    cv::Mat gray;
    if (img.channels() == 1)
        gray = img.clone();
    else if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4)
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    else
        gray = img.clone();

    if (!gray.isContinuous())
        gray = gray.clone();
    return gray;
}

cv::Mat binarize(const cv::Mat& orig)
{
    cv::Mat gray = to_gray_continuous(orig);
    cv::Mat binary;
    cv::threshold(gray, binary, 115, 255, cv::THRESH_BINARY);
    cv::Mat bgr;
    cv::cvtColor(binary, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

cv::Mat binarize_mean(const cv::Mat& orig, int offset, int window_size)
{
    cv::Mat gray = to_gray_continuous(orig);
    if (window_size <= 0)
        window_size = std::max(1, std::min(orig.cols, orig.rows) / 5);

    cv::Mat blur;
    cv::blur(gray, blur, cv::Size(window_size, window_size));

    cv::Mat binary(gray.size(), CV_8UC1);
    for (int y = 0; y < gray.rows; ++y) {
        const uchar* g = gray.ptr<uchar>(y);
        const uchar* b = blur.ptr<uchar>(y);
        uchar* out = binary.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x)
            out[x] = g[x] > b[x] - offset ? 255 : 0;
    }

    cv::Mat bgr;
    cv::cvtColor(binary, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

cv::Mat binarize_smart(const cv::Mat& orig, int, int window_size)
{
    cv::Mat gray = to_gray_continuous(orig);
    if (window_size <= 0)
        window_size = std::max(1, std::min(orig.cols, orig.rows) / 5);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(window_size, window_size));
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);

    cv::Mat bgr;
    cv::cvtColor(enhanced, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

cv::Mat binarize_sauvola(const cv::Mat& image, int window, double k, double R)
{
    cv::Mat gray = to_gray_continuous(image);
    if (window <= 0)
        window = std::max(1, std::min(gray.cols, gray.rows) / 5);

    cv::Mat gray32;
    gray.convertTo(gray32, CV_32F);
    cv::Mat mean, sqmean;
    cv::boxFilter(gray32, mean, CV_32F, cv::Size(window, window));
    cv::boxFilter(gray32.mul(gray32), sqmean, CV_32F, cv::Size(window, window));

    cv::Mat binary(gray.size(), CV_8UC1);
    for (int y = 0; y < gray.rows; ++y) {
        const uchar* g = gray.ptr<uchar>(y);
        const float* m = mean.ptr<float>(y);
        const float* sm = sqmean.ptr<float>(y);
        uchar* out = binary.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) {
            const double var = std::max(0.0, static_cast<double>(sm[x] - m[x] * m[x]));
            const double stddev = std::sqrt(var);
            const double thresh = m[x] * (1.0 + k * ((stddev / R) - 1.0));
            out[x] = g[x] > thresh ? 255 : 0;
        }
    }
    return binary;
}

cv::Mat binarize_zxing_cpp(const cv::Mat& img)
{
    cv::Mat gray = to_gray_continuous(img);
    ZXing::ImageView view(gray.data,
                          static_cast<int>(gray.total()),
                          gray.cols,
                          gray.rows,
                          ZXing::ImageFormat::Lum,
                          gray.cols,
                          1);
    ZXing::HybridBinarizer bin(view);
    auto matrix = bin.getBlackMatrix();

    cv::Mat binary(gray.rows, gray.cols, CV_8UC1);
    for (int y = 0; y < gray.rows; ++y) {
        uchar* row = binary.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x)
            row[x] = matrix->get(x, y) ? 0 : 255;
    }

    cv::Mat bgr;
    cv::cvtColor(binary, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

} // namespace qrscanner
