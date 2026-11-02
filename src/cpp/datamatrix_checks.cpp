#include "DMscanner.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

static void require(bool value, const char *message)
{
    if (!value)
        throw std::runtime_error(message);
}

// Fixed ECC200 symbols generated once with ZXing-C++ 3.1.1. Keeping their
// modules here makes these checks independent of an encoder and image files.
static cv::Mat render(const std::vector<std::string> &rows, int module = 12, int quiet = 52)
{
    cv::Mat result(int(rows.size()) * module + 2 * quiet, int(rows.front().size()) * module + 2 * quiet, CV_8UC1,
                   cv::Scalar(255));
    for (int y = 0; y < int(rows.size()); ++y)
        for (int x = 0; x < int(rows[y].size()); ++x)
            if (rows[y][x] == '1')
                result(cv::Rect(quiet + x * module, quiet + y * module, module, module)).setTo(0);
    return result;
}

static void expect(const cv::Mat &image, const std::string &text, bool thorough = false)
{
    datamatrix::ScanOptions options;
    options.thorough = thorough;
    const auto result = datamatrix::detectAndDecode(image, options);
    require(result.status == qrscanner::ScanStatus::Decoded && result.text == text,
            "Data Matrix payload mismatch");
}

int main()
{
    try {
        cv::setNumThreads(1);
        const auto square =
            render({"10101010101010", "10100011011001", "10001100011100", "11100001010001", "11110110101110",
                    "11110000000011", "10001101111000", "11111001101011", "10100010000000", "11001000001001",
                    "10001100000110", "10110001100101", "11001101100000", "11111111111111"});
        const auto rectangle = render({"10101010101010101010101010101010", "11101101101011011110111101011001",
                                       "11000100001000001010000111001110", "11101100011000111111110101111001",
                                       "10000110101010001101110001001010", "11100101010100111111110111110111",
                                       "10010100001101101000001100010000", "11111111111111111111111111111111"});
        for (const auto &entry :
             {std::make_pair(square, "HELLO123"), std::make_pair(rectangle, "DM RECT CHECK")}) {
            cv::Mat rotated = entry.first.clone();
            for (int r = 0; r < 4; ++r) {
                expect(rotated, entry.second);
                cv::Mat inverted;
                cv::bitwise_not(rotated, inverted);
                expect(inverted, entry.second);
                cv::rotate(rotated, rotated, cv::ROTATE_90_CLOCKWISE);
            }
        }
        cv::Mat bgr, bgra;
        cv::cvtColor(square, bgr, cv::COLOR_GRAY2BGR);
        cv::cvtColor(square, bgra, cv::COLOR_GRAY2BGRA);
        expect(bgr, "HELLO123");
        expect(bgra, "HELLO123");
        cv::Mat padded(square.rows + 4, square.cols + 20, CV_8UC1, cv::Scalar(255));
        auto roi = padded(cv::Rect(3, 2, square.cols, square.rows));
        square.copyTo(roi);
        require(!roi.isContinuous(), "ROI fixture must have row padding");
        expect(roi, "HELLO123");

        const auto curvedSource =
            render({"1010101010101010101010", "1110011111111110011101", "1101011000101111100110",
                    "1111110100110110110001", "1000011010010000001010", "1010100111110101001011",
                    "1010000111011011000110", "1010111000010100000011", "1011100100001010111000",
                    "1111001100011010101101", "1010101011000011001000", "1000100110011000110001",
                    "1110001111001001100110", "1100100110001010001111", "1110101111110010001010",
                    "1110100110100000001001", "1001100000001101110100", "1010110001111001001111",
                    "1010011010010110010110", "1110110110100010001111", "1000001011011011011110",
                    "1111111111111111111111"});
        int recovered = 0;
        for (double amplitude : {12., 20., 30., 40.}) {
            cv::Mat mx(curvedSource.size(), CV_32F), my(curvedSource.size(), CV_32F);
            for (int y = 0; y < mx.rows; ++y)
                for (int x = 0; x < mx.cols; ++x) {
                    mx.at<float>(y, x) = float(x);
                    my.at<float>(y, x) = float(y - amplitude * std::sin((x - 52.) / 264 * CV_PI));
                }
            cv::Mat curved;
            cv::remap(curvedSource, curved, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 255);
            expect(curved, "Curved Data Matrix regression check 2026");
            cv::Mat inverted;
            cv::bitwise_not(curved, inverted);
            expect(inverted, "Curved Data Matrix regression check 2026");
            expect(curved, "Curved Data Matrix regression check 2026", true);
            recovered += datamatrix::decodeBaseline(curved).status != qrscanner::ScanStatus::Decoded;
        }
        require(recovered > 0, "Curved checks must exercise recovery beyond the baseline");
        for (const auto &invalid :
             {cv::Mat(), cv::Mat(20, 20, CV_32F, cv::Scalar(0)), cv::Mat(20, 20, CV_8UC2, cv::Scalar(0)),
              cv::Mat(200, 200, CV_8UC1, cv::Scalar(255)), cv::Mat(200, 200, CV_8UC1, cv::Scalar(0)),
              cv::Mat(1, 20000, CV_8UC1, cv::Scalar(255))})
            require(datamatrix::detectAndDecode(invalid).status != qrscanner::ScanStatus::Decoded,
                    "Invalid input decoded");
        cv::Mat noise(300, 300, CV_8UC1);
        cv::RNG random(1);
        random.fill(noise, cv::RNG::UNIFORM, 0, 256);
        require(datamatrix::detectAndDecode(noise).status != qrscanner::ScanStatus::Decoded, "Noise decoded");
        std::cout << "Data Matrix checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
