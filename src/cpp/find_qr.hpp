#pragma once

#include "homographic_correct.hpp"
#include "tools.hpp"

#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace qrscanner {

struct ZxingPattern {
    Point center;
    int size = 0;
    Points corners;
};

struct ZxingFindResult {
    std::vector<ZxingPattern> patterns;
    cv::Mat binary;
};

struct DistributionResult {
    int size = 0;
    std::array<std::optional<Points>, 2> pattern;
};

struct SearchDistributionResult {
    std::array<std::optional<Points>, 4> pattern;
};

std::vector<cv::Point> approxQuadr(const std::vector<cv::Point>& contour);
std::pair<std::vector<cv::Point>, double> approxedQuadrR(const std::vector<cv::Point>& contour);
std::vector<int> get_colors(const std::vector<std::vector<cv::Point>>& contours, const std::vector<cv::Vec4i>& hierarchy);
std::vector<int> find_search(const std::vector<std::vector<cv::Point>>& contours,
                             const std::vector<cv::Vec4i>& hierarchy,
                             const std::vector<int>& colors,
                             int lvl = 0);
std::vector<int> find_corrective(const std::vector<std::vector<cv::Point>>& contours,
                                 const std::vector<cv::Vec4i>& hierarchy,
                                 const std::vector<int>& colors,
                                 double search_area);
Points clockwise(const Points& contour);
std::vector<Points> sort_search(const std::vector<Points>& search);
std::vector<Points> sort_search_dist(const std::vector<Points>& search);
std::vector<Points> sort_search_dir(const std::vector<Points>& search);
Points sort_corrective(const std::vector<Points>& correctives, const std::vector<Points>& search);
Points find(const cv::Mat& img);

DistributionResult recognize_distribution(const cv::Mat& img, const Points& search, const HomographyMatrix* H = nullptr);
SearchDistributionResult recognize_search_distribution(const cv::Mat& img, const Points& search, const HomographyMatrix* H = nullptr);

ZxingFindResult zxing_find_with_binary(const cv::Mat& img);
std::vector<ZxingPattern> zxing_find(const cv::Mat& img);
std::optional<std::string> zxing_check(const cv::Mat& img);
bool zxing_detect(const cv::Mat& img);
std::pair<Points, cv::Mat> find_qr_zxing(const cv::Mat& orig, const ZxingFindResult* zxing_qr = nullptr);

} // namespace qrscanner
