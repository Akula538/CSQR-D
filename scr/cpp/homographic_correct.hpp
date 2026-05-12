#pragma once

#include "tools.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <utility>

namespace qrscanner {

class HomographyMatrix {
public:
    HomographyMatrix() = default;
    HomographyMatrix(const Points& fr, const Points& to);

    Point predict(const Point& point) const;
    const cv::Matx33d& matrix() const { return H_; }

private:
    cv::Matx33d H_ = cv::Matx33d::eye();
};

cv::Matx33d find_H(const Points& current, const Points& expected);
Point F(const Point& point, const cv::Matx33d& H);
std::pair<Points, HomographyMatrix> find_homo(const Points& points, int size = 2000);

} // namespace qrscanner
