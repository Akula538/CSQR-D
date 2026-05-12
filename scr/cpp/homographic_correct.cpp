#include "homographic_correct.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <stdexcept>

namespace qrscanner {

static cv::Matx33d solve_homography_4pt(const Points& fr, const Points& to)
{
    if (fr.size() != to.size() || fr.size() < 4)
        throw std::runtime_error("Homography expects at least four paired points");

    cv::Mat A = cv::Mat::zeros(static_cast<int>(fr.size() * 2), 8, CV_64F);
    cv::Mat B = cv::Mat::zeros(static_cast<int>(fr.size() * 2), 1, CV_64F);

    for (size_t i = 0; i < fr.size(); ++i) {
        const double x = fr[i].x;
        const double y = fr[i].y;
        const double x1 = to[i].x;
        const double y1 = to[i].y;

        A.at<double>(static_cast<int>(2 * i), 0) = x;
        A.at<double>(static_cast<int>(2 * i), 1) = y;
        A.at<double>(static_cast<int>(2 * i), 2) = 1.0;
        A.at<double>(static_cast<int>(2 * i), 6) = -x * x1;
        A.at<double>(static_cast<int>(2 * i), 7) = -y * x1;

        A.at<double>(static_cast<int>(2 * i + 1), 3) = x;
        A.at<double>(static_cast<int>(2 * i + 1), 4) = y;
        A.at<double>(static_cast<int>(2 * i + 1), 5) = 1.0;
        A.at<double>(static_cast<int>(2 * i + 1), 6) = -x * y1;
        A.at<double>(static_cast<int>(2 * i + 1), 7) = -y * y1;

        B.at<double>(static_cast<int>(2 * i), 0) = x1;
        B.at<double>(static_cast<int>(2 * i + 1), 0) = y1;
    }

    cv::Mat h;
    if (!cv::solve(A, B, h, cv::DECOMP_LU) && !cv::solve(A, B, h, cv::DECOMP_SVD))
        return cv::Matx33d::eye();

    return cv::Matx33d(h.at<double>(0), h.at<double>(1), h.at<double>(2),
                       h.at<double>(3), h.at<double>(4), h.at<double>(5),
                       h.at<double>(6), h.at<double>(7), 1.0);
}

HomographyMatrix::HomographyMatrix(const Points& fr, const Points& to)
    : H_(solve_homography_4pt(fr, to))
{
}

Point HomographyMatrix::predict(const Point& point) const
{
    cv::Vec3d a(point.x, point.y, 1.0);
    cv::Vec3d b = H_ * a;
    if (b[2] == 0.0)
        return {};
    b *= 1.0 / b[2];
    return {b[0], b[1]};
}

cv::Matx33d find_H(const Points& current, const Points& expected)
{
    return solve_homography_4pt(current, expected);
}

Point F(const Point& point, const cv::Matx33d& H)
{
    cv::Vec3d a(point.x, point.y, 1.0);
    cv::Vec3d b = H * a;
    b *= 1.0 / b[2];
    return {b[0], b[1]};
}

std::pair<Points, HomographyMatrix> find_homo(const Points& points, int size)
{
    Points fr = {
        (points[0] + points[1] * 13.0 + points[2] * 169.0 + points[3] * 13.0) * (1.0 / 196.0),
        (points[5] + points[6] * 13.0 + points[7] * 169.0 + points[4] * 13.0) * (1.0 / 196.0),
        center({points[8], points[9], points[10], points[11]}),
        (points[15] + points[12] * 13.0 + points[13] * 169.0 + points[14] * 13.0) * (1.0 / 196.0),
    };

    Points to = {
        {size / 3.0, size / 3.0},
        {size * 2.0 / 3.0, size / 3.0},
        {size * 2.0 / 3.0, size * 2.0 / 3.0},
        {size / 3.0, size * 2.0 / 3.0},
    };

    HomographyMatrix H(fr, to);
    HomographyMatrix H_inv(to, fr);

    Points corrected;
    corrected.reserve(points.size());
    for (const auto& p : points)
        corrected.push_back(H.predict(p));

    return {corrected, H_inv};
}

} // namespace qrscanner
