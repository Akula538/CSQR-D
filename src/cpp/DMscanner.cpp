#include "DMscanner.hpp"
#include "Barcode.h"
#include "BarcodeFormat.h"
#include "BitMatrix.h"
#include "DecoderResult.h"
#include "ImageView.h"
#include "ReadBarcode.h"
#include "ReaderOptions.h"
#include "datamatrix/DMDecoder.h"
#include "datamatrix/DMVersion.h"
#include "vectorfield_correct.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace datamatrix
{
using qrscanner::ScanResult;
using qrscanner::ScanStatus;
using Quad = std::array<cv::Point2f, 4>;
struct LocatedCandidate {
    Quad corners;
    std::vector<cv::Point> outline;
};
struct SearchLimits {
    int patches;
    int grids;
    int flow;
    bool extraField;
};
// Fixed work budgets keep results deterministic; the extended search is opt-in.
constexpr SearchLimits fastLimits{3, 4, 0, false};
constexpr SearchLimits thoroughLimits{24, 24, 8, true};

ScanResult decodeBaseline(const cv::Mat &image)
{
    if (image.empty() || image.depth() != CV_8U ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
        return {};
    const auto format = image.channels() == 1   ? ZXing::ImageFormat::Lum
                        : image.channels() == 3 ? ZXing::ImageFormat::BGR
                                                : ZXing::ImageFormat::BGRA;
    ZXing::ImageView view(image.data, image.cols, image.rows, format, int(image.step));
    ZXing::ReaderOptions options;
    options.setFormats(ZXing::BarcodeFormat::DataMatrix)
        .setTryHarder(true)
        .setTryRotate(true)
        .setTryInvert(true)
        .setTryDownscale(true)
        .setTextMode(ZXing::TextMode::HRI);
    const auto result = ZXing::ReadBarcode(view, options);
    return result.isValid() ? ScanResult{ScanStatus::Decoded, result.text()} : ScanResult{};
}

static std::vector<LocatedCandidate> candidates(const cv::Mat &gray)
{
    cv::Mat black;
    cv::adaptiveThreshold(gray, black, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 41, 7);
    std::vector<LocatedCandidate> result;
    for (int k : {1, 3, 5, 7, 9, 13}) {
        cv::Mat mask;
        cv::morphologyEx(black, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {k, k}));
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (const auto &contour : contours) {
            const double area = cv::contourArea(contour);
            if (area < 300 || area > gray.total() * .95)
                continue;
            auto rect = cv::minAreaRect(contour);
            double w = rect.size.width, h = rect.size.height;
            if (std::min(w, h) < 18 || std::max(w, h) > std::min(w, h) * 4 || area / (w * h) < .35)
                continue;
            std::vector<cv::Point> hull, poly;
            cv::convexHull(contour, hull);
            cv::approxPolyDP(hull, poly, cv::arcLength(hull, true) * .04, true);
            Quad q;
            if (poly.size() == 4)
                for (int i = 0; i < 4; ++i)
                    q[i] = poly[i];
            else
                rect.points(q.data());
            bool duplicate = false;
            for (const auto &old : result) {
                double distance = 0;
                for (int i = 0; i < 4; ++i)
                    distance += cv::norm(q[i] - old.corners[i]);
                if (distance < 4) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                result.push_back({q, contour});
                // Keep the enclosing rectangle as a distinct hypothesis: even
                // small corner changes can matter on a low-resolution symbol.
                if (poly.size() == 4) {
                    Quad box;
                    rect.points(box.data());
                    result.push_back({box, contour});
                }
            }
        }
    }
    return result;
}

static ScanResult decodeGrid(const cv::Mat &binary, int w, int h, double dx = 0, double dy = 0)
{
    ZXing::BitMatrix bits(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int px = cvRound(16 + (x + .5 + dx) * 224 / w), py = cvRound(16 + (y + .5 + dy) * 224 / h);
            bits.set(x, y, binary.at<uchar>(std::clamp(py, 0, 255), std::clamp(px, 0, 255)) == 0);
        }
    for (int r = 0; r < 4; ++r) {
        int bw = bits.width(), bh = bits.height();
        if (ZXing::DataMatrix::VersionForDimensions(bh, bw)) {
            int errors = 0;
            for (int x = 0; x < bw; ++x) {
                errors += !bits.get(x, bh - 1);
                errors += bits.get(x, 0) != (x % 2 == 0);
            }
            for (int y = 0; y < bh; ++y) {
                errors += !bits.get(0, y);
                errors += bits.get(bw - 1, y) != (y % 2 == 1);
            }
            if (errors < .3 * (2 * bw + 2 * bh)) {
                auto decoded = ZXing::DataMatrix::Decode(bits);
                if (decoded.isValid())
                    return {ScanStatus::Decoded, decoded.content().text(ZXing::TextMode::HRI)};
            }
        }
        bits.rotate90();
    }
    return {};
}

// Trace the outer envelope. Timing edges contain white gaps, so fit only the
// outer part of the observations, then reject inward module/data transitions.
static bool edgeCurve(const cv::Mat &binary, std::array<double, 256> &curve, int degree = 2)
{
    std::vector<double> ys;
    for (int x = 30; x < 226; ++x) {
        int y = 1;
        while (y < 65 && binary.at<uchar>(y, x) != 0)
            ++y;
        ys.push_back(y);
    }
    auto sorted = ys;
    std::sort(sorted.begin(), sorted.end());
    double cutoff = sorted[sorted.size() * 65 / 100];
    std::vector<bool> use(ys.size());
    for (size_t i = 0; i < ys.size(); ++i)
        use[i] = ys[i] <= cutoff && ys[i] < 64;
    cv::Mat coeff;
    for (int iteration = 0; iteration < 6; ++iteration) {
        int count = int(std::count(use.begin(), use.end(), true));
        if (count < 30)
            return false;
        cv::Mat A(count, degree + 1, CV_64F), B(count, 1, CV_64F);
        int j = 0;
        for (size_t i = 0; i < ys.size(); ++i)
            if (use[i]) {
                double x = (double(i) + 30 - 128) / 128;
                double power = 1;
                for (int d = 0; d <= degree; ++d) {
                    A.at<double>(j, d) = power;
                    power *= x;
                }
                B.at<double>(j++) = ys[i];
            }
        if (!cv::solve(A, B, coeff, cv::DECOMP_SVD))
            return false;
        for (int x = 0; x < 256; ++x) {
            double t = (x - 128.) / 128;
            curve[x] = 0;
            double power = 1;
            for (int d = 0; d <= degree; ++d) {
                curve[x] += power * coeff.at<double>(d);
                power *= t;
            }
        }
        for (size_t i = 0; i < ys.size(); ++i) {
            double residual = ys[i] - curve[i + 30];
            use[i] = residual < 3 && residual > -6 && ys[i] < 64;
        }
    }
    return curve[30] > -20 && curve[225] > -20 && curve[30] < 65 && curve[225] < 65;
}

static cv::Mat straighten(const cv::Mat &patch, const cv::Mat &silhouette = cv::Mat(), int degree = 2)
{
    cv::Mat binary;
    if (silhouette.empty())
        cv::threshold(patch, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    else
        binary = silhouette.clone();
    std::array<std::array<double, 256>, 4> curves;
    for (int e = 0; e < 4; ++e) {
        if (!edgeCurve(binary, curves[e], degree))
            return {};
        cv::rotate(binary, binary, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    // Intermediate iterations use only rounded coordinates. Tabulate those
    // transitions exactly; retain the original double arithmetic for the final
    // subpixel mapping. This avoids repeated rounding and polynomial blending.
    cv::Mat xIndex(256, 256, CV_8U), yIndex(256, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        double t = (i - 16.) / 224;
        for (int j = 0; j < 256; ++j) {
            xIndex.at<uchar>(i, j) =
                uchar(std::clamp(cvRound((1 - t) * curves[3][255 - j] + t * (255 - curves[1][j])), 0, 255));
            yIndex.at<uchar>(i, j) =
                uchar(std::clamp(cvRound((1 - t) * curves[0][j] + t * (255 - curves[2][255 - j])), 0, 255));
        }
    }
    cv::Mat mx(256, 256, CV_32F), my(256, 256, CV_32F);
    for (int y = 0; y < 256; ++y)
        for (int x = 0; x < 256; ++x) {
            double u = (x - 16.) / 224, v = (y - 16.) / 224;
            int xx = x, yy = y;
            const auto *xRow = xIndex.ptr<uchar>(x);
            const auto *yRow = yIndex.ptr<uchar>(y);
            for (int it = 0; it < 7; ++it) {
                int nextX = xRow[yy], nextY = yRow[xx];
                xx = nextX;
                yy = nextY;
            }
            mx.at<float>(y, x) = float((1 - u) * curves[3][255 - yy] + u * (255 - curves[1][yy]));
            my.at<float>(y, x) = float((1 - v) * curves[0][xx] + v * (255 - curves[2][255 - xx]));
        }
    cv::Mat result;
    cv::remap(patch, result, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 255);
    return result;
}

static std::vector<double> timingCenters(const cv::Mat &binary, int depth, bool vertical)
{
    std::vector<int> edges{16};
    auto black = [&](int i) {
        return vertical ? binary.at<uchar>(i, 239 - depth) == 0 : binary.at<uchar>(16 + depth, i) == 0;
    };
    bool last = black(16);
    for (int i = 17; i < 240; ++i) {
        bool next = black(i);
        if (next != last) {
            edges.push_back(i);
            last = next;
        }
    }
    edges.push_back(240);
    std::vector<double> centers;
    for (size_t i = 1; i < edges.size(); ++i) {
        if (edges[i] - edges[i - 1] < 2)
            return {};
        centers.push_back((edges[i] + edges[i - 1] - 1) * .5);
    }
    return centers;
}

struct GridGuess {
    std::vector<double> xs, ys;
    double score;
};

static std::vector<GridGuess> grids(const cv::Mat &binary)
{
    std::vector<GridGuess> result;
    for (int depthX : {3, 2, 4, 5, 6, 1, 7, 8}) {
        auto xs = timingCenters(binary, depthX, false);
        if (xs.size() < 8 || xs.size() > 144 || xs.size() % 2)
            continue;
        for (int depthY : {3, 2, 4, 5, 6, 1, 7, 8}) {
            auto ys = timingCenters(binary, depthY, true);
            int w = int(xs.size()), h = int(ys.size());
            if (!ZXing::DataMatrix::VersionForDimensions(h, w))
                continue;
            int errors = 0;
            for (auto x : xs)
                errors += binary.at<uchar>(235, cvRound(x)) != 0;
            for (auto y : ys)
                errors += binary.at<uchar>(cvRound(y), 20) != 0;
            if (errors > .3 * (w + h))
                continue;
            bool duplicate = false;
            for (auto &old : result)
                if (old.xs.size() == xs.size() && old.ys.size() == ys.size()) {
                    double d = 0;
                    for (int x = 0; x < w; ++x)
                        d += std::abs(old.xs[x] - xs[x]);
                    for (int y = 0; y < h; ++y)
                        d += std::abs(old.ys[y] - ys[y]);
                    if (d / (w + h) < .5) {
                        duplicate = true;
                        break;
                    }
                }
            if (!duplicate)
                result.push_back({xs, ys, double(errors) / (w + h) + .01 * (depthX + depthY)});
        }
    }
    std::stable_sort(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.score < b.score; });
    if (result.size() > 8)
        result.resize(8);
    return result;
}

struct ClockFit {
    std::vector<double> centers;
    double cost = 1;
};

// Missing/merged timing cells must not change the symbol dimension. Segment
// the entire clock track into alternating cells, balancing observed polarity
// against smooth module pitch. Dimensions are still restricted by ECC200.
static ClockFit fitClock(const cv::Mat &binary, int modules, bool vertical)
{
    ClockFit best;
    constexpr int length = 224;
    const double pitch = double(length) / modules;
    const int minWidth = std::max(2, int(pitch * .45)), maxWidth = int(pitch * 1.8) + 1;
    for (int depth : {2, 4, 6}) {
        std::array<double, 225> prefix{};
        for (int i = 0; i < length; ++i) {
            double white = vertical ? binary.at<uchar>(16 + i, 239 - depth) / 255.
                                    : binary.at<uchar>(16 + depth, 16 + i) / 255.;
            prefix[i + 1] = prefix[i] + white;
        }
        std::vector<std::array<double, 225>> cost(modules + 1);
        std::vector<std::array<int, 225>> previous(modules + 1);
        for (auto &row : cost)
            row.fill(1e20);
        cost[0][0] = 0;
        for (int cell = 1; cell <= modules; ++cell) {
            bool white = (cell % 2 == 0) != vertical;
            int lo = std::max(cell * minWidth, length - (modules - cell) * maxWidth);
            int hi = std::min(cell * maxWidth, length - (modules - cell) * minWidth);
            for (int end = lo; end <= hi; ++end) {
                for (int width = minWidth; width <= maxWidth && width <= end; ++width) {
                    double mismatch = prefix[end] - prefix[end - width];
                    if (white)
                        mismatch = width - mismatch;
                    const double penalty = .12 * (width - pitch) * (width - pitch) / pitch;
                    double candidate = cost[cell - 1][end - width] + mismatch + penalty;
                    if (candidate < cost[cell][end]) {
                        cost[cell][end] = candidate;
                        previous[cell][end] = end - width;
                    }
                }
            }
        }
        double value = cost[modules][length] / length;
        if (value >= best.cost)
            continue;
        best.cost = value;
        best.centers.resize(modules);
        int end = length;
        for (int cell = modules; cell > 0; --cell) {
            int begin = previous[cell][end];
            best.centers[cell - 1] = 16 + (begin + end - 1) * .5;
            end = begin;
        }
    }
    return best;
}

static std::vector<GridGuess> recoverGrids(const cv::Mat &binary)
{
    // Verify both continuous sides before attempting the more costly clock fit.
    int left = 0, bottom = 0;
    for (int i = 20; i < 236; ++i) {
        left += binary.at<uchar>(i, 20) == 0;
        bottom += binary.at<uchar>(235, i) == 0;
    }
    if (left < 150 || bottom < 150)
        return {};
    std::array<std::vector<int>, 2> dimensions;
    for (int axis = 0; axis < 2; ++axis) {
        for (int depth : {2, 4, 6}) {
            int transitions = 0;
            bool last = false;
            for (int i = 16; i < 240; ++i) {
                bool black = (axis ? binary.at<uchar>(i, 239 - depth) : binary.at<uchar>(16 + depth, i)) == 0;
                if (i != 16 && black != last)
                    ++transitions;
                last = black;
            }
            int estimate = (transitions + 2) / 2 * 2;
            if (estimate < 8)
                continue;
            for (int d = estimate - 4; d <= estimate + 4; d += 2)
                if (d >= 8 && d <= 64 &&
                    std::find(dimensions[axis].begin(), dimensions[axis].end(), d) == dimensions[axis].end())
                    dimensions[axis].push_back(d);
        }
    }
    std::map<int, ClockFit> horizontal, vertical;
    std::vector<GridGuess> guesses;
    for (int w : dimensions[0])
        for (int h : dimensions[1]) {
            if (!ZXing::DataMatrix::VersionForDimensions(h, w))
                continue;
            if (!horizontal.count(w))
                horizontal[w] = fitClock(binary, w, false);
            if (!vertical.count(h))
                vertical[h] = fitClock(binary, h, true);
            const auto &x = horizontal[w];
            const auto &y = vertical[h];
            if (x.cost > .23 || y.cost > .23)
                continue;
            guesses.push_back({x.centers, y.centers, x.cost + y.cost});
        }
    std::sort(guesses.begin(), guesses.end(), [](const auto &a, const auto &b) { return a.score < b.score; });
    if (guesses.size() > 3)
        guesses.resize(3);
    return guesses;
}

static ScanResult readPoints(const cv::Mat &binary, int w, int h, const std::vector<cv::Point2d> &points,
                             double pitchX = 0, double pitchY = 0)
{
    if (pitchX == 0)
        pitchX = 224. / w;
    if (pitchY == 0)
        pitchY = 224. / h;
    std::vector<ZXing::BitMatrix> tried;
    for (auto offset : {cv::Point2d(0, 0), cv::Point2d(-.15, 0), cv::Point2d(.15, 0), cv::Point2d(0, -.15),
                        cv::Point2d(0, .15), cv::Point2d(-.15, -.15), cv::Point2d(.15, .15)}) {
        ZXing::BitMatrix bits(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                auto p = points[y * w + x];
                bits.set(x, y,
                         binary.at<uchar>(std::clamp(cvRound(p.y + offset.y * pitchY), 0, binary.rows - 1),
                                          std::clamp(cvRound(p.x + offset.x * pitchX), 0, binary.cols - 1)) == 0);
            }
        if (std::find(tried.begin(), tried.end(), bits) != tried.end())
            continue;
        auto decoded = ZXing::DataMatrix::Decode(bits);
        if (decoded.isValid())
            return {ScanStatus::Decoded, decoded.content().text(ZXing::TextMode::HRI)};
        tried.push_back(std::move(bits));
    }
    return {};
}

// Reuse the QR scanner's local gradient directions. Trace column centers from
// the top clock track and row centers from the right clock track, then intersect
// them. Unlike QR, Data Matrix has no three finder centers to seed these lines.
static cv::Mat fieldSlopes(const cv::Mat &gray, int kernel)
{
    qrscanner::DirectionField field(gray);
    cv::Mat slopes(17, 17, CV_32FC2);
    for (int y = 0; y < 17; ++y)
        for (int x = 0; x < 17; ++x) {
            auto dirs = field.get(16 + 14 * x, 16 + 14 * y, kernel);
            cv::Vec2f v(0, 0);
            double bestH = 1e9, bestV = 1e9;
            for (auto direction : {dirs.first, dirs.second})
                if (direction) {
                    double angle = *direction * CV_PI / 180;
                    // DirectionField reports gradient normals, not edge tangents.
                    double dx = -std::sin(angle), dy = std::cos(angle);
                    if (std::abs(dx) > .8 && std::abs(dy / dx) < bestH) {
                        bestH = std::abs(dy / dx);
                        v[0] = float(dy / dx);
                    }
                    if (std::abs(dy) > .8 && std::abs(dx / dy) < bestV) {
                        bestV = std::abs(dx / dy);
                        v[1] = float(dx / dy);
                    }
                }
            slopes.at<cv::Vec2f>(y, x) = v;
        }
    return slopes;
}

static std::vector<cv::Point2d> fieldPoints(const cv::Mat &slopes, const GridGuess &grid)
{
    auto slope = [&](double x, double y, int component) {
        double u = std::clamp((x - 16) / 14, 0., 15.999), v = std::clamp((y - 16) / 14, 0., 15.999);
        int ix = int(u), iy = int(v);
        u -= ix;
        v -= iy;
        return (1 - v) * ((1 - u) * slopes.at<cv::Vec2f>(iy, ix)[component] +
                          u * slopes.at<cv::Vec2f>(iy, ix + 1)[component]) +
               v * ((1 - u) * slopes.at<cv::Vec2f>(iy + 1, ix)[component] +
                    u * slopes.at<cv::Vec2f>(iy + 1, ix + 1)[component]);
    };
    int w = int(grid.xs.size()), h = int(grid.ys.size());
    std::vector<std::array<double, 256>> cols(w), rows(h);
    for (int x = 0; x < w; ++x) {
        auto &col = cols[x];
        col.fill(grid.xs[x]);
        for (int y = 21; y < 255; ++y)
            col[y + 1] = col[y] + slope(col[y], y, 1);
        for (int y = 21; y > 0; --y)
            col[y - 1] = col[y] - slope(col[y], y, 1);
    }
    for (int y = 0; y < h; ++y) {
        auto &row = rows[y];
        row.fill(grid.ys[y]);
        for (int x = 235; x > 0; --x)
            row[x - 1] = row[x] - slope(x, row[x], 0);
        for (int x = 235; x < 255; ++x)
            row[x + 1] = row[x] + slope(x, row[x], 0);
    }
    auto at = [](const auto &line, double t) {
        t = std::clamp(t, 0., 254.999);
        int i = int(t);
        return line[i] * (1 - (t - i)) + line[i + 1] * (t - i);
    };
    std::vector<cv::Point2d> points;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            cv::Point2d p(grid.xs[x], grid.ys[y]);
            for (int it = 0; it < 8; ++it)
                p = {at(cols[x], p.y), at(rows[y], p.x)};
            points.push_back(p);
        }
    return points;
}

// Register a tentative module rendering against the observed image. The grid
// itself supplies a local geometric reference; no payload or dataset labels
// participate in this refinement. Only a valid ECC200 decode is accepted.
static ScanResult refineGrid(const cv::Mat &gray, const GridGuess &grid, const std::vector<cv::Point2d> &points)
{
    int w = int(grid.xs.size()), h = int(grid.ys.size());
    if (w > 64 || h > 64)
        return {};
    constexpr int module = 12;
    cv::Mat coarse(h, w, CV_32FC2);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            coarse.at<cv::Vec2f>(y, x) = {float(points[y * w + x].x), float(points[y * w + x].y)};
    cv::Mat mapping, rectified;
    cv::resize(coarse, mapping, {w * module, h * module}, 0, 0, cv::INTER_CUBIC);
    cv::remap(gray, rectified, mapping, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 255);
    std::vector<cv::Point2d> centers;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            centers.emplace_back((x + .5) * module, (y + .5) * module);
    for (int iteration = 0; iteration < 4; ++iteration) {
        cv::Mat binary;
        cv::threshold(rectified, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        auto result = readPoints(binary, w, h, centers, module, module);
        if (result.status == ScanStatus::Decoded)
            return result;
        if (iteration == 3)
            break;
        cv::Mat cells(h, w, CV_8UC1);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                cells.at<uchar>(y, x) = binary.at<uchar>(y * module + module / 2, x * module + module / 2);
        cells.col(0).setTo(0);
        cells.row(h - 1).setTo(0);
        for (int x = 0; x < w; ++x)
            cells.at<uchar>(0, x) = (x % 2) * 255;
        for (int y = 0; y < h; ++y)
            cells.at<uchar>(y, w - 1) = ((y + 1) % 2) * 255;
        cv::Mat reference, target, flow;
        cv::resize(cells, reference, rectified.size(), 0, 0, cv::INTER_NEAREST);
        cv::GaussianBlur(reference, reference, {5, 5}, 1);
        cv::normalize(rectified, target, 0, 255, cv::NORM_MINMAX);
        cv::calcOpticalFlowFarneback(reference, target, flow, .5, 3, 31, 5, 7, 1.5, 0);
        for (int y = 0; y < flow.rows; ++y)
            for (int x = 0; x < flow.cols; ++x) {
                auto &f = flow.at<cv::Vec2f>(y, x);
                f[0] = x + std::clamp(f[0], -9.f, 9.f);
                f[1] = y + std::clamp(f[1], -9.f, 9.f);
            }
        cv::remap(rectified, rectified, flow, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 255);
    }
    return {};
}

static ScanResult sample(const cv::Mat &patch)
{
    std::array<cv::Mat, 2> binaries;
    for (int rot = 0; rot < 4; ++rot) {
        for (int method = 0; method < 2; ++method) {
            auto &binary = binaries[method];
            if (binary.empty()) {
                if (method == 0)
                    cv::threshold(patch, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                else
                    cv::adaptiveThreshold(patch, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                                          41, 5);
            }
            for (const auto &grid : grids(binary)) {
                int w = int(grid.xs.size()), h = int(grid.ys.size());
                std::vector<cv::Point2d> points;
                for (auto y : grid.ys)
                    for (auto x : grid.xs)
                        points.emplace_back(x, y);
                auto result = readPoints(binary, w, h, points);
                if (result.status == ScanStatus::Decoded)
                    return result;
                result = decodeGrid(binary, w, h);
                if (result.status == ScanStatus::Decoded)
                    return result;
            }
        }
        for (auto &binary : binaries)
            cv::rotate(binary, binary, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    return {};
}

struct PatchWork {
    cv::Mat gray;
    std::map<int, cv::Mat> slopes;
};
struct RecoveryCandidate {
    std::shared_ptr<PatchWork> patch;
    cv::Mat binary;
    GridGuess grid;
    double quality;
};

static std::vector<RecoveryCandidate> prepareRecovery(const std::vector<cv::Mat> &patches, int maxCandidates)
{
    std::vector<RecoveryCandidate> result;
    for (const auto &input : patches) {
        cv::Mat gray = input.clone();
        std::array<cv::Mat, 2> binaries;
        cv::threshold(gray, binaries[0], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::adaptiveThreshold(gray, binaries[1], 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 41, 5);
        for (int rot = 0; rot < 4; ++rot) {
            auto work = std::make_shared<PatchWork>();
            work->gray = gray;
            for (int method = 0; method < 2; ++method) {
                auto binary = binaries[method];
                auto hard = grids(binary), soft = recoverGrids(binary);
                if (hard.size() > 3)
                    hard.resize(3);
                auto combined = soft;
                combined.insert(combined.end(), hard.begin(), hard.end());
                for (size_t i = 0; i < combined.size(); ++i) {
                    auto &grid = combined[i];
                    double quality = grid.score;
                    if (i >= soft.size()) {
                        quality = .35 + grid.score;
                        for (const auto &other : soft)
                            if (grid.xs.size() == other.xs.size() && grid.ys.size() == other.ys.size())
                                quality = std::min(quality, other.score + .01 + .05 * grid.score);
                    }
                    // All clock evidence is image-derived. Longer coherent
                    // tracks are stronger evidence than small accidental grids.
                    quality += .05 * 20 / (grid.xs.size() + grid.ys.size());
                    bool duplicate = false;
                    for (size_t j = 0; j < i; ++j) {
                        auto &other = combined[j];
                        if (other.xs.size() != grid.xs.size() || other.ys.size() != grid.ys.size())
                            continue;
                        double distance = 0;
                        for (size_t k = 0; k < grid.xs.size(); ++k)
                            distance += std::abs(grid.xs[k] - other.xs[k]);
                        for (size_t k = 0; k < grid.ys.size(); ++k)
                            distance += std::abs(grid.ys[k] - other.ys[k]);
                        if (distance < .3 * (grid.xs.size() + grid.ys.size())) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate)
                        result.push_back({work, binary, grid, quality});
                }
            }
            cv::Mat next;
            cv::rotate(gray, next, cv::ROTATE_90_COUNTERCLOCKWISE);
            gray = next;
            // Out-of-place rotation preserves masks retained by earlier guesses.
            for (auto &binary : binaries) {
                cv::Mat rotated;
                cv::rotate(binary, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
                binary = rotated;
            }
        }
    }
    std::stable_sort(result.begin(), result.end(),
                     [](const auto &a, const auto &b) { return a.quality < b.quality; });
    if (result.size() > size_t(maxCandidates))
        result.resize(maxCandidates);
    return result;
}

static ScanResult recover(std::vector<RecoveryCandidate> &candidates, bool flow, const SearchLimits &limits)
{
    int limit = flow ? limits.flow : limits.grids;
    for (auto &candidate : candidates) {
        if (--limit < 0)
            break;
        auto &grid = candidate.grid;
        auto &work = *candidate.patch;
        const int w = int(grid.xs.size()), h = int(grid.ys.size());
        std::vector<cv::Point2d> regular;
        for (auto y : grid.ys)
            for (auto x : grid.xs)
                regular.emplace_back(x, y);
        auto result = readPoints(candidate.binary, w, h, regular);
        if (result.status == ScanStatus::Decoded)
            return result;
        for (int kernel : (flow ? std::vector<int>{45, 31, 0}
                                : (limits.extraField ? std::vector<int>{31, 21, 45} : std::vector<int>{31, 45}))) {
            std::vector<cv::Point2d> points = regular;
            if (kernel) {
                auto found = work.slopes.find(kernel);
                if (found == work.slopes.end())
                    found = work.slopes.emplace(kernel, fieldSlopes(work.gray, kernel)).first;
                points = fieldPoints(found->second, grid);
            }
            if (flow) {
                result = refineGrid(work.gray, grid, points);
                if (result.status == ScanStatus::Decoded)
                    return result;
            } else {
                for (double amount : {1., .5, 1.5}) {
                    auto blend = points;
                    for (size_t i = 0; i < blend.size(); ++i)
                        blend[i] = regular[i] + amount * (points[i] - regular[i]);
                    result = readPoints(candidate.binary, w, h, blend);
                    if (result.status == ScanStatus::Decoded)
                        return result;
                }
            }
        }
    }
    return {};
}

// A matrix must contain repeated transitions in both directions. This rejects
// text strokes, solid package graphics and interiors of individual modules
// before fitting curves or constructing direction fields.
static bool matrixTexture(const cv::Mat &gray, int minTransitions = 7, int minTotal = 35)
{
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    for (int axis = 0; axis < 2; ++axis) {
        int maximum = 0, total = 0;
        for (int line : {20, 40, 64, 88, 112, 136, 160, 184, 208, 235}) {
            int transitions = 0;
            const int position = line * binary.cols / 256;
            for (int t = 16 * binary.cols / 256 + 1; t < 240 * binary.cols / 256; ++t) {
                auto a = axis ? binary.at<uchar>(t, position) : binary.at<uchar>(position, t);
                auto b = axis ? binary.at<uchar>(t - 1, position) : binary.at<uchar>(position, t - 1);
                transitions += a != b;
            }
            maximum = std::max(maximum, transitions);
            total += transitions;
        }
        if (maximum < minTransitions || total < minTotal)
            return false;
    }
    return true;
}

// Rank before geometric recovery: adjacent solid sides, opposite clock tracks,
// balanced module density and a light quiet zone distinguish matrices from text.
static double patternScore(const cv::Mat &gray)
{
    cv::Mat b;
    cv::threshold(gray, b, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const int scale = gray.cols / 64;
    std::array<double, 4> solid{};
    std::array<int, 4> clock{};
    for (int e = 0; e < 4; ++e) {
        for (int depth = 2 * scale; depth <= 13 * scale; depth += scale) {
            int black = 0, transitions = 0;
            for (int x = 7 * scale; x < 57 * scale; ++x) {
                black += b.at<uchar>(depth, x) == 0;
                if (x > 7 * scale)
                    transitions += b.at<uchar>(depth, x) != b.at<uchar>(depth, x - 1);
            }
            solid[e] = std::max(solid[e], black / (50. * scale));
            clock[e] = std::max(clock[e], transitions);
        }
        cv::rotate(b, b, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    double best = 0;
    for (int e = 0; e < 4; ++e) {
        double l = std::min(solid[e], solid[(e + 1) % 4]);
        double c = std::min(1., std::min(clock[(e + 2) % 4], clock[(e + 3) % 4]) / 16.);
        best = std::max(best, l * c);
    }
    double density = cv::mean(b(cv::Rect(6 * scale, 6 * scale, 52 * scale, 52 * scale)))[0] / 255.;
    double quiet = (cv::mean(b.row(2 * scale))[0] + cv::mean(b.row(gray.rows - 1 - 2 * scale))[0] +
                    cv::mean(b.col(2 * scale))[0] + cv::mean(b.col(gray.cols - 1 - 2 * scale))[0]) /
                   (4 * 255.);
    return best * (1 - std::abs(density - .5)) * quiet * quiet;
}

static ScanResult scanOnce(const cv::Mat &image, const ScanOptions &options, int detectionSide,
                           std::vector<RecoveryCandidate> &deferred, const SearchLimits &limits)
{
    if (image.empty() || image.depth() != CV_8U ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
        return {};
    cv::Mat gray;
    if (image.channels() == 1)
        gray = image;
    else
        cv::cvtColor(image, gray, image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    double scale = std::min(1., double(detectionSide) / std::max(gray.cols, gray.rows));
    cv::Mat small;
    cv::resize(gray, small, {std::max(1, cvRound(gray.cols * scale)), std::max(1, cvRound(gray.rows * scale))}, 0,
               0, cv::INTER_AREA);
    auto r = decodeBaseline(small);
    if (r.status == ScanStatus::Decoded)
        return r;
    const auto quads = candidates(small);
    const Quad dest = {cv::Point2f(16, 16), cv::Point2f(240, 16), cv::Point2f(240, 240), cv::Point2f(16, 240)};
    if (options.debug_output_dir)
        std::filesystem::create_directories(*options.debug_output_dir);
    struct RankedPatch {
        const LocatedCandidate *located;
        cv::Mat patch;
        double score;
    };
    std::vector<RankedPatch> ranked;
    for (const auto &located : quads) {
        auto q = located.corners;
        for (auto &p : q)
            p *= float(1 / scale);
        // A cheap thumbnail rejects non-matrix contours before a full warp.
        const Quad thumbnailDest = {cv::Point2f(4, 4), cv::Point2f(60, 4), cv::Point2f(60, 60),
                                    cv::Point2f(4, 60)};
        cv::Mat thumbnail;
        cv::warpPerspective(gray, thumbnail, cv::getPerspectiveTransform(q.data(), thumbnailDest.data()), {64, 64},
                            cv::INTER_LINEAR, cv::BORDER_CONSTANT, 255);
        if (!matrixTexture(thumbnail, 5, 25))
            continue;
        ranked.push_back({&located, thumbnail, patternScore(thumbnail)});
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) { return a.score > b.score; });
    const size_t coarseLimit = size_t(limits.patches * 4);
    if (ranked.size() > coarseLimit)
        ranked.resize(coarseLimit);
    std::vector<RankedPatch> refined;
    for (const auto &candidate : ranked) {
        auto q = candidate.located->corners;
        for (auto &p : q)
            p *= float(1 / scale);
        cv::Mat patch;
        cv::warpPerspective(gray, patch, cv::getPerspectiveTransform(q.data(), dest.data()), {256, 256},
                            cv::INTER_LINEAR, cv::BORDER_CONSTANT, 255);
        if (matrixTexture(patch))
            refined.push_back({candidate.located, patch, patternScore(patch)});
    }
    ranked = std::move(refined);
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) { return a.score > b.score; });
    if (ranked.size() > size_t(limits.patches))
        ranked.resize(limits.patches);
    int i = 0;
    std::vector<cv::Mat> retry;
    for (const auto &rankedPatch : ranked) {
        const auto &located = *rankedPatch.located;
        auto patch = rankedPatch.patch;
        auto smallTransform = cv::getPerspectiveTransform(located.corners.data(), dest.data());
        std::vector<cv::Point2f> outline;
        for (auto p : located.outline)
            outline.emplace_back(p);
        cv::perspectiveTransform(outline, outline, smallTransform);
        std::vector<cv::Point> rounded;
        for (auto p : outline)
            rounded.emplace_back(cvRound(p.x), cvRound(p.y));
        cv::Mat silhouette(256, 256, CV_8UC1, cv::Scalar(255));
        cv::fillPoly(silhouette, std::vector<std::vector<cv::Point>>{rounded}, cv::Scalar(0));
        // Normalize polarity using the quiet zone outside the proposed symbol.
        const double quiet = (cv::mean(patch.row(5))[0] + cv::mean(patch.row(250))[0] + cv::mean(patch.col(5))[0] +
                              cv::mean(patch.col(250))[0]) *
                             .25;
        const bool tryInverted = quiet < cv::mean(patch(cv::Rect(16, 16, 224, 224)))[0];
        for (int polarity = 0; polarity < (tryInverted ? 2 : 1); ++polarity) {
            if (polarity)
                cv::bitwise_not(patch, patch);
            if (options.debug_output_dir)
                cv::imwrite((std::filesystem::path(*options.debug_output_dir) /
                             ("candidate_" + std::to_string(i++) + ".png"))
                                .string(),
                            patch);
            r = sample(patch);
            if (r.status == ScanStatus::Decoded)
                return r;
            for (int envelope = 0; envelope < 3; ++envelope) {
                auto corrected =
                    envelope ? straighten(patch, silhouette, envelope == 2 ? 4 : 2) : straighten(patch);
                if (!corrected.empty()) {
                    if (options.debug_output_dir)
                        cv::imwrite((std::filesystem::path(*options.debug_output_dir) /
                                     ("corrected_" + std::to_string(i - 1) +
                                      (envelope == 2 ? "_quartic"
                                       : envelope    ? "_outline"
                                                     : "") +
                                      ".png"))
                                        .string(),
                                    corrected);
                    r = sample(corrected);
                    if (r.status == ScanStatus::Decoded)
                        return r;
                    retry.push_back(corrected);
                }
            }
        }
    }
    auto prepared = prepareRecovery(retry, limits.grids);
    r = recover(prepared, false, limits);
    if (r.status == ScanStatus::Decoded)
        return r;
    deferred.insert(deferred.end(), prepared.begin(), prepared.end());
    return {};
}

ScanResult detectAndDecode(const cv::Mat &image, const ScanOptions &options)
{
    if (image.empty() || image.depth() != CV_8U ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
        return {};
    cv::Mat gray;
    if (image.channels() == 1)
        gray = image;
    else
        cv::cvtColor(image, gray, image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    std::vector<RecoveryCandidate> deferred;
    auto result = scanOnce(gray, options, 1000, deferred, fastLimits);
    if (result.status == ScanStatus::Decoded)
        return result;
    // On a uniform dark background the connected-component detector sees the
    // background instead of the inverted symbol. Retry its opposite polarity.
    // The frame check avoids doubling localization work on ordinary photographs.
    bool darkFrame = true;
    double frameMean = 0;
    for (const auto &edge : {gray.row(0), gray.row(gray.rows - 1), gray.col(0), gray.col(gray.cols - 1)}) {
        cv::Scalar mean, deviation;
        cv::meanStdDev(edge, mean, deviation);
        darkFrame &= mean[0] < 80 && deviation[0] < 8;
        frameMean += mean[0] / 4;
    }
    if (darkFrame && cv::mean(gray)[0] > frameMean + 40) {
        cv::Mat inverted;
        cv::bitwise_not(gray, inverted);
        auto invertedOptions = options;
        if (options.debug_output_dir)
            invertedOptions.debug_output_dir =
                (std::filesystem::path(*options.debug_output_dir) / "inverted").string();
        result = scanOnce(inverted, invertedOptions, 1000, deferred, fastLimits);
        if (result.status == ScanStatus::Decoded)
            return result;
    }
    result = decodeBaseline(image);
    if (result.status == ScanStatus::Decoded)
        return result;
    auto retryOptions = options;
    if (options.debug_output_dir)
        retryOptions.debug_output_dir = (std::filesystem::path(*options.debug_output_dir) / "scale1500").string();
    if (std::max(gray.cols, gray.rows) > 1000)
        result = scanOnce(gray, retryOptions, 1500, deferred, fastLimits);
    if (result.status == ScanStatus::Decoded || !options.thorough)
        return result;
    // The extended search follows the fast path, so enabling it cannot lose a
    // fast-path decode. Its extra work is restricted to difficult images.
    for (int side : {1000, 1500}) {
        if (side == 1500 && std::max(gray.cols, gray.rows) <= 1000)
            break;
        auto extended = options;
        if (options.debug_output_dir)
            extended.debug_output_dir =
                (std::filesystem::path(*options.debug_output_dir) / ("thorough" + std::to_string(side))).string();
        result = scanOnce(gray, extended, side, deferred, thoroughLimits);
        if (result.status == ScanStatus::Decoded)
            return result;
    }
    std::stable_sort(deferred.begin(), deferred.end(),
                     [](const auto &a, const auto &b) { return a.quality < b.quality; });
    return recover(deferred, true, thoroughLimits);
}
} // namespace datamatrix
