#include "dm_correct.hpp"
#include "DMscanner.hpp"
#include "decoder.hpp"
#include <cmath>
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
        {
            // A missing clock corner leaves a diagonal bevel. Its short real
            // timing side must survive instead of the longer diagonal cut.
            std::vector<cv::Point> clipped{{441, 208}, {498, 241}, {509, 315}, {415, 339}, {395, 223}};
            auto quad = datamatrix::geometry::supported_quad(clipped);
            require(quad.size() == 4, "Clipped timing contour needs four supported sides");
            double distance = 1e20;
            for (auto p : quad)
                distance = std::min(distance, cv::norm(cv::Point2d(p) - cv::Point2d(491, 192)));
            require(distance < 4, "Timing-corner repair must discard the diagonal bevel");
        }
        {
            datamatrix::geometry::Edge edge;
            edge.origin = {10, 20};
            edge.tangent = {1, 0};
            edge.normal = {0, 1};
            edge.length = 100;
            edge.begin = -.25;
            edge.end = 1.25;
            require(std::abs(edge.arc_length() - 150) < 1e-8, "Joined edge length must include corner extension");
            // y = 50*u^2, x = 100*u: compare against its analytic arc length.
            edge.coefficients = {12.5, 25, 12.5};
            auto primitive = [](double u) { return 50 * (u * std::sqrt(1 + u * u) + std::asinh(u)); };
            require(std::abs(edge.arc_length() - (primitive(edge.end) - primitive(edge.begin))) < .001,
                    "Field trace extent must follow the curved side, not its chord");
        }
        // A rectangular symbol under an oblique affine projection: inward
        // offsets follow the other lattice axis, not the edge's normal/length.
        {
            using namespace datamatrix::geometry;
            auto project = [](double x, double y) { return Point(100 + 7 * x + 3 * y, 50 + x + 13 * y); };
            std::array<Point, 4> corners{project(0, 0), project(32, 0), project(32, 8), project(0, 8)};
            Candidate candidate;
            for (int i = 0; i < 4; ++i) {
                auto &edge = candidate.edges[i];
                edge.origin = corners[i];
                Point delta = corners[(i + 1) % 4] - corners[i];
                edge.length = qrscanner::norm(delta);
                edge.tangent = delta / edge.length;
                edge.normal = {-edge.tangent.y, edge.tangent.x};
            }
            Grid grid;
            for (int x = 0; x < 32; ++x)
                grid.x.centers.push_back((x + .5) / 32);
            for (int y = 0; y < 8; ++y)
                grid.y.centers.push_back((y + .5) / 8);
            auto nodes = boundary_nodes(candidate, grid);
            for (size_t i = 0; i < nodes.first.size(); ++i)
                require(qrscanner::norm(nodes.second[i] - project(nodes.first[i].x, nodes.first[i].y)) < 1e-7,
                        "Rectangular lattice offset must preserve oblique axes");
            auto envelope = envelope_nodes(candidate, grid);
            for (size_t i = 0; i < envelope.first.size(); ++i)
                require(qrscanner::norm(envelope.second[i] - project(envelope.first[i].x, envelope.first[i].y)) <
                            1e-7,
                        "Smoothed clock phase must preserve an oblique rectangular lattice");
        }
        const auto square =
            render({"10101010101010", "10100011011001", "10001100011100", "11100001010001", "11110110101110",
                    "11110000000011", "10001101111000", "11111001101011", "10100010000000", "11001000001001",
                    "10001100000110", "10110001100101", "11001101100000", "11111111111111"});
        const auto rectangle = render({"10101010101010101010101010101010", "11101101101011011110111101011001",
                                       "11000100001000001010000111001110", "11101100011000111111110101111001",
                                       "10000110101010001101110001001010", "11100101010100111111110111110111",
                                       "10010100001101101000001100010000", "11111111111111111111111111111111"});
        {
            // Registration uses only image transitions, not a payload or a
            // reference bit pattern. A known phase error must shrink.
            using namespace datamatrix::geometry;
            Grid grid;
            for (int i = 0; i < 14; ++i) {
                grid.x.centers.push_back((i + .5) / 14);
                grid.y.centers.push_back((i + .5) / 14);
            }
            std::pair<Points, Points> nodes;
            for (double y : {.5, 4.5, 8.5, 13.5})
                for (double x : {.5, 4.5, 8.5, 13.5}) {
                    nodes.first.emplace_back(x, y);
                    nodes.second.emplace_back(51.5 + 12 * x + 2.4, 51.5 + 12 * y - 1.8);
                }
            require(align_lattice(square, grid, nodes, 0).first.empty(),
                    "Invalid refinement mesh must be rejected");
            for (int iteration = 0; iteration < 3; ++iteration)
                nodes = align_lattice(square, grid, nodes);
            double error = 0;
            for (size_t i = 0; i < nodes.first.size(); ++i)
                error += qrscanner::norm(nodes.second[i] - (Point(51.5, 51.5) + nodes.first[i] * 12));
            require(error / nodes.first.size() < 1.8,
                    "Photometric lattice refinement must reduce known phase error");

            qrscanner::GridDecodeCache cache;
            auto exact = square(cv::Rect(28, 28, 216, 216));
            for (int repeat = 0; repeat < 2; ++repeat) {
                auto decoded =
                    qrscanner::decode_matrix_grid(exact, 14, 14, ZXing::BarcodeFormat::DataMatrix, 12, &cache);
                require(decoded && *decoded == "HELLO123", "A decode cache must not suppress a successful matrix");
            }
        }
        // Weak contrast is a recovery heuristic, not permission to bypass a
        // successful decoder. Conversely, a finder without payload must fail.
        cv::Mat lowContrast;
        square.convertTo(lowContrast, CV_8U, 1. / 8, 112);
        expect(lowContrast, "HELLO123");
        cv::Mat erased = square.clone();
        erased(cv::Rect(64, 64, 12 * 12, 12 * 12)).setTo(128);
        require(datamatrix::detectAndDecode(erased).status != qrscanner::ScanStatus::Decoded,
                "A finder with erased payload must not decode");
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
        {
            using namespace datamatrix::geometry;
            cv::Mat shadow = curvedSource.clone();
            for (int y = 0; y < shadow.rows; ++y)
                shadow.row(y).convertTo(shadow.row(y), CV_8U, .18 + .82 * y / shadow.rows);
            Candidate candidate;
            candidate.bounds = {52, 52, 264, 264};
            std::array<Point, 4> corners{Point(52, 52), Point(316, 52), Point(316, 316), Point(52, 316)};
            for (int i = 0; i < 4; ++i) {
                auto &edge = candidate.edges[i];
                edge.origin = corners[i];
                edge.length = 264;
                edge.tangent = (corners[(i + 1) % 4] - corners[i]) / edge.length;
                edge.normal = {-edge.tangent.y, edge.tangent.x};
            }
            auto clocks = read_clocks(shadow, candidate);
            require(!clocks.empty() && clocks.front().x.centers.size() == 22 &&
                        clocks.front().y.centers.size() == 22,
                    "Clock reading must survive a shadow that defeats a global threshold");
        }
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
            if (amplitude == 30.) {
                // Only about 1.5 search-image pixels per module at a 1000 px
                // detection side. Decode from the original crop, not that copy.
                cv::Mat small, canvas(3072, 4080, CV_8UC1, cv::Scalar(255));
                cv::resize(curved, small, {184, 184}, 0, 0, cv::INTER_AREA);
                small.copyTo(canvas(cv::Rect(1900, 1400, small.cols, small.rows)));
                require(datamatrix::decodeBaseline(canvas).status != qrscanner::ScanStatus::Decoded,
                        "Small curved fixture must exercise recovery");
                expect(canvas, "Curved Data Matrix regression check 2026");
            }
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
        cv::Mat framed(480, 640, CV_8UC1, cv::Scalar(255));
        cv::rectangle(framed, {5, 5}, {634, 474}, cv::Scalar(0), 8);
        cv::putText(framed, "1234567890", {150, 250}, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0), 3);
        for (const auto &candidate : datamatrix::geometry::locate(framed))
            require(candidate.bounds.width < framed.cols * .9 || candidate.bounds.height < framed.rows * .9,
                    "Background/frame must not become a full-image Data Matrix candidate");
        std::cout << "Data Matrix checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
