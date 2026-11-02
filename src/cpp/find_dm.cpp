#include "find_dm.hpp"
#include "DMTrace.hpp"
#include "binarizer.hpp"
#include "datamatrix/DMVersion.h"
#include "vectorfield_correct.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>

namespace datamatrix::geometry {
using qrscanner::norm;
using qrscanner::normalize;
static std::vector<Grid> read_clock_tracks(const cv::Mat& binary, const Candidate& candidate, bool all = false);
Point Edge::at(double t) const
{
    double u = begin + t * (end - begin), z = 2 * u - 1;
    double offset = coefficients[0] + z * (coefficients[1] + z * coefficients[2]);
    return origin + tangent * (u * length) + normal * offset;
}
Point Edge::direction(double t) const { return normalize(at(t + .001) - at(t - .001)); }
double Edge::arc_length() const
{
    // The fitted chord is a coordinate scale, not the extent after joining
    // corners. Integrate the derivative over the actual joined curve.
    constexpr int steps = 16;
    double sum = 0;
    for (int i = 0; i <= steps; ++i) {
        double u = begin + (end - begin) * i / steps;
        double slope = 2 * coefficients[1] + 4 * (2 * u - 1) * coefficients[2];
        double speed = norm(tangent * length + normal * slope);
        sum += speed * (i == 0 || i == steps ? 1 : i % 2 ? 4 : 2);
    }
    return sum * std::abs(end - begin) / (3 * steps);
}

static double pixel(const cv::Mat& image, Point p)
{
    int x = cvRound(p.x), y = cvRound(p.y);
    if (x < 0 || y < 0 || x >= image.cols || y >= image.rows)
        return 1;
    return image.at<uchar>(y, x) / 255.;
}

// Fit the outer envelope, not the inward gaps between timing modules. The
// contour supplies observations in source coordinates; robust reweighting
// excludes the inward steps without ever making a perspective image.
static Edge fit_edge(const std::vector<cv::Point>& contour, Point a, Point b)
{
    Edge e;
    e.origin = a;
    e.length = norm(b - a);
    e.tangent = (b - a) / e.length;
    e.normal = {-e.tangent.y, e.tangent.x};
    constexpr int bins = 80;
    std::array<double, bins> outer;
    outer.fill(1e20);
    for (auto pp : contour) {
        Point p = Point(pp) - a;
        double t = p.dot(e.tangent) / e.length, d = p.dot(e.normal);
        if (t >= 0 && t < 1 && std::abs(d) < e.length * .22)
            outer[int(t * bins)] = std::min(outer[int(t * bins)], d);
    }
    auto fit = [&]() {
        std::vector<double> sorted;
        for (double d : outer)
            if (d < 1e19)
                sorted.push_back(d);
        if (sorted.size() < 12)
            return;
        std::sort(sorted.begin(), sorted.end());
        double cutoff = sorted[sorted.size() * 65 / 100];
        std::array<bool, bins> use{};
        for (int i = 0; i < bins; ++i)
            use[i] = outer[i] <= cutoff;
        for (int iteration = 0; iteration < 6; ++iteration) {
            int count = std::count(use.begin(), use.end(), true);
            if (count < 12)
                return;
            cv::Mat A(count, 3, CV_64F), B(count, 1, CV_64F), coeff;
            int row = 0;
            for (int i = 0; i < bins; ++i)
                if (use[i]) {
                    double z = 2 * (i + .5) / bins - 1;
                    A.at<double>(row, 0) = 1;
                    A.at<double>(row, 1) = z;
                    A.at<double>(row, 2) = z * z;
                    B.at<double>(row++) = outer[i];
                }
            cv::solve(A, B, coeff, cv::DECOMP_SVD);
            for (int k = 0; k < 3; ++k)
                e.coefficients[k] = coeff.at<double>(k);
            for (int i = 0; i < bins; ++i) {
                double r = outer[i] - (e.at((i + .5) / bins) - a).dot(e.normal);
                use[i] = r < e.length * 3 / 224 && r > -e.length * 6 / 224;
            }
        }
    };
    fit();
    return e;
}

static void join_corners(std::array<Edge, 4>& edges)
{
    std::array<double, 4> starts{}, ends{};
    ends.fill(1);
    for (int i = 0; i < 4; ++i) {
        const auto& a = edges[(i + 3) % 4];
        const auto& b = edges[i];
        double u = 1, v = 0;
        for (int it = 0; it < 8; ++it) {
            Point da = (a.at(u + .0001) - a.at(u - .0001)) / .0002;
            Point db = (b.at(v + .0001) - b.at(v - .0001)) / .0002;
            Point delta = b.at(v) - a.at(u);
            double determinant = qrscanner::cross(da, db);
            if (std::abs(determinant) < 1e-6)
                break;
            u += qrscanner::cross(delta, db) / determinant;
            v += qrscanner::cross(delta, da) / determinant;
        }
        ends[(i + 3) % 4] = std::clamp(u, .65, 1.35);
        starts[i] = std::clamp(v, -.35, .35);
    }
    for (int i = 0; i < 4; ++i) {
        double begin = edges[i].begin, span = edges[i].end - begin;
        edges[i].begin = begin + starts[i] * span;
        edges[i].end = begin + ends[i] * span;
    }
}

struct EdgeEvidence {
    double solid = 0, clock = 0, quiet = 0;
    int transitions = 0;
};
static EdgeEvidence evidence(const cv::Mat& binary, const Edge& edge)
{
    EdgeEvidence out;
    int solid = 0, clock = 0, quiet = 0;
    bool last = false;
    for (int i = 0; i < 160; ++i) {
        double t = (i + .5) / 160;
        Point d = edge.direction(t), n(-d.y, d.x), p = edge.at(t);
        bool black = pixel(binary, p + n * std::max(1.5, edge.length * .018)) < .5;
        bool support = false;
        for (double depth = std::max(.5, edge.length * .008); depth <= std::max(1.5, edge.length * .035);
             depth += .5)
            support |= pixel(binary, p + n * depth) < .5;
        solid += support;
        clock += black;
        quiet += pixel(binary, p - n * std::max(2., edge.length * .018)) > .5;
        if (i && black != last)
            ++out.transitions;
        last = black;
    }
    out.solid = solid / 160.;
    out.clock = clock / 160.;
    out.quiet = quiet / 160.;
    return out;
}

// Recover the missing timing corner by intersecting the four supported sides.
// A short bevel is not a side and must not pull a corner into the data region.
std::vector<cv::Point> supported_quad(const std::vector<cv::Point>& contour)
{
    std::vector<cv::Point> hull, quad;
    cv::convexHull(contour, hull);
    if (hull.size() < 4)
        return {};
    cv::approxPolyN(hull, quad, 4, .35f, false);
    if (quad.size() != 4 || !cv::isContourConvex(quad))
        return {};
    // A clipped timing corner adds a bevel to the hull. Minimum added area
    // alone prefers that bevel over a real side. Compare supporting-line
    // quadrilaterals, penalizing skew as well as excess area.
    std::vector<cv::Point> polygon;
    cv::approxPolyDP(hull, polygon, .015 * cv::arcLength(hull, true), true);
    const double hullArea = cv::contourArea(hull);
    auto score = [&](const std::vector<cv::Point>& q) {
        double area = cv::contourArea(q);
        if (!cv::isContourConvex(q) || area < .98 * hullArea || area > 2 * hullArea)
            return 1e20;
        double value = area / hullArea;
        for (int i = 0; i < 4; ++i) {
            auto a = Point(q[(i + 1) % 4] - q[i]), b = Point(q[(i + 3) % 4] - q[i]);
            if (norm(a) < 18 || norm(b) < 18)
                return 1e20;
            double cosine = a.dot(b) / (norm(a) * norm(b));
            value += cosine * cosine;
        }
        return value;
    };
    double best = score(quad);
    // Only an implausibly skewed initial fit needs corner repair. Preserve
    // well-supported perspective quadrilaterals rather than squaring them.
    if (best - cv::contourArea(quad) / hullArea < .5)
        return quad;
    int n = int(polygon.size());
    if (n == 5)
        for (int omitted = 0; omitted < n; ++omitted) {
            std::array<int, 4> sides{};
            int selected = 0;
            for (int i = 0; i < n; ++i)
                if (i != omitted)
                    sides[selected++] = i;
            std::vector<cv::Point> proposal;
            for (int i = 0; i < 4; ++i) {
                int j = sides[i], k = sides[(i + 1) % 4];
                Point p = polygon[j], q = polygon[k];
                Point u = Point(polygon[(j + 1) % n]) - p;
                Point v = Point(polygon[(k + 1) % n]) - q;
                double determinant = qrscanner::cross(u, v);
                if (std::abs(determinant) < .05 * norm(u) * norm(v))
                    break;
                Point hit = p + u * (qrscanner::cross(q - p, v) / determinant);
                proposal.emplace_back(cvRound(hit.x), cvRound(hit.y));
            }
            if (proposal.size() == 4 && score(proposal) < best) {
                best = score(proposal);
                quad = proposal;
            }
        }
    return quad;
}

std::vector<Candidate> locate(const cv::Mat& gray)
{
    cv::Mat binary, black;
    // Candidate search operates on the reduced image; full-resolution pixels
    // are retained for boundary refinement, clock reading and the field.
    int unit = std::max(1, std::max(gray.cols, gray.rows) / 1000);
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 41 * unit | 1, 7);
    cv::bitwise_not(binary, black);
    std::vector<Candidate> found;
    for (int radius : {0, 1, 2, 3, 4, 6}) {
        int k = 2 * radius * unit + 1;
        cv::Mat mask;
        cv::morphologyEx(black, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {k, k}));
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        for (const auto& contour : contours) {
            auto bounds = cv::boundingRect(contour);
            if (bounds.width < 24 || bounds.height < 24 || bounds.width > bounds.height * 5 ||
                bounds.height > bounds.width * 5)
                continue;
            // Extent is checked independently of contour area: holes/notches
            // in the background must not turn the entire frame into a symbol.
            if (bounds.width >= gray.cols * .92 && bounds.height >= gray.rows * .92)
                continue;
            if (bounds.x <= 1 || bounds.y <= 1 || bounds.br().x >= gray.cols - 1 || bounds.br().y >= gray.rows - 1)
                continue;
            double area = cv::contourArea(contour);
            if (area < 400 || area < bounds.area() * .28)
                continue;
            std::vector<std::vector<cv::Point>> proposals{supported_quad(contour)};
            std::vector<cv::Point> hull, inscribed;
            cv::convexHull(contour, hull);
            cv::approxPolyDP(hull, inscribed, .04 * cv::arcLength(hull, true), true);
            if (inscribed.size() == 4)
                proposals.push_back(inscribed);
            for (size_t proposal = 0; proposal < proposals.size(); ++proposal) {
                auto quad = proposals[proposal];
                if (quad.size() != 4)
                    continue;
                if (cv::contourArea(quad, true) < 0)
                    std::reverse(quad.begin(), quad.end());
                Candidate candidate;
                candidate.proposal_kind = int(proposal);
                candidate.bounds = bounds;
                bool sensible = true;
                for (int i = 0; i < 4; ++i) {
                    if (norm(Point(quad[(i + 1) % 4]) - Point(quad[i])) < 18) {
                        sensible = false;
                        break;
                    }
                    candidate.edges[i] = fit_edge(contour, quad[i], quad[(i + 1) % 4]);
                }
                if (!sensible)
                    continue;
                join_corners(candidate.edges);
                std::array<EdgeEvidence, 4> initial;
                for (int i = 0; i < 4; ++i)
                    initial[i] = evidence(binary, candidate.edges[i]);
#ifdef CSQR_DM_TRACE
                diagnostics::Lines debugEdges(4);
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j <= 40; ++j)
                        debugEdges[i].push_back(candidate.edges[i].at(j / 40.));
                DM_TRACE("proposal", {},
                         {{"x", bounds.x},
                          {"y", bounds.y},
                          {"w", bounds.width},
                          {"h", bounds.height},
                          {"s0", initial[0].solid},
                          {"s1", initial[1].solid},
                          {"s2", initial[2].solid},
                          {"s3", initial[3].solid},
                          {"q0", initial[0].quiet},
                          {"q1", initial[1].quiet},
                          {"q2", initial[2].quiet},
                          {"q3", initial[3].quiet}},
                         {}, debugEdges);
#endif
                int rotation = -1;
                double best = 0;
                for (int i = 0; i < 4; ++i) {
                    // Rotation only changes side labels; reuse their measured
                    // evidence instead of sampling the same pixels again.
                    auto t = initial[i], r = initial[(i + 1) % 4], b = initial[(i + 2) % 4],
                         l = initial[(i + 3) % 4];
                    if (b.solid < .85 || l.solid < .85 || t.clock < .05 || t.clock > .95 || r.clock < .05 ||
                        r.clock > .95 || t.transitions < 3 || r.transitions < 3)
                        continue;
                    double quiet = (t.quiet + r.quiet + b.quiet + l.quiet) / 4;
                    if (quiet < .65)
                        continue;
                    double score = b.solid + l.solid + quiet - std::abs(t.clock - .5) - std::abs(r.clock - .5);
                    if (score > best) {
                        rotation = i;
                        best = score;
                    }
                }
                if (rotation < 0)
                    continue;
                std::rotate(candidate.edges.begin(), candidate.edges.begin() + rotation, candidate.edges.end());
                auto clocks = read_clock_tracks(binary, candidate);
                candidate.score = best - 4 * (clocks.empty() ? .5 : clocks.front().score());
                bool duplicate = false;
                for (auto& old : found) {
                    if (old.proposal_kind != candidate.proposal_kind)
                        continue;
                    double distance = 0;
                    for (int i = 0; i < 4; ++i)
                        distance += norm(old.edges[i].at(.5) - candidate.edges[i].at(.5));
                    if (distance < (candidate.proposal_kind ? .02 : .07) * (bounds.width + bounds.height)) {
                        if (candidate.score > old.score)
                            old = candidate;
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate)
                    found.push_back(candidate);
            }
        }
    }
    std::sort(found.begin(), found.end(), [](auto& a, auto& b) { return a.score > b.score; });
    DM_TRACE("located", {}, {{"candidates", double(found.size())}});
    return found;
}

Candidate refine(const cv::Mat& gray, Candidate candidate)
{
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                          std::max(31, std::min(gray.cols, gray.rows) / 3) | 1, 7);
    for (int side = 0; side < 4; ++side) {
        auto& edge = candidate.edges[side];
        std::vector<cv::Point> observations;
#ifdef CSQR_DM_TRACE
        diagnostics::Lines extensions;
#endif
        const double band = std::max(3., edge.length * .04);
        for (int i = 0; i < 160; ++i) {
            double t = (i + .5) / 160;
            Point p = edge.at(t), d = edge.direction(t), n(-d.y, d.x);
            double start = -band;
            if (side < 2 && pixel(binary, p + n * start) < .5 && pixel(binary, p + n * (start + 1)) < .5) {
                for (double offset = start - .5; offset >= -3 * band; offset -= .5)
                    if (pixel(binary, p + n * offset) > .5 && pixel(binary, p + n * (offset - 1)) > .5) {
                        start = offset;
#ifdef CSQR_DM_TRACE
                        extensions.push_back({p - n * band, p + n * start});
#endif
                        break;
                    }
            }
            for (double offset = start; offset <= band; offset += .5) {
                if (pixel(binary, p + n * offset) < .5 && pixel(binary, p + n * (offset + 1)) < .5) {
                    auto hit = p + n * offset;
                    observations.emplace_back(cvRound(hit.x), cvRound(hit.y));
                    break;
                }
            }
        }
        if (observations.size() >= 40)
            edge = fit_edge(observations, edge.at(0), edge.at(1));
        DM_TRACE("boundary_refinement", {}, {{"side", side}, {"band", band}}, {}, extensions);
    }
    join_corners(candidate.edges);
    return candidate;
}

static Clock fit_clock(const std::vector<double>& signal, int modules, bool first_black)
{
    int length = int(signal.size());
    double pitch = double(length) / modules;
    int minWidth = std::max(1, int(pitch * .45)), maxWidth = int(pitch * 1.8) + 1;
    std::vector<double> prefix(length + 1), cost((modules + 1) * (length + 1), 1e20);
    std::vector<int> previous(cost.size());
    std::partial_sum(signal.begin(), signal.end(), prefix.begin() + 1);
    cost[0] = 0;
    for (int cell = 1; cell <= modules; ++cell) {
        bool black = (cell % 2 == 1) == first_black;
        int lo = std::max(cell * minWidth, length - (modules - cell) * maxWidth);
        int hi = std::min(cell * maxWidth, length - (modules - cell) * minWidth);
        for (int end = lo; end <= hi; ++end)
            for (int width = minWidth; width <= maxWidth && width <= end; ++width) {
                double white = prefix[end] - prefix[end - width];
                double mismatch = black ? white : width - white;
                double value = cost[(cell - 1) * (length + 1) + end - width] + mismatch +
                               .12 * (width - pitch) * (width - pitch) / pitch;
                int index = cell * (length + 1) + end;
                if (value < cost[index]) {
                    cost[index] = value;
                    previous[index] = end - width;
                }
            }
    }
    Clock out;
    out.cost = cost.back() / length;
    out.centers.resize(modules);
    int end = length;
    for (int cell = modules; cell > 0; --cell) {
        int start = previous[cell * (length + 1) + end];
        out.centers[cell - 1] = (start + end) * .5 / length;
        end = start;
    }
    // The clock/clock corner is white, so the component contour cannot locate
    // its outer boundary. Infer that white cell from the two adjacent black
    // cells instead of treating the contour intersection as a measured edge.
    if (modules >= 4) {
        if (first_black)
            out.centers.back() =
                out.centers[modules - 2] + .5 * (out.centers[modules - 2] - out.centers[modules - 4]);
        else
            out.centers.front() = out.centers[1] - .5 * (out.centers[3] - out.centers[1]);
    }
    return out;
}

Point inset_point(const Candidate& c, int side, double t, int across_modules, double depth)
{
    const auto& previous = c.edges[(side + 3) % 4];
    const auto& next = c.edges[(side + 1) % 4];
    double step = depth / across_modules;
    Point atStart = previous.at(1 - step) - previous.at(1);
    Point atEnd = next.at(step) - next.at(0);
    return c.edges[side].at(t) + atStart * (1 - t) + atEnd * t;
}

static cv::Mat clock_binary(const cv::Mat& gray, const Candidate& c)
{
    int margin = std::max(c.bounds.width, c.bounds.height) / 12;
    auto bounds = (c.bounds + cv::Size(2 * margin, 2 * margin) - cv::Point(margin, margin)) &
                  cv::Rect(0, 0, gray.cols, gray.rows);
    cv::Mat binary, local;
    double level = cv::threshold(gray(bounds), local, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(gray, binary, level, 255, cv::THRESH_BINARY);
    // The quiet strip is white even under a fold's shadow. If the global
    // threshold makes a substantial part black, use QR's local binarizer.
    bool unevenLight = false;
    for (int axis = 0; axis < 2; ++axis) {
        int dark = 0;
        const auto& edge = c.edges[axis];
        for (int i = 0; i < 32; ++i) {
            double t = (i + .5) / 32;
            Point direction = edge.direction(t), normal(-direction.y, direction.x);
            dark += pixel(binary, edge.at(t) - normal * std::max(2., edge.length / 24)) < .5;
        }
        unevenLight |= dark > 8;
    }
    if (unevenLight) {
        auto localBinary = qrscanner::binarize_zxing_cpp(gray);
        cv::cvtColor(localBinary, binary, cv::COLOR_BGR2GRAY);
    }
    return binary;
}

std::vector<Grid> read_clocks(const cv::Mat& gray, const Candidate& c)
{
    auto grids = read_clock_tracks(clock_binary(gray, c), c, true);
    auto relaxed = grids;
    // Validate on the source crop, where small timing cells are still visible.
    // A loose coarse-image score must not send text/noise into field recovery.
    grids.erase(std::remove_if(grids.begin(), grids.end(),
                               [](const Grid& g) { return g.x.cost > .12 || g.y.cost > .12 || g.score() > .18; }),
                grids.end());
    if (grids.empty() && c.score >= 2) {
        for (const auto& g : relaxed)
            if (g.x.cost < .18 && g.y.cost < .18 && g.score() < .28)
                grids.push_back(g);
        if (grids.size() > 3)
            grids.resize(3);
    } else {
        size_t keep = grids.size() > 1 && grids[1].score() <= 1.5 * grids[0].score() ? 2 : 1;
        if (grids.size() > keep)
            grids.resize(keep);
    }
    return grids;
}

static std::vector<Grid> read_clock_tracks(const cv::Mat& binary, const Candidate& c, bool all)
{
    std::array<std::vector<double>, 2> signal;
    std::array<int, 2> estimate{};
    for (int axis = 0; axis < 2; ++axis) {
        auto& edge = c.edges[axis];
        auto& s = signal[axis];
        for (int i = 0; i < 256; ++i) {
            double t = (i + .5) / 256;
            Point d = edge.direction(t), n(-d.y, d.x);
            s.push_back(pixel(binary, edge.at(t) + n * std::max(1.5, edge.length * .018)));
        }
        auto raw = s;
        for (int i = 0; i < 256; ++i) {
            int votes = 0;
            for (int j = -2; j <= 2; ++j)
                votes += raw[std::clamp(i + j, 0, 255)] > .5;
            s[i] = votes >= 3 ? 1. : 0.;
        }
        for (int i = 1; i < 256; ++i)
            estimate[axis] += (s[i] > .5) != (s[i - 1] > .5);
        estimate[axis] = (estimate[axis] + 2) / 2 * 2;
    }
    DM_TRACE("clock_estimate", binary, {{"x", estimate[0]}, {"y", estimate[1]}});
    std::vector<Grid> grids;
    int low = std::max(8, std::min(estimate[0], estimate[1]) - 4),
        high = std::min(144, std::max(estimate[0], estimate[1]) + 4);
    for (int w = low; w <= high; w += 2)
        for (int h = low; h <= high; h += 2) {
            if (w != h && (std::abs(w - estimate[0]) > 4 || std::abs(h - estimate[1]) > 4))
                continue;
            if (!ZXing::DataMatrix::VersionForDimensions(h, w))
                continue;
            std::array<std::vector<double>, 2> tracks;
            for (int axis = 0; axis < 2; ++axis) {
                for (int i = 0; i < 256; ++i) {
                    double t = (i + .5) / 256;
                    double white = 0;
                    // Integrate inside the timing row, without crossing into
                    // the data area or testing alternative sampling tracks.
                    for (double depth : {.25, .5, .75})
                        white += pixel(binary, inset_point(c, axis, t, axis ? w : h, depth));
                    tracks[axis].push_back(white >= 1.5 ? 1. : 0.);
                }
            }
            Grid g{fit_clock(tracks[0], w, true), fit_clock(tracks[1], h, false)};
            DM_TRACE("clock_fit", {}, {{"w", w}, {"h", h}, {"x_cost", g.x.cost}, {"y_cost", g.y.cost}});
            if (g.x.cost < .24 && g.y.cost < .24)
                grids.push_back(g);
        }
    std::sort(grids.begin(), grids.end(), [](auto& a, auto& b) { return a.score() < b.score(); });
    // Keep a second dimension only when timing evidence is ambiguous. A
    // clearly worse segmentation does not justify two more field traversals.
    size_t keep = grids.size() > 1 && grids[1].score() <= 1.5 * grids[0].score() ? 2 : 1;
    if (!all && grids.size() > keep)
        grids.resize(keep);
    return grids;
}

Candidate to_source(Candidate candidate, double detection_scale)
{
    for (auto& edge : candidate.edges) {
        edge.origin /= detection_scale;
        edge.length /= detection_scale;
        for (auto& coefficient : edge.coefficients)
            coefficient /= detection_scale;
    }
    candidate.bounds = {
        cvFloor(candidate.bounds.x / detection_scale), cvFloor(candidate.bounds.y / detection_scale),
        cvCeil(candidate.bounds.width / detection_scale), cvCeil(candidate.bounds.height / detection_scale)};
    return candidate;
}

Candidate translated(Candidate c, Point shift)
{
    c.bounds += cv::Point(cvRound(shift.x), cvRound(shift.y));
    for (auto& e : c.edges)
        e.origin += shift;
    return c;
}
Grid refine_clocks(qrscanner::DirectionField& field, const cv::Mat& gray, const Candidate& c, Grid grid)
{
    cv::Mat binary = clock_binary(gray, c);
    int w = int(grid.x.centers.size()), h = int(grid.y.centers.size());
    double pitch = (c.edges[0].length / w + c.edges[1].length / h) * .5;
    int kernel = std::max(35, int(3 * pitch)) | 1;
    for (int axis = 0; axis < 2; ++axis) {
        auto& clock = axis ? grid.y : grid.x;
        auto& edge = c.edges[axis];
        auto line = qrscanner::constructPolylineBetween(field, inset_point(c, axis, 0, axis ? w : h),
                                                        inset_point(c, axis, 1, axis ? w : h), edge.direction(0),
                                                        -edge.direction(1), 30, kernel);
        double length = line.p2t(line.points.back());
        std::vector<double> signal;
        for (int i = 0; i < 256; ++i)
            signal.push_back(pixel(binary, line.t2p(length * (i + .5) / 256)));
        auto fitted = fit_clock(signal, axis ? h : w, axis == 0);
        if (fitted.cost > clock.cost + .03)
            continue;
        Points points;
        bool valid = true;
        for (size_t i = 0; i < fitted.centers.size(); ++i) {
            auto p = line.t2p(length * fitted.centers[i]);
            valid &= norm(p - inset_point(c, axis, clock.centers[i], axis ? w : h)) < pitch;
            points.push_back(p);
        }
        if (valid)
            clock.points = points;
        DM_TRACE("clock_line", {}, {{"axis", axis}, {"accepted", valid}, {"cost", fitted.cost}}, {}, {line.points},
                 points);
    }
    return grid;
}
} // namespace datamatrix::geometry
