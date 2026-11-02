#pragma once
#include "tools.hpp"

namespace qrscanner {
class DirectionField;
}
#include <array>

namespace datamatrix::geometry {
using qrscanner::Point;
using qrscanner::Points;

// Coordinates belong to the image used by this stage. The scanner maps coarse
// detections back to source pixels before constructing the crop and its field.
struct Edge {
    Point origin, tangent, normal;
    std::array<double, 3> coefficients{};
    double length = 0, begin = 0, end = 1;
    Point at(double t) const;
    Point direction(double t) const;
    double arc_length() const;
};
struct Candidate {
    std::array<Edge, 4> edges; // clockwise, top/right clocks, bottom/left solid
    cv::Rect bounds;
    double score = 0;
    int proposal_kind = 0;
};
struct Clock {
    std::vector<double> centers;
    Points points;
    double cost = 1;
};
struct Grid {
    Clock x, y;
    double score() const { return x.cost + y.cost; }
};
std::vector<Candidate> locate(const cv::Mat& gray);
// Supporting-line envelope, including repair of a clipped timing corner.
std::vector<cv::Point> supported_quad(const std::vector<cv::Point>& contour);
Candidate refine(const cv::Mat& gray, Candidate candidate);
std::vector<Grid> read_clocks(const cv::Mat& gray, const Candidate& candidate);
Grid refine_clocks(qrscanner::DirectionField& field, const cv::Mat& gray, const Candidate& candidate, Grid grid);
Candidate to_source(Candidate candidate, double detection_scale);
Candidate translated(Candidate candidate, Point shift);
Point inset_point(const Candidate& candidate, int side, double t, int across_modules, double depth = .5);
} // namespace datamatrix::geometry
