#pragma once

#include "find_dm.hpp"
#include "vectorfield_correct.hpp"

namespace datamatrix::geometry {

std::pair<Points, Points> boundary_nodes(const Candidate& candidate, const Grid& grid);
std::pair<Points, Points> envelope_nodes(const Candidate& candidate, const Grid& grid);
// Coarse photometric alignment; all samples come from the untouched source crop.
std::pair<Points, Points> align_lattice(const cv::Mat& gray, const Grid& grid,
                                        const std::pair<Points, Points>& nodes, int mesh = 5,
                                        double* maximum_shift = nullptr);
// Joint refinement against the source field, clock anchors and image transitions.
std::pair<Points, Points> refine_lattice(const cv::Mat& gray, const Candidate& candidate, const Grid& grid,
                                         const std::pair<Points, Points>& nodes, qrscanner::DirectionField& field);
std::pair<Points, Points> field_nodes(qrscanner::DirectionField& field, const Candidate& candidate,
                                      const Grid& grid, int spacing = 5, int steps = 20);
} // namespace datamatrix::geometry
