#pragma once

#include "tools.hpp"

#include <array>

namespace qrscanner {

struct ParabolaFit {
    Point A;
    Point B;
    Point C;
};

ParabolaFit fit_parabola_from4(const std::array<Point, 4>& points, double tol = 1e-12, int maxiter = 100);

} // namespace qrscanner
