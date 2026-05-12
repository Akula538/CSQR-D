#include "math_base.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace qrscanner {

ParabolaFit fit_parabola_from4(const std::array<Point, 4>& points, double tol, int maxiter)
{
    const Point P0 = points[0];
    const Point P1 = points[1];
    const Point P2 = points[2];
    const Point P3 = points[3];
    const Point A = P0;
    const Point D = P3 - P0;

    auto F = [&](const cv::Vec2d& tt) {
        const double t1 = tt[0];
        const double t2 = tt[1];
        const double s1 = t1 * t1 - t1;
        const double s2 = t2 * t2 - t2;
        const Point V1 = (P1 - A) - D * t1;
        const Point V2 = (P2 - A) - D * t2;
        const Point r = V1 * s2 - V2 * s1;
        return cv::Vec2d(r.x, r.y);
    };

    auto norm2 = [](const cv::Vec2d& v) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1]);
    };

    auto newton_solve = [&](cv::Vec2d initial, cv::Vec2d& out) {
        cv::Vec2d t = initial;
        for (int k = 0; k < maxiter; ++k) {
            const cv::Vec2d f = F(t);
            const double fnorm = norm2(f);
            if (fnorm < tol) {
                out = t;
                return true;
            }

            const double eps = k < 5 ? 1e-8 : 1e-9;
            cv::Matx22d J;
            for (int i = 0; i < 2; ++i) {
                cv::Vec2d dt(0.0, 0.0);
                dt[i] = eps;
                const cv::Vec2d col = (F(t + dt) - f) * (1.0 / eps);
                J(0, i) = col[0];
                J(1, i) = col[1];
            }

            cv::Vec2d delta;
            if (!cv::solve(cv::Mat(J), cv::Mat(-f), cv::Mat(delta), cv::DECOMP_LU)) {
                out = t;
                return false;
            }

            double alpha = 1.0;
            bool accepted = false;
            for (int step = 0; step < 20; ++step) {
                const cv::Vec2d t_new = t + alpha * delta;
                if (!(-2.0 < t_new[0] && t_new[0] < 3.0 && -2.0 < t_new[1] && t_new[1] < 3.0)) {
                    alpha *= 0.5;
                    continue;
                }
                const cv::Vec2d f_new = F(t_new);
                if (norm2(f_new) < fnorm + 1e-15) {
                    t = t_new;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if (!accepted) {
                out = t;
                return false;
            }
        }
        out = t;
        return false;
    };

    const std::vector<cv::Vec2d> guesses = {
        {0.25, 0.5},
        {0.2, 0.8},
        {0.33, 0.66},
        {0.15, 0.4},
        {0.4, 0.7},
    };

    bool found = false;
    cv::Vec2d solution;
    for (const auto& g : guesses) {
        cv::Vec2d tsol;
        if (!newton_solve(g, tsol))
            continue;
        if (0.0 < tsol[0] && tsol[0] < tsol[1] && tsol[1] < 1.0) {
            solution = tsol;
            found = true;
            break;
        }
    }

    if (!found) {
        for (const auto& g : guesses) {
            cv::Vec2d tsol;
            if (newton_solve(g, tsol)) {
                solution = tsol;
                found = true;
                break;
            }
        }
    }

    if (!found)
        throw std::runtime_error("Failed to fit parabola from four points");

    const double t1 = solution[0];
    const double s1 = t1 * t1 - t1;
    const Point V1 = (P1 - A) - D * t1;
    if (std::abs(s1) < 1e-10)
        throw std::runtime_error("Degenerate parabola fit");

    const Point C = V1 * (1.0 / s1);
    const Point B = D - C;
    return {A, B, C};
}

} // namespace qrscanner
