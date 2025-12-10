#pragma once
#include <vector>
#include "Point.h"

namespace ZXing {
class BitMatrix;
}

struct FinderPatternCorners
{
    std::vector<ZXing::PointF> corners; // size = 4
};

std::vector<FinderPatternCorners> GetFinderPatternCandidates(const ZXing::BitMatrix& image);
