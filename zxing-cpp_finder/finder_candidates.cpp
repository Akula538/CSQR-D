#include "finder_candidates.h"
#include "QRDetector.h"
#include "ConcentricFinder.h"
#include "BitMatrix.h"

using namespace ZXing;
using namespace ZXing::QRCode;

std::vector<FinderPatternCorners> GetFinderPatternCandidates(const BitMatrix& image)
{
    std::vector<FinderPatternCorners> result;

    // 1) Найти ВСЕ кандидаты — оригинальная функция
    auto patterns = FindFinderPatterns(image, /*tryHarder=*/true);

    // 2) Для каждого кандидата попытаться найти УГЛЫ (оригинальная функция)
    for (const auto& cp : patterns) {

        // FindConcentricPatternCorners(image, center, size, ringCount)
        auto quad = FindConcentricPatternCorners(image, cp, cp.size, 2);

        if (quad) {
            FinderPatternCorners fp;
            fp.corners.reserve(4);

            // quad -> QuadrilateralF (tl,tr,br,bl)
            for (auto& c : *quad)
                fp.corners.push_back(c);

            result.push_back(fp);
        }
    }

    return result;
}
