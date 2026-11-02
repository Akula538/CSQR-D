// Optional benchmark adapter. The historical scanner is compiled unchanged
// under another namespace; it is never linked into the production CLI/library.
#define datamatrix legacy_datamatrix
#include "DMscanner.hpp"
#include CSQR_DM_LEGACY_SOURCE
#undef datamatrix

qrscanner::ScanResult decodeLegacy(const cv::Mat &image) { return legacy_datamatrix::detectAndDecode(image); }
