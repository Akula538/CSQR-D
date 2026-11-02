#pragma once

// These hooks exist only in the separate diagnostic executable. The regular
// library compiles every DM_TRACE/DM_SCOPE expression out, including arguments.
#ifdef CSQR_DM_TRACE
#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ZXing
{
class BitMatrix;
class DecoderResult;
} // namespace ZXing
namespace datamatrix::diagnostics
{
using Numbers = std::map<std::string, double>;
using Strings = std::map<std::string, std::string>;
using Lines = std::vector<std::vector<cv::Point2d>>;
int emit(const std::string &kind, const cv::Mat &image = {}, const Numbers &numbers = {},
         const Strings &strings = {}, const Lines &lines = {}, const std::vector<cv::Point2d> &points = {});
struct Scope {
    Scope(const std::string &kind, const cv::Mat &image = {}, const Numbers &numbers = {},
          const Strings &strings = {}, const Lines &lines = {}, const std::vector<cv::Point2d> &points = {});
    ~Scope();
};
void recordDecode(const ZXing::BitMatrix &bits, const ZXing::DecoderResult &decoded, double dx = 0, double dy = 0);
} // namespace datamatrix::diagnostics
#define DM_TRACE(...) datamatrix::diagnostics::emit(__VA_ARGS__)
#define DM_SCOPE(name, ...) datamatrix::diagnostics::Scope name(__VA_ARGS__)
#else
#define DM_TRACE(...) ((void)0)
#define DM_SCOPE(name, ...) ((void)0)
#endif
