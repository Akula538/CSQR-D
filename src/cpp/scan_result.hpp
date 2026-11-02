#pragma once

#include <string>

namespace qrscanner {

enum class ScanStatus { Decoded, Ctf, None };

struct ScanResult {
    ScanStatus status = ScanStatus::None;
    std::string text;
};

} // namespace qrscanner
