#pragma once

#include <QString>

namespace bitabyte::features::columns {

struct VisibleByteColumn {
    int byteIndex = 0;
    int bitStart = 0;
    int bitEnd = 7;
    int byteEndIndex = 0;
    int absoluteStartBit = 0;
    int absoluteEndBit = 7;
    int definitionIndex = -1;
    bool isUndefined = false;
    QString splitLabel;
    QString splitColorName = QStringLiteral("None");
    QString splitDisplayFormat;

    [[nodiscard]] int bitWidth() const {
        return absoluteEndBit - absoluteStartBit + 1;
    }

    [[nodiscard]] bool overlapsByte(int targetByteIndex) const {
        return targetByteIndex >= byteIndex && targetByteIndex <= byteEndIndex;
    }
};

}  // namespace bitabyte::features::columns
