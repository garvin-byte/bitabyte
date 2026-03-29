#pragma once

#include <QString>
#include <QtGlobal>

namespace bitabyte::features::columns {

struct ByteColumnDefinition {
    int startByte = 0;
    int endByte = 0;
    QString label;
    QString displayFormat = QStringLiteral("hex");
    QString colorName = QStringLiteral("None");
    QString unit = QStringLiteral("byte");
    int startBit = 0;
    int totalBits = 8;

    [[nodiscard]] int byteCount() const {
        if (unit == QStringLiteral("bit")) {
            return totalBits > 0 ? ((totalBits + 7) / 8) : 0;
        }
        return endByte >= startByte ? (endByte - startByte + 1) : 0;
    }

    [[nodiscard]] bool coversDisplayByte(int displayByte) const {
        if (unit == QStringLiteral("bit")) {
            const int endAbsoluteBit = startBit + qMax(1, totalBits) - 1;
            const int firstByte = startBit / 8;
            const int lastByte = endAbsoluteBit / 8;
            return displayByte >= firstByte && displayByte <= lastByte;
        }
        return displayByte >= startByte && displayByte <= endByte;
    }

    [[nodiscard]] int startAbsoluteBit() const {
        if (unit == QStringLiteral("bit")) {
            return startBit;
        }
        return startByte * 8;
    }

    [[nodiscard]] int endAbsoluteBit() const {
        if (unit == QStringLiteral("bit")) {
            return startBit + qMax(1, totalBits) - 1;
        }
        return ((endByte + 1) * 8) - 1;
    }
};

}  // namespace bitabyte::features::columns
