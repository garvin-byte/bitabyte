#pragma once

#include <optional>

#include <QByteArray>
#include <QString>

namespace bitabyte::core {

[[nodiscard]] QString formatByteOffset(qsizetype byteOffset);
[[nodiscard]] QString formatByteHeaderLabel(int col);
[[nodiscard]] QString formatByteValue(unsigned char byteValue);
[[nodiscard]] QString formatBitsAsHexWithTrailingBits(
    const QByteArray& bits,
    bool includeHexPrefix = false
);
[[nodiscard]] QString formatPrintableAscii(unsigned char byteValue);
[[nodiscard]] std::optional<qsizetype> parseByteOffsetText(const QString& offsetText);

}  // namespace bitabyte::core
