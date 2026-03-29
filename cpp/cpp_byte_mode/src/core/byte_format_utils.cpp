#include "core/byte_format_utils.h"

namespace bitabyte::core {

QString formatByteOffset(qsizetype byteOffset) {
    return QStringLiteral("%1").arg(byteOffset, 8, 16, QLatin1Char('0')).toUpper();
}

QString formatByteHeaderLabel(int col) {
    return QStringLiteral("%1").arg(col, 2, 16, QLatin1Char('0')).toUpper();
}

QString formatByteValue(unsigned char byteValue) {
    return QStringLiteral("%1")
        .arg(static_cast<int>(byteValue), 2, 16, QLatin1Char('0'))
        .toUpper();
}

QString formatBitsAsHexWithTrailingBits(const QByteArray& bits, bool includeHexPrefix) {
    if (bits.isEmpty()) {
        return {};
    }

    const int trailingBitCount = bits.size() % 4;
    const int hexBitCount = bits.size() - trailingBitCount;
    if (hexBitCount <= 0) {
        QString trailingBitsText;
        trailingBitsText.reserve(bits.size());
        for (char bitValue : bits) {
            trailingBitsText.append(bitValue != 0 ? QLatin1Char('1') : QLatin1Char('0'));
        }
        return trailingBitsText;
    }

    QString hexText;
    hexText.reserve((hexBitCount / 4) + (includeHexPrefix ? 2 : 0));
    if (includeHexPrefix) {
        hexText.append(QStringLiteral("0x"));
    }

    static constexpr char kHexDigits[] = "0123456789ABCDEF";
    for (int bitIndex = 0; bitIndex < hexBitCount; bitIndex += 4) {
        int nibbleValue = 0;
        for (int nibbleOffset = 0; nibbleOffset < 4; ++nibbleOffset) {
            nibbleValue = (nibbleValue << 1) | (bits.at(bitIndex + nibbleOffset) != 0 ? 1 : 0);
        }
        hexText.append(QLatin1Char(kHexDigits[nibbleValue & 0x0F]));
    }
    if (trailingBitCount <= 0) {
        return hexText;
    }

    QString trailingBitsText;
    trailingBitsText.reserve(trailingBitCount);
    for (int bitIndex = hexBitCount; bitIndex < bits.size(); ++bitIndex) {
        trailingBitsText.append(bits.at(bitIndex) != 0 ? QLatin1Char('1') : QLatin1Char('0'));
    }
    return QStringLiteral("%1 %2").arg(hexText, trailingBitsText);
}

QString formatPrintableAscii(unsigned char byteValue) {
    if (byteValue >= 32 && byteValue <= 126) {
        return QString(QChar::fromLatin1(static_cast<char>(byteValue)));
    }

    return QStringLiteral(".");
}

std::optional<qsizetype> parseByteOffsetText(const QString& offsetText) {
    const QString trimmedText = offsetText.trimmed();
    if (trimmedText.isEmpty()) {
        return std::nullopt;
    }

    bool converted = false;
    qlonglong parsedOffset = 0;

    if (trimmedText.startsWith(QStringLiteral("0x"), Qt::CaseInsensitive)) {
        parsedOffset = trimmedText.mid(2).toLongLong(&converted, 16);
    } else if (trimmedText.endsWith(QStringLiteral("h"), Qt::CaseInsensitive)) {
        parsedOffset = trimmedText.left(trimmedText.size() - 1).toLongLong(&converted, 16);
    } else {
        parsedOffset = trimmedText.toLongLong(&converted, 10);
    }

    if (!converted || parsedOffset < 0) {
        return std::nullopt;
    }

    return static_cast<qsizetype>(parsedOffset);
}

}  // namespace bitabyte::core
