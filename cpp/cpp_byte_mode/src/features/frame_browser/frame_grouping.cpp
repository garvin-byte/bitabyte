#include "features/frame_browser/frame_grouping.h"

#include "core/byte_format_utils.h"

#include <QStringView>
#include <QtGlobal>

namespace bitabyte::features::frame_browser {
namespace {

quint64 bitsToUnsignedValue(const QByteArray& rawBits) {
    quint64 numericValue = 0;
    for (char bitValue : rawBits) {
        numericValue = (numericValue << 1) | (bitValue != 0 ? quint64{1} : quint64{0});
    }
    return numericValue;
}

QByteArray packedBytesForBits(const QByteArray& rawBits) {
    const int byteCount = (rawBits.size() + 7) / 8;
    QByteArray packedBytes(byteCount, '\0');
    for (int bitIndex = 0; bitIndex < rawBits.size(); ++bitIndex) {
        if (rawBits.at(bitIndex) == 0) {
            continue;
        }
        const int byteIndex = bitIndex / 8;
        const int bitOffset = bitIndex % 8;
        packedBytes[byteIndex] = static_cast<char>(
            static_cast<unsigned char>(packedBytes.at(byteIndex)) | static_cast<unsigned char>(1u << (7 - bitOffset))
        );
    }
    return packedBytes;
}

QString formatBitsForDisplay(const QByteArray& rawBits, QStringView displayFormat) {
    if (rawBits.isEmpty()) {
        return QStringLiteral("(missing)");
    }

    if (displayFormat == QStringLiteral("binary")) {
        QString binaryText;
        binaryText.reserve(rawBits.size());
        for (char bitValue : rawBits) {
            binaryText.append(bitValue != 0 ? QLatin1Char('1') : QLatin1Char('0'));
        }
        return binaryText;
    }

    if (displayFormat == QStringLiteral("decimal")) {
        if (rawBits.size() > 64) {
            return core::formatBitsAsHexWithTrailingBits(rawBits, true);
        }
        return QString::number(bitsToUnsignedValue(rawBits));
    }

    if (displayFormat == QStringLiteral("ascii") && rawBits.size() % 8 == 0) {
        const QByteArray packedBytes = packedBytesForBits(rawBits);
        QString asciiText;
        asciiText.reserve(packedBytes.size());
        for (char packedByte : packedBytes) {
            asciiText.append(core::formatPrintableAscii(static_cast<unsigned char>(packedByte)));
        }
        return asciiText;
    }

    if (displayFormat == QStringLiteral("hex") || displayFormat.isEmpty()) {
        return core::formatBitsAsHexWithTrailingBits(rawBits, true);
    }

    return core::formatBitsAsHexWithTrailingBits(rawBits, true);
}

FrameGroupValue frameLengthValue(const framing::FrameSpan& frameSpan) {
    FrameGroupValue value;
    const qsizetype lengthBits = frameSpan.lengthBits;
    const int lengthBytes = static_cast<int>((lengthBits + 7) / 8);
    value.displayText = QStringLiteral("%1 bytes").arg(lengthBytes);
    value.numericValue = static_cast<quint64>(qMax(0, lengthBytes));
    value.hasNumericValue = true;
    value.rawBits = QByteArray::number(lengthBytes);
    return value;
}

FrameGroupValue fieldValue(
    const data::ByteDataSource& dataSource,
    const framing::FrameSpan& frameSpan,
    const FrameGroupingKey& key
) {
    FrameGroupValue value;
    if (key.bitWidth <= 0 || key.startBit < 0) {
        value.missing = true;
        value.displayText = QStringLiteral("(missing)");
        return value;
    }

    const qsizetype endBit = static_cast<qsizetype>(key.startBit) + static_cast<qsizetype>(key.bitWidth) - 1;
    if (endBit >= frameSpan.lengthBits) {
        value.missing = true;
        value.displayText = QStringLiteral("(missing)");
        return value;
    }

    value.rawBits = dataSource.bitRange(frameSpan.startBit + key.startBit, key.bitWidth);
    const QString normalizedDisplayFormat = key.displayFormat.trimmed().toLower();
    value.displayText = formatBitsForDisplay(value.rawBits, normalizedDisplayFormat);
    if (value.rawBits.size() <= 64
        && (normalizedDisplayFormat == QStringLiteral("hex")
            || normalizedDisplayFormat == QStringLiteral("decimal")
            || normalizedDisplayFormat == QStringLiteral("binary")
            || normalizedDisplayFormat.isEmpty())) {
        value.numericValue = bitsToUnsignedValue(value.rawBits);
        value.hasNumericValue = true;
    }
    return value;
}

QString scopeKeyComponent(const FrameGroupingKey& groupingKey, const FrameGroupValue& groupValue) {
    QString rawBitsText;
    rawBitsText.reserve(groupValue.rawBits.size());
    for (char bitValue : groupValue.rawBits) {
        rawBitsText.append(bitValue != 0 ? QLatin1Char('1') : QLatin1Char('0'));
    }

    const QString valueToken = groupValue.missing
        ? QStringLiteral("missing")
        : (rawBitsText.isEmpty() ? groupValue.displayText : rawBitsText);
    return QStringLiteral("%1:%2:%3=%4")
        .arg(static_cast<int>(groupingKey.kind))
        .arg(groupingKey.startBit)
        .arg(groupingKey.bitWidth)
        .arg(valueToken);
}

}  // namespace

FrameGroupValue evaluateFrameGroupValue(
    const data::ByteDataSource& dataSource,
    const framing::FrameSpan& frameSpan,
    const FrameGroupingKey& key
) {
    switch (key.kind) {
    case FrameGroupingKeyKind::FrameLength:
        return frameLengthValue(frameSpan);
    case FrameGroupingKeyKind::FieldValue:
    default:
        return fieldValue(dataSource, frameSpan, key);
    }
}

int compareFrameGroupValues(const FrameGroupValue& leftValue, const FrameGroupValue& rightValue) {
    if (leftValue.missing != rightValue.missing) {
        return leftValue.missing ? 1 : -1;
    }

    if (leftValue.hasNumericValue && rightValue.hasNumericValue && leftValue.numericValue != rightValue.numericValue) {
        return leftValue.numericValue < rightValue.numericValue ? -1 : 1;
    }

    if (leftValue.rawBits != rightValue.rawBits) {
        if (leftValue.rawBits.size() != rightValue.rawBits.size()) {
            return leftValue.rawBits.size() < rightValue.rawBits.size() ? -1 : 1;
        }
        return leftValue.rawBits < rightValue.rawBits ? -1 : 1;
    }

    return QString::compare(leftValue.displayText, rightValue.displayText, Qt::CaseInsensitive);
}

bool frameMatchesFilterClause(
    const data::ByteDataSource& dataSource,
    const framing::FrameSpan& frameSpan,
    const FrameFilterClause& filterClause
) {
    return evaluateFrameGroupValue(dataSource, frameSpan, filterClause.key) == filterClause.value;
}

QString appendScopeKey(
    const QString& parentScopeKey,
    const FrameGroupingKey& groupingKey,
    const FrameGroupValue& groupValue
) {
    const QString component = scopeKeyComponent(groupingKey, groupValue);
    return parentScopeKey.isEmpty()
        ? component
        : QStringLiteral("%1/%2").arg(parentScopeKey, component);
}

QString parentScopeKey(const QString& scopeKey) {
    const int separatorIndex = scopeKey.lastIndexOf(QLatin1Char('/'));
    if (separatorIndex < 0) {
        return {};
    }
    return scopeKey.left(separatorIndex);
}

}  // namespace bitabyte::features::frame_browser
