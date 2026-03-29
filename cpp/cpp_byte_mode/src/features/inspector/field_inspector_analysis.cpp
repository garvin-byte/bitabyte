#include "features/inspector/field_inspector_analysis.h"

#include "core/byte_format_utils.h"
#include "data/byte_data_source.h"
#include "features/framing/frame_layout.h"

#include <QByteArray>
#include <QHash>
#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>

namespace bitabyte::features::inspector {
namespace {

QString formatBitsAsBinary(const QByteArray& bitValues) {
    QString binaryText;
    binaryText.reserve(bitValues.size());
    for (char bitValue : bitValues) {
        binaryText.append(bitValue != 0 ? QLatin1Char('1') : QLatin1Char('0'));
    }
    return binaryText;
}

QString formatBitsAsHexWithTrailingBits(const QByteArray& bitValues) {
    if (bitValues.isEmpty()) {
        return QStringLiteral("-");
    }

    quint64 numericValue = 0;
    for (char bitValue : bitValues) {
        numericValue = (numericValue << 1) | (bitValue != 0 ? 1 : 0);
    }

    const int trailingBitCount = bitValues.size() % 4;
    const int hexBitCount = bitValues.size() - trailingBitCount;
    if (hexBitCount <= 0) {
        return formatBitsAsBinary(bitValues);
    }

    const quint64 hexValue = trailingBitCount > 0 ? (numericValue >> trailingBitCount) : numericValue;
    const int hexWidth = qMax(1, hexBitCount / 4);
    const QString hexText = QStringLiteral("0x%1").arg(hexValue, hexWidth, 16, QLatin1Char('0')).toUpper();
    if (trailingBitCount <= 0) {
        return hexText;
    }

    QString trailingBitsText;
    trailingBitsText.reserve(trailingBitCount);
    const quint64 trailingMask = (quint64{1} << trailingBitCount) - 1;
    const quint64 trailingBitsValue = numericValue & trailingMask;
    for (int bitOffset = trailingBitCount - 1; bitOffset >= 0; --bitOffset) {
        trailingBitsText.append(((trailingBitsValue >> bitOffset) & quint64{1}) != 0 ? QLatin1Char('1') : QLatin1Char('0'));
    }
    return QStringLiteral("%1 %2").arg(hexText, trailingBitsText);
}

QByteArray packBitsToBytes(const QByteArray& bitValues) {
    QByteArray packedBytes;
    if (bitValues.isEmpty() || (bitValues.size() % 8) != 0) {
        return packedBytes;
    }

    packedBytes.reserve(bitValues.size() / 8);
    for (int byteOffset = 0; byteOffset < bitValues.size(); byteOffset += 8) {
        unsigned char byteValue = 0;
        for (int bitOffset = 0; bitOffset < 8; ++bitOffset) {
            byteValue = static_cast<unsigned char>(byteValue << 1);
            if (bitValues.at(byteOffset + bitOffset) != 0) {
                byteValue = static_cast<unsigned char>(byteValue | 0x01);
            }
        }
        packedBytes.append(static_cast<char>(byteValue));
    }

    return packedBytes;
}

QString formatAsciiText(const QByteArray& packedBytes) {
    if (packedBytes.isEmpty()) {
        return QStringLiteral("-");
    }

    QString asciiText;
    asciiText.reserve(packedBytes.size());
    for (unsigned char byteValue : packedBytes) {
        asciiText.append(core::formatPrintableAscii(byteValue));
    }
    return asciiText;
}

std::optional<quint64> unsignedBigEndianValue(const QByteArray& bitValues) {
    if (bitValues.isEmpty() || bitValues.size() > 64) {
        return std::nullopt;
    }

    quint64 numericValue = 0;
    for (char bitValue : bitValues) {
        numericValue = (numericValue << 1) | (bitValue != 0 ? 1 : 0);
    }
    return numericValue;
}

std::optional<quint64> unsignedLittleEndianValue(const QByteArray& packedBytes) {
    if (packedBytes.isEmpty() || packedBytes.size() > 8) {
        return std::nullopt;
    }

    quint64 numericValue = 0;
    for (int byteIndex = packedBytes.size() - 1; byteIndex >= 0; --byteIndex) {
        numericValue = (numericValue << 8) | static_cast<unsigned char>(packedBytes.at(byteIndex));
    }
    return numericValue;
}

std::optional<qint64> signedValue(quint64 unsignedValue, int bitWidth) {
    if (bitWidth <= 0 || bitWidth > 64) {
        return std::nullopt;
    }

    if (bitWidth == 64) {
        return static_cast<qint64>(unsignedValue);
    }

    const quint64 signMask = quint64{1} << (bitWidth - 1);
    if ((unsignedValue & signMask) == 0) {
        return static_cast<qint64>(unsignedValue);
    }

    const quint64 extensionMask = (~quint64{0}) << bitWidth;
    return static_cast<qint64>(unsignedValue | extensionMask);
}

QString formatFloatValue(const QByteArray& packedBytes, bool littleEndian) {
    if (packedBytes.size() != 4 && packedBytes.size() != 8) {
        return QStringLiteral("-");
    }

    QByteArray orderedBytes = packedBytes;
    if (littleEndian) {
        std::reverse(orderedBytes.begin(), orderedBytes.end());
    }

    if (orderedBytes.size() == 4) {
        quint32 rawValue = 0;
        for (unsigned char byteValue : orderedBytes) {
            rawValue = (rawValue << 8) | byteValue;
        }

        float floatValue = 0.0f;
        static_assert(sizeof(floatValue) == sizeof(rawValue));
        std::memcpy(&floatValue, &rawValue, sizeof(floatValue));
        return std::isfinite(floatValue)
            ? QString::number(static_cast<double>(floatValue), 'g', 9)
            : QStringLiteral("-");
    }

    quint64 rawValue = 0;
    for (unsigned char byteValue : orderedBytes) {
        rawValue = (rawValue << 8) | byteValue;
    }

    double doubleValue = 0.0;
    static_assert(sizeof(doubleValue) == sizeof(rawValue));
    std::memcpy(&doubleValue, &rawValue, sizeof(doubleValue));
    return std::isfinite(doubleValue)
        ? QString::number(doubleValue, 'g', 15)
        : QStringLiteral("-");
}

QString formatUnsignedValue(const std::optional<quint64>& numericValue) {
    return numericValue.has_value() ? QString::number(numericValue.value()) : QStringLiteral("-");
}

QString formatSignedValue(const std::optional<qint64>& numericValue) {
    return numericValue.has_value() ? QString::number(numericValue.value()) : QStringLiteral("-");
}

QString formatMeanValue(long double meanValue) {
    const long double roundedValue = std::round(meanValue);
    if (std::fabs(static_cast<double>(meanValue - roundedValue)) < 0.0005) {
        return QString::number(static_cast<qlonglong>(roundedValue));
    }
    return QString::number(static_cast<double>(meanValue), 'f', 3);
}

double calculateEntropy(const QHash<QByteArray, int>& countsByValue, int sampleCount) {
    if (sampleCount <= 0) {
        return 0.0;
    }

    double entropy = 0.0;
    for (auto countIterator = countsByValue.cbegin(); countIterator != countsByValue.cend(); ++countIterator) {
        const double probability = static_cast<double>(countIterator.value()) / static_cast<double>(sampleCount);
        if (probability <= 0.0) {
            continue;
        }
        entropy -= probability * std::log2(probability);
    }

    return entropy;
}

QByteArray extractRowBits(
    const data::ByteDataSource& dataSource,
    const features::framing::FrameLayout& frameLayout,
    int row,
    const FieldSelection& fieldSelection
) {
    QByteArray rowBits;
    const int bitWidth = fieldSelection.endBit - fieldSelection.startBit + 1;
    if (bitWidth <= 0) {
        return rowBits;
    }

    const qsizetype rowLengthBits = frameLayout.rowLengthBits(dataSource, row);
    if (fieldSelection.startBit < 0 || fieldSelection.endBit >= rowLengthBits) {
        return rowBits;
    }

    const qsizetype absoluteStartBit = frameLayout.rowStartBit(dataSource, row) + fieldSelection.startBit;
    return dataSource.bitRange(absoluteStartBit, bitWidth);
}

QString formatBitRangeText(int startBit, int endBit) {
    if (startBit > endBit) {
        return QStringLiteral("-");
    }

    const int startByteIndex = startBit / 8;
    const int endByteIndex = endBit / 8;
    const QString byteRangeText = startByteIndex == endByteIndex
        ? QStringLiteral("byte %1").arg(startByteIndex)
        : QStringLiteral("bytes %1-%2").arg(startByteIndex).arg(endByteIndex);
    return QStringLiteral("bits %1-%2 | %3").arg(startBit).arg(endBit).arg(byteRangeText);
}

}  // namespace

FieldInspectorAnalysis analyzeField(
    const data::ByteDataSource& dataSource,
    const features::framing::FrameLayout& frameLayout,
    const FieldSelection& fieldSelection,
    int currentRow
) {
    FieldInspectorAnalysis analysis;
    const int bitWidth = fieldSelection.endBit - fieldSelection.startBit + 1;
    if (!dataSource.hasData() || bitWidth <= 0) {
        return analysis;
    }

    analysis.hasField = true;
    analysis.fieldLabel = fieldSelection.label.trimmed().isEmpty() ? QStringLiteral("(unnamed field)") : fieldSelection.label.trimmed();
    analysis.positionText = formatBitRangeText(fieldSelection.startBit, fieldSelection.endBit);
    analysis.bitWidth = bitWidth;

    const int totalRowCount = static_cast<int>(frameLayout.rowCount(dataSource));
    const QByteArray currentRowBits = currentRow >= 0 && currentRow < totalRowCount
        ? extractRowBits(dataSource, frameLayout, currentRow, fieldSelection)
        : QByteArray();
    const QByteArray currentRowBytes = packBitsToBytes(currentRowBits);

    analysis.currentRowText = currentRowBits.isEmpty()
        ? QStringLiteral("-")
        : QStringLiteral("row %1").arg(currentRow);
    analysis.currentHexValue = currentRowBits.isEmpty() ? QStringLiteral("-") : formatBitsAsHexWithTrailingBits(currentRowBits);
    analysis.currentBinaryValue = currentRowBits.isEmpty() ? QStringLiteral("-") : formatBitsAsBinary(currentRowBits);
    analysis.currentAsciiValue = formatAsciiText(currentRowBytes);

    const std::optional<quint64> currentUnsignedBigEndianValue = unsignedBigEndianValue(currentRowBits);
    const std::optional<quint64> currentUnsignedLittleEndianValue = unsignedLittleEndianValue(currentRowBytes);
    analysis.currentUnsignedBigEndianValue = formatUnsignedValue(currentUnsignedBigEndianValue);
    analysis.currentUnsignedLittleEndianValue = formatUnsignedValue(currentUnsignedLittleEndianValue);
    analysis.currentSignedBigEndianValue = formatSignedValue(
        currentUnsignedBigEndianValue.has_value() ? signedValue(currentUnsignedBigEndianValue.value(), bitWidth) : std::nullopt
    );
    analysis.currentSignedLittleEndianValue = formatSignedValue(
        (currentUnsignedLittleEndianValue.has_value() && (bitWidth % 8) == 0)
            ? signedValue(currentUnsignedLittleEndianValue.value(), bitWidth)
            : std::nullopt
    );
    analysis.currentFloatBigEndianValue = formatFloatValue(currentRowBytes, false);
    analysis.currentFloatLittleEndianValue = formatFloatValue(currentRowBytes, true);

    QHash<QByteArray, int> countsByValue;
    QList<quint64> numericValues;
    numericValues.reserve(totalRowCount);

    bool monotonicIncreasing = true;
    bool havePreviousNumericValue = false;
    quint64 previousNumericValue = 0;
    long double numericSum = 0.0L;
    quint64 minimumNumericValue = std::numeric_limits<quint64>::max();
    quint64 maximumNumericValue = 0;
    QByteArray modeBits;
    int modeCount = 0;

    for (int row = 0; row < totalRowCount; ++row) {
        const QByteArray rowBits = extractRowBits(dataSource, frameLayout, row, fieldSelection);
        if (rowBits.isEmpty()) {
            ++analysis.missingFrameCount;
            continue;
        }

        ++analysis.analyzedFrameCount;
        const int updatedCount = countsByValue.value(rowBits, 0) + 1;
        countsByValue.insert(rowBits, updatedCount);
        if (updatedCount > modeCount) {
            modeCount = updatedCount;
            modeBits = rowBits;
        }

        const std::optional<quint64> numericValue = unsignedBigEndianValue(rowBits);
        if (!numericValue.has_value()) {
            monotonicIncreasing = false;
            continue;
        }

        numericValues.append(numericValue.value());
        numericSum += static_cast<long double>(numericValue.value());
        minimumNumericValue = qMin(minimumNumericValue, numericValue.value());
        maximumNumericValue = qMax(maximumNumericValue, numericValue.value());
        if (havePreviousNumericValue && numericValue.value() < previousNumericValue) {
            monotonicIncreasing = false;
        }
        previousNumericValue = numericValue.value();
        havePreviousNumericValue = true;
    }

    analysis.uniqueValueCount = countsByValue.size();
    analysis.isConstant = analysis.uniqueValueCount == 1 && analysis.analyzedFrameCount > 0;
    analysis.isMonotonicIncreasing = monotonicIncreasing && numericValues.size() >= 2;

    if (!numericValues.isEmpty()) {
        analysis.minValueText = QString::number(minimumNumericValue);
        analysis.maxValueText = QString::number(maximumNumericValue);
        analysis.meanValueText = formatMeanValue(numericSum / static_cast<long double>(numericValues.size()));
    } else {
        analysis.minValueText = QStringLiteral("-");
        analysis.maxValueText = QStringLiteral("-");
        analysis.meanValueText = QStringLiteral("-");
    }

    analysis.modeValueText = modeCount > 0
        ? QStringLiteral("%1 (%2)").arg(formatBitsAsHexWithTrailingBits(modeBits)).arg(modeCount)
        : QStringLiteral("-");

    const double entropyBits = calculateEntropy(countsByValue, analysis.analyzedFrameCount);
    const double sampleEntropyCeiling = analysis.analyzedFrameCount > 0
        ? std::min<double>(bitWidth, std::log2(static_cast<double>(analysis.analyzedFrameCount)))
        : 0.0;
    analysis.entropyText = analysis.analyzedFrameCount > 0
        ? QStringLiteral("%1 / %2")
            .arg(QString::number(entropyBits, 'f', 3))
            .arg(QString::number(sampleEntropyCeiling, 'f', 3))
        : QStringLiteral("-");
    analysis.isHighEntropy = analysis.analyzedFrameCount >= 8
        && sampleEntropyCeiling > 0.0
        && entropyBits >= (sampleEntropyCeiling * 0.85);

    struct HistogramEntry {
        QByteArray bits;
        int count = 0;
    };

    QVector<HistogramEntry> histogramEntries;
    histogramEntries.reserve(countsByValue.size());
    for (auto countIterator = countsByValue.cbegin(); countIterator != countsByValue.cend(); ++countIterator) {
        HistogramEntry histogramEntry;
        histogramEntry.bits = countIterator.key();
        histogramEntry.count = countIterator.value();
        histogramEntries.append(histogramEntry);
    }

    std::sort(
        histogramEntries.begin(),
        histogramEntries.end(),
        [](const HistogramEntry& leftEntry, const HistogramEntry& rightEntry) {
            if (leftEntry.count != rightEntry.count) {
                return leftEntry.count > rightEntry.count;
            }
            return formatBitsAsHexWithTrailingBits(leftEntry.bits) < formatBitsAsHexWithTrailingBits(rightEntry.bits);
        }
    );

    constexpr int kMaximumHistogramBins = 12;
    const int histogramBinCount = qMin(kMaximumHistogramBins, histogramEntries.size());
    analysis.histogramBins.reserve(histogramBinCount);
    for (int index = 0; index < histogramBinCount; ++index) {
        HistogramBin histogramBin;
        histogramBin.label = formatBitsAsHexWithTrailingBits(histogramEntries.at(index).bits);
        histogramBin.count = histogramEntries.at(index).count;
        histogramBin.fraction = analysis.analyzedFrameCount > 0
            ? static_cast<double>(histogramBin.count) / static_cast<double>(analysis.analyzedFrameCount)
            : 0.0;
        analysis.histogramBins.append(histogramBin);
    }

    return analysis;
}

}  // namespace bitabyte::features::inspector
