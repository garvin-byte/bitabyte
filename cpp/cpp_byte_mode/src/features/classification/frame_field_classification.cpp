#include "features/classification/frame_field_classification.h"

#include "core/byte_format_utils.h"
#include "data/byte_data_source.h"
#include "features/framing/frame_layout.h"

#include <QByteArray>
#include <QHash>
#include <QStringList>
#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <optional>

namespace bitabyte::features::classification {
namespace {

struct CounterAnalysisResult {
    bool likelyCounter = false;
    quint64 dominantStep = 0;
    int dominantStepCount = 0;
    int comparableTransitionCount = 0;
    int wrapCount = 0;
    double confidence = 0.0;
};

struct ColumnObservation {
    QByteArray referenceBits;
    bool hasReferenceValue = false;
    bool allRowsMatch = true;
    int analyzedFrameCount = 0;
    QList<quint64> numericValues;
    bool numericCompatible = true;
    QHash<QByteArray, int> countsByValue;
};

struct CounterEvaluation {
    CounterAnalysisResult analysis;
    bool byteSwapped = false;
    bool bitReversed = false;
    bool descending = false;
};

struct RowSpan {
    qsizetype startBit = -1;
    qsizetype lengthBits = 0;
};

CounterAnalysisResult analyzeCounterPattern(
    const QList<quint64>& numericValues,
    int uniqueValueCount,
    int analyzedFrameCount,
    int bitWidth
);

QString normalizedDisplayFormat(const QString& displayFormat) {
    const QString loweredFormat = displayFormat.trimmed().toLower();
    if (loweredFormat == QStringLiteral("dec")) {
        return QStringLiteral("decimal");
    }
    return loweredFormat.isEmpty() ? QStringLiteral("hex") : loweredFormat;
}

QString formatBitsAsBinary(const QByteArray& bitValues) {
    QString binaryText;
    binaryText.reserve(bitValues.size());
    for (char bitValue : bitValues) {
        binaryText.append(bitValue != 0 ? QLatin1Char('1') : QLatin1Char('0'));
    }
    return binaryText;
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

std::optional<quint64> unsignedValue(const QByteArray& bitValues) {
    if (bitValues.isEmpty() || bitValues.size() > 63) {
        return std::nullopt;
    }

    quint64 numericValue = 0;
    for (char bitValue : bitValues) {
        numericValue = (numericValue << 1) | (bitValue != 0 ? 1 : 0);
    }
    return numericValue;
}

QString formatDisplayValue(const QByteArray& bitValues, const QString& displayFormat) {
    if (bitValues.isEmpty()) {
        return QStringLiteral("-");
    }

    const QString normalizedFormat = normalizedDisplayFormat(displayFormat);
    if (normalizedFormat == QStringLiteral("binary")) {
        return formatBitsAsBinary(bitValues);
    }

    if (normalizedFormat == QStringLiteral("decimal")) {
        const std::optional<quint64> numericValue = unsignedValue(bitValues);
        return numericValue.has_value() ? QString::number(numericValue.value()) : QStringLiteral("-");
    }

    if (normalizedFormat == QStringLiteral("ascii")) {
        const QByteArray packedBytes = packBitsToBytes(bitValues);
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

    return core::formatBitsAsHexWithTrailingBits(bitValues, true);
}

QByteArray extractRowBits(
    const data::ByteDataSource& dataSource,
    const RowSpan& rowSpan,
    const FrameFieldColumnSnapshot& columnSnapshot
) {
    const features::columns::VisibleByteColumn& visibleColumn = columnSnapshot.visibleColumn;
    if (visibleColumn.absoluteStartBit < 0 || visibleColumn.absoluteEndBit < visibleColumn.absoluteStartBit) {
        return {};
    }

    if (rowSpan.startBit < 0 || rowSpan.lengthBits <= 0 || visibleColumn.absoluteEndBit >= rowSpan.lengthBits) {
        return {};
    }

    const qsizetype absoluteStartBit = rowSpan.startBit + visibleColumn.absoluteStartBit;
    return dataSource.bitRange(absoluteStartBit, visibleColumn.bitWidth());
}

QVector<RowSpan> buildRowSpans(
    const data::ByteDataSource& dataSource,
    const features::framing::FrameLayout& frameLayout,
    int totalRowCount
) {
    QVector<RowSpan> rowSpans;
    rowSpans.reserve(totalRowCount);
    for (int row = 0; row < totalRowCount; ++row) {
        RowSpan rowSpan;
        rowSpan.startBit = frameLayout.rowStartBit(dataSource, row);
        rowSpan.lengthBits = frameLayout.rowLengthBits(dataSource, row);
        rowSpans.append(rowSpan);
    }
    return rowSpans;
}

QString formatCounterStepValue(quint64 stepValue, int bitWidth) {
    Q_UNUSED(bitWidth);
    return QString::number(stepValue);
}

QString formatRangeLabel(int startBit, int endBit) {
    if (startBit < 0 || endBit < startBit) {
        return QStringLiteral("-");
    }

    if ((startBit % 8) == 0 && (endBit % 8) == 7) {
        const int startByte = startBit / 8;
        const int endByte = endBit / 8;
        if (startByte == endByte) {
            return QStringLiteral("Byte %1").arg(startByte);
        }
        return QStringLiteral("Bytes %1-%2").arg(startByte).arg(endByte);
    }

    if (startBit == endBit) {
        return QStringLiteral("Bit %1").arg(startBit);
    }
    return QStringLiteral("Bits %1-%2").arg(startBit).arg(endBit);
}

ColumnObservation observeColumnSnapshot(
    const data::ByteDataSource& dataSource,
    const QVector<RowSpan>& rowSpans,
    const FrameFieldColumnSnapshot& columnSnapshot,
    int totalRowCount
) {
    ColumnObservation observation;
    observation.numericCompatible = columnSnapshot.visibleColumn.bitWidth() <= 63;
    observation.numericValues.reserve(totalRowCount);
    observation.countsByValue.reserve(totalRowCount);

    for (int row = 0; row < totalRowCount; ++row) {
        const QByteArray rowBits = extractRowBits(dataSource, rowSpans.at(row), columnSnapshot);
        if (rowBits.isEmpty()) {
            continue;
        }

        ++observation.analyzedFrameCount;
        observation.countsByValue.insert(rowBits, observation.countsByValue.value(rowBits, 0) + 1);
        if (!observation.hasReferenceValue) {
            observation.referenceBits = rowBits;
            observation.hasReferenceValue = true;
        } else if (rowBits != observation.referenceBits) {
            observation.allRowsMatch = false;
        }

        if (observation.numericCompatible) {
            const std::optional<quint64> numericValue = unsignedValue(rowBits);
            if (numericValue.has_value()) {
                observation.numericValues.append(numericValue.value());
            } else {
                observation.numericCompatible = false;
            }
        }
    }

    return observation;
}

quint64 reverseBitsWithinWidth(quint64 numericValue, int bitWidth) {
    quint64 reversedValue = 0;
    for (int bitIndex = 0; bitIndex < bitWidth; ++bitIndex) {
        reversedValue = (reversedValue << 1) | (numericValue & quint64{1});
        numericValue >>= 1;
    }
    return reversedValue;
}

std::optional<quint64> byteSwappedUnsignedValue(quint64 numericValue, int bitWidth) {
    if (bitWidth <= 0 || bitWidth > 64 || (bitWidth % 8) != 0) {
        return std::nullopt;
    }

    quint64 swappedValue = 0;
    const int byteCount = bitWidth / 8;
    for (int byteIndex = 0; byteIndex < byteCount; ++byteIndex) {
        swappedValue = (swappedValue << 8) | (numericValue & quint64{0xFF});
        numericValue >>= 8;
    }
    return swappedValue;
}

QList<quint64> descendingCounterValues(const QList<quint64>& numericValues, int bitWidth) {
    QList<quint64> descendingValues;
    descendingValues.reserve(numericValues.size());
    if (bitWidth <= 0 || bitWidth > 63) {
        return descendingValues;
    }

    const quint64 counterMask = (quint64{1} << bitWidth) - 1;
    for (quint64 numericValue : numericValues) {
        descendingValues.append(counterMask - numericValue);
    }
    return descendingValues;
}

QString counterTransformSuffix(const CounterEvaluation& evaluation) {
    QStringList suffixParts;
    if (evaluation.byteSwapped) {
        suffixParts.append(QStringLiteral("LE"));
    }
    if (evaluation.bitReversed) {
        suffixParts.append(QStringLiteral("LSBF"));
    }
    if (evaluation.descending) {
        suffixParts.append(QStringLiteral("Down"));
    }

    return suffixParts.isEmpty()
        ? QString()
        : QStringLiteral(" | %1").arg(suffixParts.join(QStringLiteral(" + ")));
}

QString counterSummaryText(const CounterEvaluation& evaluation, int bitWidth) {
    return QStringLiteral("Step %1 | Confidence %2%3%4")
        .arg(formatCounterStepValue(evaluation.analysis.dominantStep, bitWidth))
        .arg(QString::number(evaluation.analysis.confidence * 100.0, 'f', 0))
        .arg(
            evaluation.analysis.wrapCount > 0
                ? QStringLiteral(" | Wraps %1").arg(evaluation.analysis.wrapCount)
                : QString()
        )
        .arg(counterTransformSuffix(evaluation));
}

CounterEvaluation evaluateCounterObservation(
    const ColumnObservation& observation,
    int bitWidth
) {
    CounterEvaluation evaluation;
    if (!observation.numericCompatible
        || observation.analyzedFrameCount <= 0
        || observation.numericValues.size() != observation.analyzedFrameCount) {
        return evaluation;
    }

    auto considerCounterValues = [&](const QList<quint64>& numericValues,
                                     bool byteSwapped,
                                     bool bitReversed,
                                     bool descending) {
        if (numericValues.size() != observation.analyzedFrameCount) {
            return;
        }

        CounterEvaluation candidateEvaluation;
        candidateEvaluation.analysis = analyzeCounterPattern(
            numericValues,
            observation.countsByValue.size(),
            observation.analyzedFrameCount,
            bitWidth
        );
        candidateEvaluation.byteSwapped = byteSwapped;
        candidateEvaluation.bitReversed = bitReversed;
        candidateEvaluation.descending = descending;

        const bool candidateRanksBetter =
            (!evaluation.analysis.likelyCounter && candidateEvaluation.analysis.likelyCounter)
            || (candidateEvaluation.analysis.likelyCounter == evaluation.analysis.likelyCounter
                && candidateEvaluation.analysis.confidence > evaluation.analysis.confidence)
            || (candidateEvaluation.analysis.likelyCounter == evaluation.analysis.likelyCounter
                && qFuzzyCompare(candidateEvaluation.analysis.confidence, evaluation.analysis.confidence)
                && (candidateEvaluation.byteSwapped != evaluation.byteSwapped
                    ? !candidateEvaluation.byteSwapped
                    : (candidateEvaluation.bitReversed != evaluation.bitReversed
                        ? !candidateEvaluation.bitReversed
                        : !candidateEvaluation.descending && evaluation.descending)));
        if (candidateRanksBetter) {
            evaluation = candidateEvaluation;
        }
    };

    auto considerCounterFamilies = [&](const QList<quint64>& numericValues,
                                       bool byteSwapped,
                                       bool bitReversed) {
        considerCounterValues(numericValues, byteSwapped, bitReversed, false);
        considerCounterValues(descendingCounterValues(numericValues, bitWidth), byteSwapped, bitReversed, true);
    };

    considerCounterFamilies(observation.numericValues, false, false);

    QList<quint64> bitReversedValues;
    bitReversedValues.reserve(observation.numericValues.size());
    for (quint64 numericValue : observation.numericValues) {
        bitReversedValues.append(reverseBitsWithinWidth(numericValue, bitWidth));
    }
    if (bitReversedValues.size() == observation.analyzedFrameCount) {
        considerCounterFamilies(bitReversedValues, false, true);
    }

    if ((bitWidth % 8) == 0 && bitWidth > 8) {
        QList<quint64> byteSwappedValues;
        byteSwappedValues.reserve(observation.numericValues.size());
        QList<quint64> byteSwappedBitReversedValues;
        byteSwappedBitReversedValues.reserve(observation.numericValues.size());
        for (quint64 numericValue : observation.numericValues) {
            const std::optional<quint64> byteSwappedValue = byteSwappedUnsignedValue(numericValue, bitWidth);
            if (!byteSwappedValue.has_value()) {
                byteSwappedValues.clear();
                byteSwappedBitReversedValues.clear();
                break;
            }
            byteSwappedValues.append(byteSwappedValue.value());
            byteSwappedBitReversedValues.append(
                reverseBitsWithinWidth(byteSwappedValue.value(), bitWidth)
            );
        }

        if (byteSwappedValues.size() == observation.analyzedFrameCount) {
            considerCounterFamilies(byteSwappedValues, true, false);
        }
        if (byteSwappedBitReversedValues.size() == observation.analyzedFrameCount) {
            considerCounterFamilies(byteSwappedBitReversedValues, true, true);
        }
    }

    return evaluation;
}

bool canParticipateInCounterWindow(const FrameFieldColumnSnapshot& columnSnapshot) {
    const features::columns::VisibleByteColumn& visibleColumn = columnSnapshot.visibleColumn;
    return visibleColumn.bitWidth() == 8
        && visibleColumn.isUndefined
        && visibleColumn.splitLabel.isEmpty();
}

std::optional<FrameFieldColumnSnapshot> mergedCounterWindowSnapshot(
    const QVector<FrameFieldColumnSnapshot>& columnSnapshots,
    int startIndex,
    int windowColumnCount
) {
    if (startIndex < 0 || windowColumnCount < 2 || startIndex + windowColumnCount > columnSnapshots.size()) {
        return std::nullopt;
    }

    const FrameFieldColumnSnapshot& firstSnapshot = columnSnapshots.at(startIndex);
    if (!canParticipateInCounterWindow(firstSnapshot)) {
        return std::nullopt;
    }

    int endBit = firstSnapshot.visibleColumn.absoluteEndBit;
    features::columns::VisibleByteColumn mergedVisibleColumn = firstSnapshot.visibleColumn;
    for (int columnOffset = 1; columnOffset < windowColumnCount; ++columnOffset) {
        const FrameFieldColumnSnapshot& nextSnapshot = columnSnapshots.at(startIndex + columnOffset);
        if (!canParticipateInCounterWindow(nextSnapshot)
            || nextSnapshot.visibleColumn.absoluteStartBit != endBit + 1) {
            return std::nullopt;
        }

        endBit = nextSnapshot.visibleColumn.absoluteEndBit;
        mergedVisibleColumn.byteEndIndex = nextSnapshot.visibleColumn.byteEndIndex;
        mergedVisibleColumn.absoluteEndBit = nextSnapshot.visibleColumn.absoluteEndBit;
    }

    FrameFieldColumnSnapshot mergedSnapshot;
    mergedSnapshot.visibleColumnIndex = firstSnapshot.visibleColumnIndex;
    mergedSnapshot.visibleColumn = mergedVisibleColumn;
    mergedSnapshot.label = formatRangeLabel(
        mergedVisibleColumn.absoluteStartBit,
        mergedVisibleColumn.absoluteEndBit
    );
    mergedSnapshot.displayFormat = QStringLiteral("hex");
    return mergedSnapshot;
}

QVector<int> overlappingVisibleColumnIndices(
    const QVector<FrameFieldColumnSnapshot>& displayColumns,
    int startBit,
    int endBit
) {
    QVector<int> visibleColumnIndices;
    for (const FrameFieldColumnSnapshot& displayColumn : displayColumns) {
        if (displayColumn.visibleColumn.absoluteEndBit < startBit
            || displayColumn.visibleColumn.absoluteStartBit > endBit) {
            continue;
        }
        visibleColumnIndices.append(displayColumn.visibleColumnIndex);
    }
    return visibleColumnIndices;
}

bool counterHintOverlaps(const FrameFieldHint& leftHint, const FrameFieldHint& rightHint) {
    return !(leftHint.absoluteEndBit < rightHint.absoluteStartBit
        || rightHint.absoluteEndBit < leftHint.absoluteStartBit);
}

bool counterHintRanksBefore(const FrameFieldHint& leftHint, const FrameFieldHint& rightHint) {
    if (!qFuzzyCompare(leftHint.confidenceScore, rightHint.confidenceScore)) {
        return leftHint.confidenceScore > rightHint.confidenceScore;
    }

    const int leftWidth = leftHint.absoluteEndBit - leftHint.absoluteStartBit + 1;
    const int rightWidth = rightHint.absoluteEndBit - rightHint.absoluteStartBit + 1;
    if (leftWidth != rightWidth) {
        return leftWidth > rightWidth;
    }

    return leftHint.absoluteStartBit < rightHint.absoluteStartBit;
}

double shannonEntropy(int oneCount, int sampleCount) {
    if (sampleCount <= 0) {
        return 0.0;
    }

    const double oneFraction = static_cast<double>(oneCount) / static_cast<double>(sampleCount);
    const double zeroFraction = 1.0 - oneFraction;
    double entropy = 0.0;
    if (oneFraction > 0.0) {
        entropy -= oneFraction * std::log2(oneFraction);
    }
    if (zeroFraction > 0.0) {
        entropy -= zeroFraction * std::log2(zeroFraction);
    }
    return entropy;
}

CounterAnalysisResult analyzeCounterPattern(
    const QList<quint64>& numericValues,
    int uniqueValueCount,
    int analyzedFrameCount,
    int bitWidth
) {
    CounterAnalysisResult result;
    if (numericValues.size() < 6 || analyzedFrameCount < 6 || bitWidth <= 0 || bitWidth > 63) {
        return result;
    }

    const quint64 modulus = quint64{1} << bitWidth;
    QHash<quint64, int> stepCounts;
    int repeatedTransitionCount = 0;
    int backwardBreakCount = 0;

    for (int valueIndex = 1; valueIndex < numericValues.size(); ++valueIndex) {
        const quint64 previousValue = numericValues.at(valueIndex - 1);
        const quint64 currentValue = numericValues.at(valueIndex);

        if (currentValue == previousValue) {
            ++repeatedTransitionCount;
            continue;
        }

        quint64 deltaValue = 0;
        bool wrapped = false;
        if (currentValue > previousValue) {
            deltaValue = currentValue - previousValue;
        } else {
            deltaValue = (modulus - previousValue) + currentValue;
            wrapped = true;
        }

        if (deltaValue == 0) {
            ++backwardBreakCount;
            continue;
        }

        if (wrapped) {
            ++result.wrapCount;
        }

        stepCounts.insert(deltaValue, stepCounts.value(deltaValue, 0) + 1);
        ++result.comparableTransitionCount;
    }

    if (result.comparableTransitionCount < 4 || stepCounts.isEmpty()) {
        return result;
    }

    quint64 dominantStep = 0;
    int dominantStepCount = 0;
    for (auto stepIterator = stepCounts.cbegin(); stepIterator != stepCounts.cend(); ++stepIterator) {
        if (stepIterator.value() > dominantStepCount
            || (stepIterator.value() == dominantStepCount && stepIterator.key() < dominantStep)) {
            dominantStep = stepIterator.key();
            dominantStepCount = stepIterator.value();
        }
    }

    const double dominantStepRatio =
        static_cast<double>(dominantStepCount) / static_cast<double>(result.comparableTransitionCount);
    const quint64 maxPossibleUniqueValues =
        (bitWidth < 63) ? (quint64{1} << bitWidth) : quint64{0x7FFFFFFFFFFFFFFF};
    const int effectiveMaxUnique =
        static_cast<int>(qMin(static_cast<quint64>(analyzedFrameCount), maxPossibleUniqueValues));
    const double uniqueValueRatio =
        static_cast<double>(uniqueValueCount) / static_cast<double>(qMax(1, effectiveMaxUnique));
    const int transitionCount = qMax(1, numericValues.size() - 1);
    const double repeatedTransitionRatio =
        static_cast<double>(repeatedTransitionCount) / static_cast<double>(transitionCount);
    const double usableTransitionRatio =
        static_cast<double>(result.comparableTransitionCount) / static_cast<double>(transitionCount);
    const double backwardBreakPenalty = backwardBreakCount > 0 ? 0.25 : 0.0;

    result.dominantStep = dominantStep;
    result.dominantStepCount = dominantStepCount;
    result.confidence = qBound(
        0.0,
        0.50 * dominantStepRatio
            + 0.25 * uniqueValueRatio
            + 0.20 * usableTransitionRatio
            + 0.05 * (result.wrapCount > 0 ? 1.0 : 0.0)
            - 0.35 * repeatedTransitionRatio
            - backwardBreakPenalty,
        1.0
    );
    result.likelyCounter =
        dominantStepRatio >= 0.70
        && uniqueValueRatio >= 0.60
        && repeatedTransitionRatio <= 0.25
        && backwardBreakCount == 0
        && result.confidence >= 0.68;
    return result;
}

}  // namespace

FrameFieldClassificationResult classifyFramedVisibleColumns(
    const data::ByteDataSource& dataSource,
    const features::framing::FrameLayout& frameLayout,
    const QVector<FrameFieldColumnSnapshot>& columnSnapshots
) {
    FrameFieldClassificationResult result;
    if (!dataSource.hasData() || !frameLayout.isFramed() || columnSnapshots.isEmpty()) {
        return result;
    }

    const int totalRowCount = static_cast<int>(frameLayout.rowCount(dataSource));
    const QVector<RowSpan> rowSpans = buildRowSpans(dataSource, frameLayout, totalRowCount);
    QVector<FrameFieldHint> singleColumnCounterHints;
    for (const FrameFieldColumnSnapshot& columnSnapshot : columnSnapshots) {
        if (columnSnapshot.visibleColumnIndex < 0
            || columnSnapshot.visibleColumn.bitWidth() <= 0) {
            continue;
        }

        const ColumnObservation observation = observeColumnSnapshot(
            dataSource,
            rowSpans,
            columnSnapshot,
            totalRowCount
        );

        if (!observation.hasReferenceValue || observation.analyzedFrameCount <= 0) {
            continue;
        }

        if (observation.allRowsMatch
            && observation.analyzedFrameCount == totalRowCount
            && observation.analyzedFrameCount >= 2) {
            FrameFieldHint constantHint;
            constantHint.visibleColumnIndex = columnSnapshot.visibleColumnIndex;
            constantHint.absoluteStartBit = columnSnapshot.visibleColumn.absoluteStartBit;
            constantHint.absoluteEndBit = columnSnapshot.visibleColumn.absoluteEndBit;
            constantHint.visibleColumnIndices = {columnSnapshot.visibleColumnIndex};
            constantHint.label = columnSnapshot.label;
            constantHint.valueText = formatDisplayValue(observation.referenceBits, columnSnapshot.displayFormat);
            constantHint.summaryText = QStringLiteral("Constant %1").arg(constantHint.valueText);
            result.constantHints.append(constantHint);
            result.constantVisibleColumnIndices.insert(columnSnapshot.visibleColumnIndex);
            continue;
        }

        if (!observation.numericCompatible
            || observation.numericValues.size() != observation.analyzedFrameCount) {
            continue;
        }

        if (observation.analyzedFrameCount < totalRowCount) {
            continue;
        }

        const CounterEvaluation counterEvaluation = evaluateCounterObservation(
            observation,
            columnSnapshot.visibleColumn.bitWidth()
        );
        if (!counterEvaluation.analysis.likelyCounter) {
            continue;
        }

        FrameFieldHint counterHint;
        counterHint.visibleColumnIndex = columnSnapshot.visibleColumnIndex;
        counterHint.absoluteStartBit = columnSnapshot.visibleColumn.absoluteStartBit;
        counterHint.absoluteEndBit = columnSnapshot.visibleColumn.absoluteEndBit;
        counterHint.visibleColumnIndices = {columnSnapshot.visibleColumnIndex};
        counterHint.confidenceScore = counterEvaluation.analysis.confidence;
        counterHint.label = columnSnapshot.label;
        counterHint.summaryText = counterSummaryText(
            counterEvaluation,
            columnSnapshot.visibleColumn.bitWidth()
        );
        singleColumnCounterHints.append(counterHint);
        result.counterVisibleColumnIndices.insert(columnSnapshot.visibleColumnIndex);
    }

    QVector<FrameFieldHint> mergedWindowCounterHints;
    QSet<int> visibleColumnsCoveredByMergedCounters;
    for (int startIndex = 0; startIndex < columnSnapshots.size(); ++startIndex) {
        for (int windowColumnCount = qMin(4, columnSnapshots.size() - startIndex); windowColumnCount >= 2; --windowColumnCount) {
            const std::optional<FrameFieldColumnSnapshot> mergedSnapshot =
                mergedCounterWindowSnapshot(columnSnapshots, startIndex, windowColumnCount);
            if (!mergedSnapshot.has_value()) {
                continue;
            }

            const ColumnObservation observation = observeColumnSnapshot(
                dataSource,
                rowSpans,
                mergedSnapshot.value(),
                totalRowCount
            );
            if (!observation.numericCompatible
                || observation.analyzedFrameCount <= 0
                || observation.numericValues.size() != observation.analyzedFrameCount) {
                continue;
            }

            if (observation.analyzedFrameCount < totalRowCount) {
                continue;
            }

            const CounterEvaluation counterEvaluation = evaluateCounterObservation(
                observation,
                mergedSnapshot->visibleColumn.bitWidth()
            );
            if (!counterEvaluation.analysis.likelyCounter) {
                continue;
            }

            FrameFieldHint counterHint;
            counterHint.visibleColumnIndex = mergedSnapshot->visibleColumnIndex;
            counterHint.absoluteStartBit = mergedSnapshot->visibleColumn.absoluteStartBit;
            counterHint.absoluteEndBit = mergedSnapshot->visibleColumn.absoluteEndBit;
            counterHint.confidenceScore = counterEvaluation.analysis.confidence;
            counterHint.label = mergedSnapshot->label;
            counterHint.summaryText = counterSummaryText(
                counterEvaluation,
                mergedSnapshot->visibleColumn.bitWidth()
            );
            for (int columnOffset = 0; columnOffset < windowColumnCount; ++columnOffset) {
                const int visibleColumnIndex = columnSnapshots.at(startIndex + columnOffset).visibleColumnIndex;
                counterHint.visibleColumnIndices.append(visibleColumnIndex);
                result.counterVisibleColumnIndices.insert(visibleColumnIndex);
                visibleColumnsCoveredByMergedCounters.insert(visibleColumnIndex);
            }
            mergedWindowCounterHints.append(counterHint);
            break;
        }
    }

    for (const FrameFieldHint& counterHint : singleColumnCounterHints) {
        if (!counterHint.visibleColumnIndices.isEmpty()
            && visibleColumnsCoveredByMergedCounters.contains(counterHint.visibleColumnIndices.first())) {
            continue;
        }
        result.counterHints.append(counterHint);
    }
    for (const FrameFieldHint& counterHint : mergedWindowCounterHints) {
        result.counterHints.append(counterHint);
    }

    auto byFieldPosition = [](const FrameFieldHint& leftHint, const FrameFieldHint& rightHint) {
        if (leftHint.absoluteStartBit != rightHint.absoluteStartBit) {
            return leftHint.absoluteStartBit < rightHint.absoluteStartBit;
        }
        return leftHint.visibleColumnIndex < rightHint.visibleColumnIndex;
    };
    std::sort(result.constantHints.begin(), result.constantHints.end(), byFieldPosition);
    std::sort(result.counterHints.begin(), result.counterHints.end(), byFieldPosition);
    return result;
}

QVector<FrameFieldHint> discoverCounterHintsByBitWindow(
    const data::ByteDataSource& dataSource,
    const features::framing::FrameLayout& frameLayout,
    const QVector<FrameFieldColumnSnapshot>& displayColumns,
    int startBit,
    int endBit
) {
    QVector<FrameFieldHint> keptHints;
    if (!dataSource.hasData() || !frameLayout.isFramed() || displayColumns.isEmpty() || startBit > endBit) {
        return keptHints;
    }

    const int totalRowCount = static_cast<int>(frameLayout.rowCount(dataSource));
    const QVector<RowSpan> rowSpans = buildRowSpans(dataSource, frameLayout, totalRowCount);
    const int boundedStartBit = qMax(0, startBit);
    const int boundedEndBit = qMax(boundedStartBit, endBit);
    const int profileWidth = boundedEndBit - boundedStartBit + 1;
    QVector<int> sampleCounts(profileWidth, 0);
    QVector<int> oneCounts(profileWidth, 0);
    for (const RowSpan& rowSpan : rowSpans) {
        if (rowSpan.startBit < 0 || rowSpan.lengthBits <= boundedStartBit) {
            continue;
        }

        const int rowProfileWidth = qMin(
            profileWidth,
            static_cast<int>(rowSpan.lengthBits - boundedStartBit)
        );
        const QByteArray rowBits = dataSource.bitRange(rowSpan.startBit + boundedStartBit, rowProfileWidth);
        for (int bitOffset = 0; bitOffset < rowBits.size(); ++bitOffset) {
            ++sampleCounts[bitOffset];
            if (rowBits.at(bitOffset) != 0) {
                ++oneCounts[bitOffset];
            }
        }
    }

    QVector<double> entropyProfile(profileWidth, 0.0);
    for (int bitOffset = 0; bitOffset < profileWidth; ++bitOffset) {
        entropyProfile[bitOffset] = shannonEntropy(oneCounts.at(bitOffset), sampleCounts.at(bitOffset));
    }

    constexpr double kGradientTolerance = 0.08;
    constexpr double kMinimumGradientSpan = 0.30;
    constexpr int kMinimumGradientWidthBits = 3;
    constexpr int kMaximumCounterWidthBits = 63;
    constexpr int kExhaustiveSubwindowWidthBits = 16;
    constexpr int kMinimumCounterWidthBits = 2;

    QVector<FrameFieldHint> candidateHints;
    QSet<quint64> seenWindowKeys;
    auto appendCounterWindow = [&](int counterStartBit, int counterEndBit) {
        if (counterStartBit < boundedStartBit || counterEndBit > boundedEndBit || counterEndBit < counterStartBit) {
            return;
        }

        const quint64 windowKey =
            (static_cast<quint64>(static_cast<quint32>(counterStartBit)) << 32)
            | static_cast<quint32>(counterEndBit);
        if (seenWindowKeys.contains(windowKey)) {
            return;
        }
        seenWindowKeys.insert(windowKey);

        FrameFieldColumnSnapshot counterSnapshot;
        counterSnapshot.visibleColumnIndex = displayColumns.first().visibleColumnIndex;
        counterSnapshot.visibleColumn.absoluteStartBit = counterStartBit;
        counterSnapshot.visibleColumn.absoluteEndBit = counterEndBit;
        counterSnapshot.label = formatRangeLabel(counterStartBit, counterEndBit);
        counterSnapshot.displayFormat = QStringLiteral("hex");

        const ColumnObservation observation = observeColumnSnapshot(
            dataSource,
            rowSpans,
            counterSnapshot,
            totalRowCount
        );
        if (!observation.numericCompatible
            || observation.analyzedFrameCount <= 0
            || observation.numericValues.size() != observation.analyzedFrameCount) {
            return;
        }

        if (observation.analyzedFrameCount < totalRowCount) {
            return;
        }

        const int counterBitWidth = counterEndBit - counterStartBit + 1;
        const CounterEvaluation counterEvaluation = evaluateCounterObservation(
            observation,
            counterBitWidth
        );
        if (!counterEvaluation.analysis.likelyCounter) {
            return;
        }

        FrameFieldHint counterHint;
        counterHint.visibleColumnIndex = displayColumns.first().visibleColumnIndex;
        counterHint.absoluteStartBit = counterStartBit;
        counterHint.absoluteEndBit = counterEndBit;
        counterHint.visibleColumnIndices = overlappingVisibleColumnIndices(
            displayColumns,
            counterHint.absoluteStartBit,
            counterHint.absoluteEndBit
        );
        counterHint.confidenceScore = counterEvaluation.analysis.confidence;
        counterHint.label = counterSnapshot.label;
        counterHint.summaryText = counterSummaryText(counterEvaluation, counterBitWidth);
        if (!counterHint.visibleColumnIndices.isEmpty()) {
            candidateHints.append(counterHint);
        }
    };

    auto appendGradientCandidates = [&](bool increasingEntropy) {
        int runStartOffset = 0;
        while (runStartOffset < entropyProfile.size()) {
            int runEndOffset = runStartOffset;
            while (runEndOffset + 1 < entropyProfile.size()) {
                const double currentEntropy = entropyProfile.at(runEndOffset);
                const double nextEntropy = entropyProfile.at(runEndOffset + 1);
                const bool continuesRun = increasingEntropy
                    ? (nextEntropy >= currentEntropy - kGradientTolerance)
                    : (nextEntropy <= currentEntropy + kGradientTolerance);
                if (!continuesRun) {
                    break;
                }
                ++runEndOffset;
            }

            int trimmedRunStart = runStartOffset;
            int trimmedRunEnd = runEndOffset;
            while (trimmedRunStart < trimmedRunEnd && entropyProfile.at(trimmedRunStart) < 0.05) {
                ++trimmedRunStart;
            }
            while (trimmedRunEnd > trimmedRunStart && entropyProfile.at(trimmedRunEnd) < 0.05) {
                --trimmedRunEnd;
            }

            const int runWidthBits = runEndOffset - runStartOffset + 1;
            const double gradientSpan = increasingEntropy
                ? entropyProfile.at(runEndOffset) - entropyProfile.at(runStartOffset)
                : entropyProfile.at(runStartOffset) - entropyProfile.at(runEndOffset);
            if (runWidthBits >= kMinimumGradientWidthBits
                && trimmedRunEnd >= trimmedRunStart
                && gradientSpan >= kMinimumGradientSpan) {
                const int regionStartBit = boundedStartBit + trimmedRunStart;
                const int regionEndBit = qMin(
                    boundedStartBit + trimmedRunEnd,
                    regionStartBit + kMaximumCounterWidthBits - 1
                );
                appendCounterWindow(regionStartBit, regionEndBit);

                const int regionWidthBits = regionEndBit - regionStartBit + 1;
                const int maxSubwindowWidth = qMin(regionWidthBits, kExhaustiveSubwindowWidthBits);
                for (int windowWidthBits = maxSubwindowWidth; windowWidthBits >= kMinimumCounterWidthBits; --windowWidthBits) {
                    for (int windowStartBit = regionStartBit;
                         windowStartBit + windowWidthBits - 1 <= regionEndBit;
                         ++windowStartBit) {
                        appendCounterWindow(windowStartBit, windowStartBit + windowWidthBits - 1);
                    }
                }
            }

            runStartOffset = runEndOffset + 1;
        }
    };

    appendGradientCandidates(true);
    appendGradientCandidates(false);

    std::sort(candidateHints.begin(), candidateHints.end(), counterHintRanksBefore);
    for (const FrameFieldHint& candidateHint : candidateHints) {
        bool overlapsKeptHint = false;
        for (const FrameFieldHint& keptHint : keptHints) {
            if (counterHintOverlaps(candidateHint, keptHint)) {
                overlapsKeptHint = true;
                break;
            }
        }
        if (overlapsKeptHint) {
            continue;
        }
        keptHints.append(candidateHint);
        if (keptHints.size() >= 4) {
            break;
        }
    }

    std::sort(keptHints.begin(), keptHints.end(), [](const FrameFieldHint& leftHint, const FrameFieldHint& rightHint) {
        if (leftHint.absoluteStartBit != rightHint.absoluteStartBit) {
            return leftHint.absoluteStartBit < rightHint.absoluteStartBit;
        }
        return leftHint.absoluteEndBit < rightHint.absoluteEndBit;
    });
    return keptHints;
}

}  // namespace bitabyte::features::classification
