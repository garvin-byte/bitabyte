#include "features/bitstream_sync_discovery/bitstream_sync_discovery_engine.h"

#include "data/byte_data_source.h"
#include "features/bitstream_sync_discovery/bitstream_sync_discovery_internal.h"

#include <QByteArray>
#include <QHash>
#include <QSet>
#include <QtGlobal>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <optional>

namespace bitabyte::features::bitstream_sync_discovery {
namespace {

using namespace detail;

[[nodiscard]] QVector<double> entropyProfileForFrameGroup(
    const data::ByteDataSource& dataSource,
    const QVector<framing::FrameSpan>& frameSpans,
    const QVector<int>& frameIndexes,
    int profileLengthBits
);

[[nodiscard]] double sampleReliabilityScore(
    qsizetype frameCount
);

[[nodiscard]] unsigned char readBitValue(const char* byteData, qsizetype bitIndex) {
    const qsizetype byteIndex = bitIndex / 8;
    const int bitIndexInByte = static_cast<int>(bitIndex % 8);
    const unsigned char byteValue = static_cast<unsigned char>(byteData[byteIndex]);
    return static_cast<unsigned char>((byteValue >> (7 - bitIndexInByte)) & 0x01);
}

[[nodiscard]] PatternValue readBitWindowValue(const char* byteData, qsizetype startBit, int bitWidth) {
    const unsigned char* unsignedByteData = reinterpret_cast<const unsigned char*>(byteData);
    qsizetype byteIndex = startBit / 8;
    int bitOffsetInByte = static_cast<int>(startBit % 8);
    int remainingBits = bitWidth;
    PatternValue windowValue;
    while (remainingBits > 0) {
        const int availableBits = 8 - bitOffsetInByte;
        const int chunkWidth = qMin(availableBits, remainingBits);
        const quint64 chunkMask = (quint64{1} << chunkWidth) - 1;
        const quint64 chunkValue =
            (static_cast<quint64>(unsignedByteData[byteIndex]) >> (availableBits - chunkWidth)) & chunkMask;
        windowValue = patternValueShiftLeft(windowValue, chunkWidth);
        windowValue.lower |= chunkValue;
        remainingBits -= chunkWidth;
        ++byteIndex;
        bitOffsetInByte = 0;
    }
    return windowValue;
}

[[nodiscard]] bool isCanceled(const std::atomic_bool* cancelRequested) {
    return cancelRequested != nullptr && cancelRequested->load(std::memory_order_relaxed);
}

[[nodiscard]] double shannonEntropy(int oneCount, int sampleCount) {
    if (sampleCount <= 0) {
        return 1.0;
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

[[nodiscard]] SingleBitStatistics singleBitStatisticsAtOffset(
    const char* byteData,
    qsizetype totalBitCount,
    const QVector<qsizetype>& matchStartBits,
    int relativeBitOffset
) {
    int oneCount = 0;
    int zeroCount = 0;

    for (qsizetype matchStartBit : matchStartBits) {
        const qsizetype sampleBit = matchStartBit + relativeBitOffset;
        if (sampleBit < 0 || sampleBit >= totalBitCount) {
            continue;
        }

        if (readBitValue(byteData, sampleBit) == 0) {
            ++zeroCount;
        } else {
            ++oneCount;
        }
    }

    SingleBitStatistics statistics;
    statistics.sampleCount = zeroCount + oneCount;
    statistics.entropy = shannonEntropy(oneCount, statistics.sampleCount);
    statistics.dominantFraction = statistics.sampleCount > 0
        ? static_cast<double>(qMax(zeroCount, oneCount)) / static_cast<double>(statistics.sampleCount)
        : 0.5;
    return statistics;
}

[[nodiscard]] QVector<double> entropyProfileForAlignedMatches(
    const char* byteData,
    qsizetype totalBitCount,
    const QVector<qsizetype>& matchStartBits,
    int startRelativeBit,
    int endRelativeBitExclusive
) {
    QVector<double> entropyProfile;
    const int profileWidth = qMax(0, endRelativeBitExclusive - startRelativeBit);
    entropyProfile.reserve(profileWidth);

    for (int relativeBit = startRelativeBit; relativeBit < endRelativeBitExclusive; ++relativeBit) {
        entropyProfile.append(
            singleBitStatisticsAtOffset(byteData, totalBitCount, matchStartBits, relativeBit).entropy
        );
    }

    return entropyProfile;
}

[[nodiscard]] GapStatistics summarizeGaps(const QVector<qsizetype>& startBits) {
    GapStatistics statistics;
    if (startBits.size() < 2) {
        return statistics;
    }

    QVector<qsizetype> gaps;
    gaps.reserve(startBits.size() - 1);

    double gapSum = 0.0;
    for (int index = 1; index < startBits.size(); ++index) {
        const qsizetype gapBits = startBits.at(index) - startBits.at(index - 1);
        gaps.append(gapBits);
        gapSum += static_cast<double>(gapBits);
    }

    std::sort(gaps.begin(), gaps.end());
    statistics.medianGapBits = gaps.at(gaps.size() / 2);
    statistics.meanGapBits = gapSum / static_cast<double>(gaps.size());

    if (statistics.meanGapBits <= 0.0) {
        return statistics;
    }

    double variance = 0.0;
    for (qsizetype gapBits : gaps) {
        const double delta = static_cast<double>(gapBits) - statistics.meanGapBits;
        variance += delta * delta;
    }
    variance /= static_cast<double>(gaps.size());
    statistics.coefficientVariation = std::sqrt(variance) / statistics.meanGapBits;
    return statistics;
}

[[nodiscard]] bool allBitsIdentical(PatternValue patternValue, int bitWidth) {
    if (bitWidth <= 1) {
        return true;
    }

    const PatternValue maskedPatternValue = patternValueMasked(patternValue, bitWidth);
    return maskedPatternValue == PatternValue{} || maskedPatternValue == patternValueMask(bitWidth);
}

[[nodiscard]] RunLengthStatistics analyzeRunLengths(PatternValue patternValue, int bitWidth) {
    RunLengthStatistics statistics;
    if (bitWidth <= 0) {
        return statistics;
    }

    QVector<int> runLengths;
    runLengths.reserve(bitWidth);

    int currentRunLength = 1;
    unsigned currentBitValue = patternValueBitAt(patternValue, bitWidth - 1) ? 1U : 0U;
    for (int bitIndex = bitWidth - 2; bitIndex >= 0; --bitIndex) {
        const unsigned bitValue = patternValueBitAt(patternValue, bitIndex) ? 1U : 0U;
        if (bitValue == currentBitValue) {
            ++currentRunLength;
            continue;
        }

        runLengths.append(currentRunLength);
        currentRunLength = 1;
        currentBitValue = bitValue;
    }
    runLengths.append(currentRunLength);

    statistics.runCount = runLengths.size();
    statistics.minimumRunLength = *std::min_element(runLengths.begin(), runLengths.end());
    statistics.maximumRunLength = *std::max_element(runLengths.begin(), runLengths.end());

    QVector<int> distinctRunLengths = runLengths;
    std::sort(distinctRunLengths.begin(), distinctRunLengths.end());
    distinctRunLengths.erase(std::unique(distinctRunLengths.begin(), distinctRunLengths.end()), distinctRunLengths.end());
    statistics.distinctRunLengthCount = distinctRunLengths.size();
    return statistics;
}

[[nodiscard]] bool shouldRejectForLowComplexity(PatternValue patternValue, int bitWidth) {
    const RunLengthStatistics runLengthStatistics = analyzeRunLengths(patternValue, bitWidth);
    if (runLengthStatistics.distinctRunLengthCount >= 2) {
        return false;
    }

    // Keep long block syncs such as FF00FF or FFFF0000 alive; reject the truly trivial
    // alternating / training-style patterns that are dominated by very short uniform runs.
    const bool isLongBlockPattern =
        bitWidth >= 16
        && runLengthStatistics.runCount <= 3
        && runLengthStatistics.minimumRunLength >= 4;
    return !isLongBlockPattern;
}

[[nodiscard]] bool isRepetitiveSubpattern(PatternValue patternValue, int bitWidth) {
    if (bitWidth <= 1) {
        return false;
    }

    for (int divisor = 1; divisor <= bitWidth / 2; ++divisor) {
        if (bitWidth % divisor != 0) {
            continue;
        }

        const PatternValue subpatternValue = extractPatternSegment(patternValue, bitWidth, 0, divisor);
        bool repeatsCleanly = true;
        for (int blockIndex = 1; blockIndex < bitWidth / divisor; ++blockIndex) {
            if (extractPatternSegment(patternValue, bitWidth, blockIndex * divisor, divisor) != subpatternValue) {
                repeatsCleanly = false;
                break;
            }
        }

        if (repeatsCleanly) {
            return true;
        }
    }

    return false;
}

[[nodiscard]] BitstreamSyncDiscoverySettings normalizedSettings(const BitstreamSyncDiscoverySettings& settings) {
    BitstreamSyncDiscoverySettings normalizedSettings = settings;
    normalizedSettings.minimumPatternBits = qBound(
        1,
        normalizedSettings.minimumPatternBits,
        kMaximumSupportedSyncPatternBits
    );
    normalizedSettings.maximumPatternBits = qBound(
        1,
        normalizedSettings.maximumPatternBits,
        kMaximumSupportedSyncPatternBits
    );
    if (normalizedSettings.maximumPatternBits < normalizedSettings.minimumPatternBits) {
        std::swap(normalizedSettings.minimumPatternBits, normalizedSettings.maximumPatternBits);
    }
    normalizedSettings.minimumMatchCount = qMax(3, normalizedSettings.minimumMatchCount);
    normalizedSettings.maximumCandidatesPerWidth = qBound(1, normalizedSettings.maximumCandidatesPerWidth, 16);
    normalizedSettings.maximumResultCount = qBound(1, normalizedSettings.maximumResultCount, 200);
    normalizedSettings.minimumExpectedFrameBits = qMax(8, normalizedSettings.minimumExpectedFrameBits);
    normalizedSettings.entropyThreshold = qBound(0.05, normalizedSettings.entropyThreshold, 0.95);
    normalizedSettings.maximumGapCoefficientVariation =
        qBound(0.1, normalizedSettings.maximumGapCoefficientVariation, 10.0);
    return normalizedSettings;
}

[[nodiscard]] QVector<qsizetype> searchPatternOccurrences(
    const char* byteData,
    qsizetype totalBitCount,
    PatternValue patternValue,
    int patternBitWidth,
    int minimumGapBits,
    OccurrenceSearchCache* occurrenceSearchCache,
    const std::atomic_bool* cancelRequested
) {
    QVector<qsizetype> matchStartBits;
    if (patternBitWidth <= 0 || totalBitCount < patternBitWidth) {
        return matchStartBits;
    }

    const OccurrenceSearchCacheKey cacheKey{
        patternValue,
        patternBitWidth,
        minimumGapBits,
    };
    if (occurrenceSearchCache != nullptr) {
        const auto cacheIterator = occurrenceSearchCache->constFind(cacheKey);
        if (cacheIterator != occurrenceSearchCache->constEnd()) {
            return cacheIterator.value();
        }
    }

    const qsizetype maximumStartBit = totalBitCount - patternBitWidth;
    const qsizetype reserveEstimate = qMin<qsizetype>(
        (maximumStartBit / qMax(1, minimumGapBits)) + 1,
        std::numeric_limits<int>::max()
    );
    matchStartBits.reserve(static_cast<int>(reserveEstimate));
    const PatternValue patternMask = patternValueMask(patternBitWidth);
    PatternValue rollingWindowValue = readBitWindowValue(byteData, 0, patternBitWidth);
    const unsigned char* unsignedByteData = reinterpret_cast<const unsigned char*>(byteData);
    BitCursor nextBitCursor{
        unsignedByteData,
        patternBitWidth / 8,
        patternBitWidth % 8,
    };

    if (rollingWindowValue == patternValue) {
        matchStartBits.append(0);
    }

    if (isCanceled(cancelRequested)) {
        return {};
    }

    for (qsizetype startBit = 1; startBit <= maximumStartBit; ++startBit) {
        if ((startBit & (kCancelPollStrideBits - 1)) == 0 && isCanceled(cancelRequested)) {
            return {};
        }

        const unsigned char nextBitValue = nextBitCursor.currentBit();
        nextBitCursor.advance();
        rollingWindowValue = patternValueAppendBit(rollingWindowValue, nextBitValue);
        if (patternBitWidth < kMaximumSupportedSyncPatternBits) {
            rollingWindowValue = patternValueAnd(rollingWindowValue, patternMask);
        }

        if (rollingWindowValue != patternValue) {
            continue;
        }

        if (!matchStartBits.isEmpty()
            && startBit - matchStartBits.constLast() < minimumGapBits) {
            continue;
        }

        matchStartBits.append(startBit);
    }

    if (occurrenceSearchCache != nullptr) {
        occurrenceSearchCache->insert(cacheKey, matchStartBits);
    }
    return matchStartBits;
}

[[nodiscard]] double quickPreanalysisScore(
    const data::ByteDataSource& dataSource,
    const PreanalyzedCandidate& candidate,
    const BitstreamSyncDiscoverySettings& settings
) {
    const double sampleReliability = qBound(
        0.25,
        sampleReliabilityScore(candidate.frameSpans.size()),
        1.0
    );
    const double spacingScore = candidate.adjustedGapStatistics.coefficientVariation <= 0.0
        ? 1.0
        : qBound(0.0, 1.0 - qMin(candidate.adjustedGapStatistics.coefficientVariation, 1.0), 1.0);
    const double uniquenessScore = candidate.patternRecord.matchCount > 0
        ? qBound(
              0.0,
              1.0 - (static_cast<double>(candidate.patternRecord.strongestNearMissCount)
                     / static_cast<double>(candidate.patternRecord.matchCount)),
              1.0
          )
        : 0.0;
    const double expectedFrameCount = candidate.frameLengthSummary.medianLengthBits > 0
        ? static_cast<double>(dataSource.bitCount()) / static_cast<double>(candidate.frameLengthSummary.medianLengthBits)
        : static_cast<double>(candidate.frameSpans.size());
    const double occurrenceScore = expectedFrameCount > 0.0
        ? qBound(0.0, static_cast<double>(candidate.frameSpans.size()) / expectedFrameCount, 1.0)
        : 0.0;
    const double widthPreferenceScore =
        0.65 * qBound(
                   0.0,
                   static_cast<double>(canonicalWidthPreferenceScore(candidate.adjustedPattern.bitWidth)) / 3.0,
                   1.0
               )
        + 0.35 * qBound(0.0, static_cast<double>(candidate.adjustedPattern.bitWidth) / 32.0, 1.0);

    return 100.0 * (
        0.30 * candidate.adjustedPattern.cliffSharpnessScore
        + 0.25 * spacingScore
        + 0.20 * uniquenessScore
        + 0.15 * occurrenceScore
        + 0.10 * widthPreferenceScore
    ) * sampleReliability;
}

void ensureEntropyProfileForFrameGroup(
    const data::ByteDataSource& dataSource,
    const QVector<framing::FrameSpan>& frameSpans,
    FrameLengthGroup* frameLengthGroup
) {
    if (frameLengthGroup == nullptr) {
        return;
    }

    const int entropyScanLengthBits = static_cast<int>(qMax<qsizetype>(1, frameLengthGroup->summary.medianLengthBits));
    if (frameLengthGroup->entropyProfile.size() == entropyScanLengthBits) {
        return;
    }

    frameLengthGroup->entropyProfile = entropyProfileForFrameGroup(
        dataSource,
        frameSpans,
        frameLengthGroup->frameIndexes,
        entropyScanLengthBits
    );
}

[[nodiscard]] QVector<PatternOccurrenceRecord> collectCandidatePatternRecordsForWidth(
    const data::ByteDataSource& dataSource,
    int patternBitWidth,
    const BitstreamSyncDiscoverySettings& settings,
    int startBitStep,
    const std::atomic_bool* cancelRequested
) {
    QVector<PatternOccurrenceRecord> patternRecords;
    const qsizetype totalBitCount = dataSource.bitCount();
    if (patternBitWidth <= 0 || totalBitCount < patternBitWidth) {
        return patternRecords;
    }

    startBitStep = qMax(1, startBitStep);
    const int minimumGapBits = qMax(patternBitWidth, settings.minimumExpectedFrameBits / 2);
    const char* byteData = dataSource.rawBytes().constData();
    const qsizetype maximumStartBit = totalBitCount - patternBitWidth;
    const PatternValue patternMask = patternValueMask(patternBitWidth);
    const unsigned char* unsignedByteData = reinterpret_cast<const unsigned char*>(byteData);

    QHash<PatternValue, PatternAccumulator> accumulatorsByPattern;
    accumulatorsByPattern.reserve(static_cast<int>(qMin<qsizetype>((maximumStartBit / startBitStep) + 1, 250000)));

    if (isCanceled(cancelRequested)) {
        return {};
    }

    if (startBitStep == 1) {
        PatternValue rollingWindowValue = readBitWindowValue(byteData, 0, patternBitWidth);
        BitCursor nextBitCursor{
            unsignedByteData,
            patternBitWidth / 8,
            patternBitWidth % 8,
        };
        {
            PatternAccumulator& accumulator = accumulatorsByPattern[rollingWindowValue];
            accumulator.lastAcceptedStartBit = 0;
            accumulator.acceptedStartBits.append(0);
        }

        for (qsizetype startBit = 1; startBit <= maximumStartBit; ++startBit) {
            if ((startBit & (kCancelPollStrideBits - 1)) == 0 && isCanceled(cancelRequested)) {
                return {};
            }

            const unsigned char nextBitValue = nextBitCursor.currentBit();
            nextBitCursor.advance();
            rollingWindowValue = patternValueAppendBit(rollingWindowValue, nextBitValue);
            if (patternBitWidth < kMaximumSupportedSyncPatternBits) {
                rollingWindowValue = patternValueAnd(rollingWindowValue, patternMask);
            }

            PatternAccumulator& accumulator = accumulatorsByPattern[rollingWindowValue];
            if (!accumulator.acceptedStartBits.isEmpty()
                && startBit - accumulator.lastAcceptedStartBit < minimumGapBits) {
                continue;
            }

            accumulator.lastAcceptedStartBit = startBit;
            accumulator.acceptedStartBits.append(startBit);
        }
    } else {
        for (qsizetype startBit = 0; startBit <= maximumStartBit; startBit += startBitStep) {
            if ((startBit & (kCancelPollStrideBits - 1)) == 0 && isCanceled(cancelRequested)) {
                return {};
            }

            const PatternValue windowValue = readBitWindowValue(byteData, startBit, patternBitWidth);
            PatternAccumulator& accumulator = accumulatorsByPattern[windowValue];
            if (!accumulator.acceptedStartBits.isEmpty()
                && startBit - accumulator.lastAcceptedStartBit < minimumGapBits) {
                continue;
            }

            accumulator.lastAcceptedStartBit = startBit;
            accumulator.acceptedStartBits.append(startBit);
        }
    }

    QHash<PatternValue, qsizetype> acceptedCountsByPattern;
    acceptedCountsByPattern.reserve(accumulatorsByPattern.size());

    for (auto accumulatorIterator = accumulatorsByPattern.constBegin();
         accumulatorIterator != accumulatorsByPattern.constEnd();
         ++accumulatorIterator) {
        const QVector<qsizetype>& acceptedStartBits = accumulatorIterator.value().acceptedStartBits;
        if (acceptedStartBits.size() < settings.minimumMatchCount) {
            continue;
        }

        const PatternValue patternValue = accumulatorIterator.key();
        if (allBitsIdentical(patternValue, patternBitWidth)) {
            continue;
        }
        if (shouldRejectForLowComplexity(patternValue, patternBitWidth)) {
            continue;
        }
        if (isRepetitiveSubpattern(patternValue, patternBitWidth)) {
            continue;
        }

        const GapStatistics gapStatistics = summarizeGaps(acceptedStartBits);
        if (gapStatistics.medianGapBits < settings.minimumExpectedFrameBits) {
            continue;
        }
        if (gapStatistics.coefficientVariation > settings.maximumGapCoefficientVariation) {
            continue;
        }

        acceptedCountsByPattern.insert(patternValue, acceptedStartBits.size());

        PatternOccurrenceRecord patternRecord;
        patternRecord.patternValue = patternValue;
        patternRecord.bitWidth = patternBitWidth;
        patternRecord.matchStartBits = acceptedStartBits;
        patternRecord.matchCount = acceptedStartBits.size();
        patternRecord.medianGapBits = gapStatistics.medianGapBits;
        patternRecord.gapCoefficientVariation = gapStatistics.coefficientVariation;
        patternRecord.preliminaryScore =
            static_cast<double>(patternRecord.matchCount) / (1.0 + gapStatistics.coefficientVariation);
        patternRecords.append(patternRecord);
    }

    if (patternRecords.isEmpty()) {
        return patternRecords;
    }

    for (PatternOccurrenceRecord& patternRecord : patternRecords) {
        int strongestNearMissCount = 0;
        int leftNearMissCount = 0;
        int rightNearMissCount = 0;
        for (int bitIndex = 0; bitIndex < patternRecord.bitWidth; ++bitIndex) {
            const PatternValue nearMissPatternValue =
                patternValueWithFlippedBit(patternRecord.patternValue, bitIndex);
            const int nearMissCount = static_cast<int>(acceptedCountsByPattern.value(nearMissPatternValue, 0));
            strongestNearMissCount = qMax(strongestNearMissCount, nearMissCount);
            if (bitIndex == patternRecord.bitWidth - 1) {
                leftNearMissCount = nearMissCount;
            }
            if (bitIndex == 0) {
                rightNearMissCount = nearMissCount;
            }
        }

        patternRecord.strongestNearMissCount = strongestNearMissCount;
        patternRecord.outerLeftNearMissCount = leftNearMissCount;
        patternRecord.outerRightNearMissCount = rightNearMissCount;
    }

    std::sort(patternRecords.begin(), patternRecords.end(), patternRecordRanksBefore);
    const int retainedCandidateLimit = internalCandidateRetentionLimit(settings);
    if (patternRecords.size() > retainedCandidateLimit) {
        const PatternOccurrenceRecord& cutoffRecord = patternRecords.at(retainedCandidateLimit - 1);
        const int hardRetentionLimit = qMin(
            patternRecords.size(),
            qMax(retainedCandidateLimit + 12, retainedCandidateLimit)
        );
        int retainedCount = retainedCandidateLimit;
        while (retainedCount < patternRecords.size() && retainedCount < hardRetentionLimit) {
            const PatternOccurrenceRecord& candidateRecord = patternRecords.at(retainedCount);
            if (candidateRecord.matchCount != cutoffRecord.matchCount) {
                break;
            }
            if (std::abs(candidateRecord.gapCoefficientVariation - cutoffRecord.gapCoefficientVariation) > 0.05) {
                break;
            }
            ++retainedCount;
        }
        patternRecords.resize(retainedCount);
    }

    return patternRecords;
}

[[nodiscard]] bool containsStartBit(const QVector<qsizetype>& sortedStartBits, qsizetype targetStartBit) {
    return std::binary_search(sortedStartBits.begin(), sortedStartBits.end(), targetStartBit);
}

[[nodiscard]] int transitionWidthFromBoundary(
    const QVector<double>& entropyProfile,
    int startIndex,
    int step,
    double entropyThreshold
) {
    if (entropyProfile.isEmpty() || startIndex < 0 || startIndex >= entropyProfile.size()) {
        return 1;
    }

    int transitionWidth = 0;
    bool crossedThreshold = false;
    for (int index = startIndex; index >= 0 && index < entropyProfile.size(); index += step) {
        ++transitionWidth;
        if (entropyProfile.at(index) >= entropyThreshold) {
            crossedThreshold = true;
        }
        if (crossedThreshold && entropyProfile.at(index) >= kEntropyHighThreshold) {
            break;
        }
    }
    return qMax(1, transitionWidth);
}

[[nodiscard]] TrimEvaluation evaluateTrim(
    const data::ByteDataSource& dataSource,
    const QVector<qsizetype>& currentMatchStartBits,
    PatternValue currentPatternValue,
    int currentPatternBitWidth,
    TrimSide trimSide,
    const BitstreamSyncDiscoverySettings& settings,
    double outerNearMissRatio,
    OccurrenceSearchCache* occurrenceSearchCache,
    const std::atomic_bool* cancelRequested
) {
    TrimEvaluation evaluation;
    if (currentPatternBitWidth <= settings.minimumPatternBits || currentMatchStartBits.isEmpty()) {
        return evaluation;
    }

    const char* byteData = dataSource.rawBytes().constData();
    const qsizetype totalBitCount = dataSource.bitCount();
    const int newPatternBitWidth = currentPatternBitWidth - 1;
    const PatternValue newPatternValue = trimSide == TrimSide::Left
        ? patternValueMasked(currentPatternValue, newPatternBitWidth)
        : patternValueShiftRight(currentPatternValue, 1);
    const int minimumGapBits = qMax(newPatternBitWidth, settings.minimumExpectedFrameBits / 2);
    QVector<qsizetype> trimmedMatchStartBits = searchPatternOccurrences(
        byteData,
        totalBitCount,
        newPatternValue,
        newPatternBitWidth,
        minimumGapBits,
        occurrenceSearchCache,
        cancelRequested
    );
    if (trimmedMatchStartBits.isEmpty() || isCanceled(cancelRequested)) {
        return evaluation;
    }

    evaluation.patternValue = newPatternValue;
    evaluation.bitWidth = newPatternBitWidth;
    evaluation.matchStartBits = trimmedMatchStartBits;

    QVector<qsizetype> projectedCurrentStartBits;
    projectedCurrentStartBits.reserve(currentMatchStartBits.size());
    for (qsizetype currentStartBit : currentMatchStartBits) {
        projectedCurrentStartBits.append(trimSide == TrimSide::Left ? currentStartBit + 1 : currentStartBit);
    }

    evaluation.superset = true;
    for (qsizetype projectedStartBit : projectedCurrentStartBits) {
        if (!containsStartBit(trimmedMatchStartBits, projectedStartBit)) {
            evaluation.superset = false;
            break;
        }
    }
    if (!evaluation.superset) {
        return evaluation;
    }

    bool hasObservedBit = false;
    bool newBitsUniform = true;
    int firstObservedBitValue = -1;

    for (qsizetype trimmedStartBit : trimmedMatchStartBits) {
        if (containsStartBit(projectedCurrentStartBits, trimmedStartBit)) {
            continue;
        }

        ++evaluation.newMatchCount;

        const qsizetype removedBitIndex = trimSide == TrimSide::Left
            ? trimmedStartBit - 1
            : trimmedStartBit + newPatternBitWidth;
        if (removedBitIndex < 0 || removedBitIndex >= totalBitCount) {
            continue;
        }

        const int removedBitValue = readBitValue(byteData, removedBitIndex);
        if (!hasObservedBit) {
            firstObservedBitValue = removedBitValue;
            hasObservedBit = true;
            continue;
        }
        if (removedBitValue != firstObservedBitValue) {
            newBitsUniform = false;
        }
    }

    evaluation.newMatchRatio = currentMatchStartBits.isEmpty()
        ? 0.0
        : static_cast<double>(evaluation.newMatchCount) / static_cast<double>(currentMatchStartBits.size());
    evaluation.newBitsUniform = !hasObservedBit || newBitsUniform;
    evaluation.boundaryStatistics = singleBitStatisticsAtOffset(
        byteData,
        totalBitCount,
        currentMatchStartBits,
        trimSide == TrimSide::Left ? -1 : currentPatternBitWidth
    );

    if (evaluation.boundaryStatistics.entropy >= settings.entropyThreshold) {
        return evaluation;
    }
    if (evaluation.newMatchCount <= 0) {
        return evaluation;
    }
    if (evaluation.newMatchCount < 2 && evaluation.newMatchRatio < 0.05 && outerNearMissRatio < 0.50) {
        return evaluation;
    }
    if (evaluation.newBitsUniform && evaluation.newMatchRatio < 0.10 && outerNearMissRatio < 0.60) {
        return evaluation;
    }
    if (!evaluation.newBitsUniform && outerNearMissRatio < 0.15) {
        return evaluation;
    }
    if (evaluation.newMatchRatio <= 0.01 && outerNearMissRatio < 0.35) {
        return evaluation;
    }

    evaluation.confidence =
        (1.0 - evaluation.boundaryStatistics.entropy)
        + (evaluation.newBitsUniform ? 0.0 : 0.4)
        + qMin(0.9, evaluation.newMatchRatio * 3.0)
        + qMin(0.8, outerNearMissRatio);
    evaluation.shouldTrim = true;
    return evaluation;
}

[[nodiscard]] WidthAdjustedPattern determineAdjustedPattern(
    const data::ByteDataSource& dataSource,
    const PatternOccurrenceRecord& patternRecord,
    const BitstreamSyncDiscoverySettings& settings,
    OccurrenceSearchCache* occurrenceSearchCache,
    const std::atomic_bool* cancelRequested
) {
    WidthAdjustedPattern adjustedPattern;
    adjustedPattern.patternValue = patternRecord.patternValue;
    adjustedPattern.bitWidth = patternRecord.bitWidth;
    adjustedPattern.matchStartBits = patternRecord.matchStartBits;

    PatternValue currentPatternValue = patternRecord.patternValue;
    int currentPatternBitWidth = patternRecord.bitWidth;
    QVector<qsizetype> currentMatchStartBits = patternRecord.matchStartBits;
    bool firstTrimPass = true;

    while (currentPatternBitWidth > settings.minimumPatternBits) {
        if (isCanceled(cancelRequested)) {
            return {};
        }

        const double outerLeftNearMissRatio = firstTrimPass && patternRecord.matchCount > 0
            ? static_cast<double>(patternRecord.outerLeftNearMissCount) / static_cast<double>(patternRecord.matchCount)
            : 0.0;
        const double outerRightNearMissRatio = firstTrimPass && patternRecord.matchCount > 0
            ? static_cast<double>(patternRecord.outerRightNearMissCount) / static_cast<double>(patternRecord.matchCount)
            : 0.0;

        const TrimEvaluation leftTrimEvaluation = evaluateTrim(
            dataSource,
            currentMatchStartBits,
            currentPatternValue,
            currentPatternBitWidth,
            TrimSide::Left,
            settings,
            outerLeftNearMissRatio,
            occurrenceSearchCache,
            cancelRequested
        );
        const TrimEvaluation rightTrimEvaluation = evaluateTrim(
            dataSource,
            currentMatchStartBits,
            currentPatternValue,
            currentPatternBitWidth,
            TrimSide::Right,
            settings,
            outerRightNearMissRatio,
            occurrenceSearchCache,
            cancelRequested
        );
        if (isCanceled(cancelRequested)) {
            return {};
        }

        const bool canTrimLeft = leftTrimEvaluation.shouldTrim;
        const bool canTrimRight = rightTrimEvaluation.shouldTrim;
        if (!canTrimLeft && !canTrimRight) {
            break;
        }

        const bool trimLeft = canTrimLeft
            && (!canTrimRight || leftTrimEvaluation.confidence >= rightTrimEvaluation.confidence);
        const TrimEvaluation& chosenTrimEvaluation = trimLeft ? leftTrimEvaluation : rightTrimEvaluation;

        currentPatternValue = chosenTrimEvaluation.patternValue;
        currentPatternBitWidth = chosenTrimEvaluation.bitWidth;
        currentMatchStartBits = chosenTrimEvaluation.matchStartBits;
        if (trimLeft) {
            ++adjustedPattern.leftTrimBits;
        } else {
            ++adjustedPattern.rightTrimBits;
        }
        firstTrimPass = false;
    }

    adjustedPattern.patternValue = currentPatternValue;
    adjustedPattern.bitWidth = currentPatternBitWidth;
    adjustedPattern.matchStartBits = currentMatchStartBits;

    const char* byteData = dataSource.rawBytes().constData();
    const qsizetype totalBitCount = dataSource.bitCount();
    adjustedPattern.leftBoundaryStatistics =
        singleBitStatisticsAtOffset(byteData, totalBitCount, currentMatchStartBits, -1);
    adjustedPattern.rightBoundaryStatistics =
        singleBitStatisticsAtOffset(byteData, totalBitCount, currentMatchStartBits, currentPatternBitWidth);

    const QVector<double> localEntropyProfile = entropyProfileForAlignedMatches(
        byteData,
        totalBitCount,
        currentMatchStartBits,
        -kEntropyWindowPaddingBits,
        currentPatternBitWidth + kEntropyWindowPaddingBits
    );
    const int leftBoundaryProfileIndex = kEntropyWindowPaddingBits - 1;
    const int rightBoundaryProfileIndex = kEntropyWindowPaddingBits + currentPatternBitWidth;
    adjustedPattern.leftTransitionWidth = transitionWidthFromBoundary(
        localEntropyProfile,
        leftBoundaryProfileIndex,
        -1,
        settings.entropyThreshold
    );
    adjustedPattern.rightTransitionWidth = transitionWidthFromBoundary(
        localEntropyProfile,
        rightBoundaryProfileIndex,
        1,
        settings.entropyThreshold
    );
    const double averageTransitionWidth =
        static_cast<double>(adjustedPattern.leftTransitionWidth + adjustedPattern.rightTransitionWidth) / 2.0;
    adjustedPattern.cliffSharpnessScore = qBound(
        0.0,
        1.0 - (averageTransitionWidth / static_cast<double>(qMax(1, currentPatternBitWidth))),
        1.0
    );
    return adjustedPattern;
}

[[nodiscard]] QVector<framing::FrameSpan> buildFrameSpans(
    qsizetype totalBitCount,
    const QVector<qsizetype>& matchStartBits,
    int syncBitWidth
) {
    QVector<framing::FrameSpan> frameSpans;
    if (matchStartBits.isEmpty()) {
        return frameSpans;
    }

    frameSpans.reserve(matchStartBits.size());
    for (int matchIndex = 0; matchIndex < matchStartBits.size(); ++matchIndex) {
        const qsizetype frameStartBit = matchStartBits.at(matchIndex);
        const qsizetype nextFrameStartBit =
            matchIndex + 1 < matchStartBits.size() ? matchStartBits.at(matchIndex + 1) : totalBitCount;
        const qsizetype frameLengthBits = qMax<qsizetype>(0, nextFrameStartBit - frameStartBit);
        if (frameLengthBits < syncBitWidth) {
            continue;
        }

        framing::FrameSpan frameSpan;
        frameSpan.startBit = frameStartBit;
        frameSpan.lengthBits = frameLengthBits;
        frameSpans.append(frameSpan);
    }

    return frameSpans;
}

[[nodiscard]] FrameLengthSummary summarizeFrameLengths(const QVector<framing::FrameSpan>& frameSpans) {
    FrameLengthSummary frameLengthSummary;
    if (frameSpans.isEmpty()) {
        return frameLengthSummary;
    }

    QVector<qsizetype> frameLengths;
    frameLengths.reserve(frameSpans.size());

    double totalLengthBits = 0.0;
    for (const framing::FrameSpan& frameSpan : frameSpans) {
        frameLengths.append(frameSpan.lengthBits);
        totalLengthBits += static_cast<double>(frameSpan.lengthBits);
    }

    std::sort(frameLengths.begin(), frameLengths.end());
    frameLengthSummary.minimumLengthBits = frameLengths.first();
    frameLengthSummary.maximumLengthBits = frameLengths.last();
    frameLengthSummary.medianLengthBits = frameLengths.at(frameLengths.size() / 2);
    frameLengthSummary.averageLengthBits = totalLengthBits / static_cast<double>(frameLengths.size());
    return frameLengthSummary;
}

[[nodiscard]] FrameLengthSummary summarizeFrameGroup(
    const QVector<framing::FrameSpan>& frameSpans,
    const QVector<int>& frameIndexes
) {
    QVector<framing::FrameSpan> groupedFrameSpans;
    groupedFrameSpans.reserve(frameIndexes.size());
    for (int frameIndex : frameIndexes) {
        if (frameIndex < 0 || frameIndex >= frameSpans.size()) {
            continue;
        }
        groupedFrameSpans.append(frameSpans.at(frameIndex));
    }
    return summarizeFrameLengths(groupedFrameSpans);
}

[[nodiscard]] QVector<FrameLengthGroup> buildInitialFrameLengthGroups(const QVector<framing::FrameSpan>& frameSpans) {
    QVector<FrameLengthGroup> frameLengthGroups;
    if (frameSpans.isEmpty()) {
        return frameLengthGroups;
    }

    struct LengthIndexedFrame {
        qsizetype lengthBits = 0;
        int frameIndex = -1;
    };

    QVector<LengthIndexedFrame> sortedFrames;
    sortedFrames.reserve(frameSpans.size());
    for (int frameIndex = 0; frameIndex < frameSpans.size(); ++frameIndex) {
        LengthIndexedFrame indexedFrame;
        indexedFrame.lengthBits = frameSpans.at(frameIndex).lengthBits;
        indexedFrame.frameIndex = frameIndex;
        sortedFrames.append(indexedFrame);
    }

    std::sort(
        sortedFrames.begin(),
        sortedFrames.end(),
        [](const LengthIndexedFrame& leftFrame, const LengthIndexedFrame& rightFrame) {
            if (leftFrame.lengthBits != rightFrame.lengthBits) {
                return leftFrame.lengthBits < rightFrame.lengthBits;
            }
            return leftFrame.frameIndex < rightFrame.frameIndex;
        }
    );

    FrameLengthGroup currentGroup;
    currentGroup.frameIndexes.append(sortedFrames.first().frameIndex);
    qsizetype previousLengthBits = sortedFrames.first().lengthBits;

    for (int sortedIndex = 1; sortedIndex < sortedFrames.size(); ++sortedIndex) {
        const LengthIndexedFrame& indexedFrame = sortedFrames.at(sortedIndex);
        if (indexedFrame.lengthBits - previousLengthBits > kLengthClusterGapBits) {
            frameLengthGroups.append(currentGroup);
            currentGroup = FrameLengthGroup();
        }

        currentGroup.frameIndexes.append(indexedFrame.frameIndex);
        previousLengthBits = indexedFrame.lengthBits;
    }
    frameLengthGroups.append(currentGroup);

    for (FrameLengthGroup& frameLengthGroup : frameLengthGroups) {
        std::sort(frameLengthGroup.frameIndexes.begin(), frameLengthGroup.frameIndexes.end());
        frameLengthGroup.summary = summarizeFrameGroup(frameSpans, frameLengthGroup.frameIndexes);
    }

    return frameLengthGroups;
}

[[nodiscard]] QVector<double> entropyProfileForFrameGroup(
    const data::ByteDataSource& dataSource,
    const QVector<framing::FrameSpan>& frameSpans,
    const QVector<int>& frameIndexes,
    int profileLengthBits
) {
    QVector<double> entropyProfile;
    if (profileLengthBits <= 0 || frameIndexes.isEmpty()) {
        return entropyProfile;
    }

    entropyProfile.resize(profileLengthBits);
    QVector<int> sampleCounts(profileLengthBits, 0);
    QVector<int> oneCounts(profileLengthBits, 0);

    const char* byteData = dataSource.rawBytes().constData();
    const qsizetype totalBitCount = dataSource.bitCount();
    for (int frameIndex : frameIndexes) {
        if (frameIndex < 0 || frameIndex >= frameSpans.size()) {
            continue;
        }

        const framing::FrameSpan& frameSpan = frameSpans.at(frameIndex);
        const int scanLengthBits = qMin(profileLengthBits, static_cast<int>(frameSpan.lengthBits));
        for (int relativeBit = 0; relativeBit < scanLengthBits; ++relativeBit) {
            const qsizetype absoluteBit = frameSpan.startBit + relativeBit;
            if (absoluteBit < 0 || absoluteBit >= totalBitCount) {
                continue;
            }

            ++sampleCounts[relativeBit];
            if (readBitValue(byteData, absoluteBit) != 0) {
                ++oneCounts[relativeBit];
            }
        }
    }

    for (int bitIndex = 0; bitIndex < profileLengthBits; ++bitIndex) {
        entropyProfile[bitIndex] = shannonEntropy(oneCounts[bitIndex], sampleCounts[bitIndex]);
    }
    return entropyProfile;
}

[[nodiscard]] QVector<EntropyIsland> findLowEntropyIslands(
    const QVector<double>& entropyProfile,
    const QVector<int>& frameIndexes,
    double entropyThreshold
) {
    QVector<EntropyIsland> entropyIslands;
    bool inIsland = false;
    int islandStartBit = 0;

    for (int bitIndex = 0; bitIndex < entropyProfile.size(); ++bitIndex) {
        if (entropyProfile.at(bitIndex) < entropyThreshold) {
            if (!inIsland) {
                inIsland = true;
                islandStartBit = bitIndex;
            }
            continue;
        }

        if (!inIsland) {
            continue;
        }

        EntropyIsland entropyIsland;
        entropyIsland.startBit = islandStartBit;
        entropyIsland.endBit = bitIndex - 1;
        double totalEntropy = 0.0;
        for (int islandBit = entropyIsland.startBit; islandBit <= entropyIsland.endBit; ++islandBit) {
            totalEntropy += entropyProfile.at(islandBit);
        }
        entropyIsland.averageEntropy =
            totalEntropy / static_cast<double>(qMax(1, entropyIsland.endBit - entropyIsland.startBit + 1));
        entropyIsland.supportingFrameCount = frameIndexes.size();
        entropyIslands.append(entropyIsland);
        inIsland = false;
    }

    if (inIsland) {
        EntropyIsland entropyIsland;
        entropyIsland.startBit = islandStartBit;
        entropyIsland.endBit = entropyProfile.size() - 1;
        double totalEntropy = 0.0;
        for (int islandBit = entropyIsland.startBit; islandBit <= entropyIsland.endBit; ++islandBit) {
            totalEntropy += entropyProfile.at(islandBit);
        }
        entropyIsland.averageEntropy =
            totalEntropy / static_cast<double>(qMax(1, entropyIsland.endBit - entropyIsland.startBit + 1));
        entropyIsland.supportingFrameCount = frameIndexes.size();
        entropyIslands.append(entropyIsland);
    }

    return entropyIslands;
}

[[nodiscard]] int entropyIslandWidthBits(const EntropyIsland& entropyIsland) {
    return entropyIsland.endBit - entropyIsland.startBit + 1;
}

[[nodiscard]] bool isStrongEntropyIsland(
    const EntropyIsland& entropyIsland,
    double entropyThreshold
) {
    const int islandWidthBits = entropyIslandWidthBits(entropyIsland);
    if (islandWidthBits < kStrongIslandMinimumWidthBits) {
        return false;
    }

    const double strongEntropyThreshold = qMin(entropyThreshold * 0.85, 0.24);
    return entropyIsland.averageEntropy <= strongEntropyThreshold;
}

[[nodiscard]] bool isValidationEntropyIsland(
    const EntropyIsland& entropyIsland,
    int syncBitWidth,
    double entropyThreshold
) {
    if (entropyIsland.endBit < syncBitWidth) {
        return false;
    }

    const int islandWidthBits = entropyIslandWidthBits(entropyIsland);
    if (islandWidthBits < kValidationIslandMinimumWidthBits) {
        return false;
    }

    const double validationEntropyThreshold = qMin(entropyThreshold * 0.75, 0.18);
    return entropyIsland.averageEntropy <= validationEntropyThreshold;
}

[[nodiscard]] int strongEntropyIslandCount(
    const QVector<EntropyIsland>& entropyIslands,
    double entropyThreshold
) {
    int count = 0;
    for (const EntropyIsland& entropyIsland : entropyIslands) {
        if (isStrongEntropyIsland(entropyIsland, entropyThreshold)) {
            ++count;
        }
    }
    return count;
}

[[nodiscard]] int validationEntropyIslandCount(
    const QVector<EntropyIsland>& entropyIslands,
    int syncBitWidth,
    double entropyThreshold
) {
    int count = 0;
    for (const EntropyIsland& entropyIsland : entropyIslands) {
        if (isValidationEntropyIsland(entropyIsland, syncBitWidth, entropyThreshold)) {
            ++count;
        }
    }
    return count;
}

[[nodiscard]] double averageStrongEntropyIslandWidth(
    const QVector<EntropyIsland>& entropyIslands,
    double entropyThreshold
) {
    double totalWidthBits = 0.0;
    int count = 0;
    for (const EntropyIsland& entropyIsland : entropyIslands) {
        if (!isStrongEntropyIsland(entropyIsland, entropyThreshold)) {
            continue;
        }

        totalWidthBits += static_cast<double>(entropyIslandWidthBits(entropyIsland));
        ++count;
    }

    return count > 0 ? totalWidthBits / static_cast<double>(count) : 0.0;
}

[[nodiscard]] double meanAbsoluteDifference(
    const QVector<double>& leftEntropyProfile,
    const QVector<double>& rightEntropyProfile,
    int overlapBits
) {
    if (overlapBits <= 0) {
        return std::numeric_limits<double>::infinity();
    }

    double totalDifference = 0.0;
    for (int bitIndex = 0; bitIndex < overlapBits; ++bitIndex) {
        totalDifference += std::abs(leftEntropyProfile.at(bitIndex) - rightEntropyProfile.at(bitIndex));
    }
    return totalDifference / static_cast<double>(overlapBits);
}

void mergeFrameLengthGroupsByEntropy(
    const data::ByteDataSource& dataSource,
    const QVector<framing::FrameSpan>& frameSpans,
    QVector<FrameLengthGroup>* frameLengthGroups
) {
    if (frameLengthGroups == nullptr || frameLengthGroups->size() < 2) {
        return;
    }

    bool mergedAnyGroup = true;
    while (mergedAnyGroup) {
        mergedAnyGroup = false;

        for (FrameLengthGroup& frameLengthGroup : *frameLengthGroups) {
            ensureEntropyProfileForFrameGroup(dataSource, frameSpans, &frameLengthGroup);
        }

        for (int leftIndex = 0; leftIndex < frameLengthGroups->size() && !mergedAnyGroup; ++leftIndex) {
            for (int rightIndex = leftIndex + 1; rightIndex < frameLengthGroups->size(); ++rightIndex) {
                const FrameLengthGroup& leftGroup = frameLengthGroups->at(leftIndex);
                const FrameLengthGroup& rightGroup = frameLengthGroups->at(rightIndex);
                const int overlapBits = static_cast<int>(qMin(
                    leftGroup.summary.medianLengthBits,
                    rightGroup.summary.medianLengthBits
                ));
                if (overlapBits <= 0) {
                    continue;
                }

                const double profileDifference = meanAbsoluteDifference(
                    leftGroup.entropyProfile,
                    rightGroup.entropyProfile,
                    overlapBits
                );
                if (profileDifference >= 0.1) {
                    continue;
                }

                FrameLengthGroup mergedGroup = leftGroup;
                mergedGroup.frameIndexes += rightGroup.frameIndexes;
                std::sort(mergedGroup.frameIndexes.begin(), mergedGroup.frameIndexes.end());
                mergedGroup.summary = summarizeFrameGroup(frameSpans, mergedGroup.frameIndexes);
                mergedGroup.entropyProfile.clear();
                mergedGroup.lowEntropyIslands.clear();
                (*frameLengthGroups)[leftIndex] = mergedGroup;
                frameLengthGroups->removeAt(rightIndex);
                mergedAnyGroup = true;
                break;
            }
        }
    }

    for (FrameLengthGroup& frameLengthGroup : *frameLengthGroups) {
        ensureEntropyProfileForFrameGroup(dataSource, frameSpans, &frameLengthGroup);
    }
}

[[nodiscard]] QString classifyFrameDistribution(
    const FrameLengthSummary& frameLengthSummary,
    const QVector<FrameLengthGroup>& frameLengthGroups
) {
    if (frameLengthGroups.isEmpty()) {
        return QStringLiteral("irregular");
    }

    const qsizetype globalSpreadBits = frameLengthSummary.maximumLengthBits - frameLengthSummary.minimumLengthBits;
    if (globalSpreadBits <= 1) {
        return QStringLiteral("fixed-length");
    }

    bool allGroupsTight = true;
    for (const FrameLengthGroup& frameLengthGroup : frameLengthGroups) {
        const qsizetype groupSpreadBits =
            frameLengthGroup.summary.maximumLengthBits - frameLengthGroup.summary.minimumLengthBits;
        if (groupSpreadBits > 1) {
            allGroupsTight = false;
            break;
        }
    }

    if (frameLengthGroups.size() >= 2 && frameLengthGroups.size() <= 8 && allGroupsTight) {
        return QStringLiteral("multi-type fixed");
    }

    const qsizetype variableSpreadThresholdBits = qMax<qsizetype>(8, frameLengthSummary.medianLengthBits / 8);
    if (frameLengthGroups.size() <= 8 && globalSpreadBits >= variableSpreadThresholdBits) {
        return QStringLiteral("variable-length");
    }

    if (frameLengthGroups.size() > 8 && globalSpreadBits >= variableSpreadThresholdBits) {
        int smallAdjacentGapCount = 0;
        for (int groupIndex = 1; groupIndex < frameLengthGroups.size(); ++groupIndex) {
            const qsizetype previousMedianLengthBits = frameLengthGroups.at(groupIndex - 1).summary.medianLengthBits;
            const qsizetype currentMedianLengthBits = frameLengthGroups.at(groupIndex).summary.medianLengthBits;
            if (currentMedianLengthBits - previousMedianLengthBits <= (kLengthClusterGapBits * 2)) {
                ++smallAdjacentGapCount;
            }
        }

        const double continuityRatio = frameLengthGroups.size() > 1
            ? static_cast<double>(smallAdjacentGapCount) / static_cast<double>(frameLengthGroups.size() - 1)
            : 0.0;
        if (continuityRatio >= 0.70) {
            return QStringLiteral("variable-length");
        }
    }

    return QStringLiteral("irregular");
}

[[nodiscard]] double protocolClassificationScore(const QString& protocolClassification) {
    if (protocolClassification == QStringLiteral("fixed-length")) {
        return 1.0;
    }
    if (protocolClassification == QStringLiteral("multi-type fixed")) {
        return 0.95;
    }
    if (protocolClassification == QStringLiteral("variable-length")) {
        return 0.85;
    }
    return 0.35;
}

[[nodiscard]] double sampleReliabilityScore(
    qsizetype frameCount
) {
    if (frameCount <= kSampleReliabilityFloorFrameCount) {
        return 0.0;
    }

    return qBound(
        0.0,
        static_cast<double>(frameCount - kSampleReliabilityFloorFrameCount) / kSampleReliabilityRampFrameCount,
        1.0
    );
}

[[nodiscard]] bool islandsOverlap(
    const EntropyIsland& leftIsland,
    const EntropyIsland& rightIsland,
    double toleranceBits
) {
    return static_cast<double>(leftIsland.endBit) + toleranceBits >= static_cast<double>(rightIsland.startBit)
        && static_cast<double>(rightIsland.endBit) + toleranceBits >= static_cast<double>(leftIsland.startBit);
}

[[nodiscard]] int sharedHeaderCliffCount(
    const QVector<FrameLengthGroup>& frameLengthGroups,
    double entropyThreshold
) {
    if (frameLengthGroups.isEmpty()) {
        return 0;
    }

    if (frameLengthGroups.size() == 1) {
        return strongEntropyIslandCount(frameLengthGroups.first().lowEntropyIslands, entropyThreshold);
    }

    int sharedHeaderLimitBits = std::numeric_limits<int>::max();
    for (const FrameLengthGroup& frameLengthGroup : frameLengthGroups) {
        sharedHeaderLimitBits = qMin(
            sharedHeaderLimitBits,
            static_cast<int>(frameLengthGroup.summary.medianLengthBits)
        );
    }
    if (sharedHeaderLimitBits <= 0) {
        return 0;
    }

    int sharedIslandCount = 0;
    const QVector<EntropyIsland>& referenceIslands = frameLengthGroups.first().lowEntropyIslands;
    for (const EntropyIsland& referenceIsland : referenceIslands) {
        if (!isStrongEntropyIsland(referenceIsland, entropyThreshold)) {
            continue;
        }
        if (referenceIsland.startBit >= sharedHeaderLimitBits) {
            continue;
        }

        EntropyIsland clippedReferenceIsland = referenceIsland;
        clippedReferenceIsland.endBit = qMin(clippedReferenceIsland.endBit, sharedHeaderLimitBits - 1);

        bool sharedAcrossAllGroups = true;
        for (int groupIndex = 1; groupIndex < frameLengthGroups.size(); ++groupIndex) {
            bool foundOverlap = false;
            for (const EntropyIsland& candidateIsland : frameLengthGroups.at(groupIndex).lowEntropyIslands) {
                if (!isStrongEntropyIsland(candidateIsland, entropyThreshold)) {
                    continue;
                }
                if (candidateIsland.startBit >= sharedHeaderLimitBits) {
                    continue;
                }

                EntropyIsland clippedCandidateIsland = candidateIsland;
                clippedCandidateIsland.endBit = qMin(clippedCandidateIsland.endBit, sharedHeaderLimitBits - 1);
                if (islandsOverlap(clippedReferenceIsland, clippedCandidateIsland, kSharedIslandToleranceBits)) {
                    foundOverlap = true;
                    break;
                }
            }

            if (!foundOverlap) {
                sharedAcrossAllGroups = false;
                break;
            }
        }

        if (sharedAcrossAllGroups) {
            ++sharedIslandCount;
        }
    }

    return sharedIslandCount;
}

[[nodiscard]] double validationBoostFromEntropyIslands(
    const QVector<FrameLengthGroup>& frameLengthGroups,
    int syncBitWidth,
    double entropyThreshold
) {
    if (frameLengthGroups.isEmpty()) {
        return 0.0;
    }

    double log10InverseRandomProbability = 0.0;
    int contributingGroupCount = 0;
    for (const FrameLengthGroup& frameLengthGroup : frameLengthGroups) {
        const int supportingFrameCount = qMax(0, frameLengthGroup.frameIndexes.size() - 1);
        if (supportingFrameCount <= 0) {
            continue;
        }

        double groupLog10Contribution = 0.0;
        for (const EntropyIsland& entropyIsland : frameLengthGroup.lowEntropyIslands) {
            if (!isValidationEntropyIsland(entropyIsland, syncBitWidth, entropyThreshold)) {
                continue;
            }

            const int islandWidthBits = entropyIslandWidthBits(entropyIsland);
            groupLog10Contribution +=
                static_cast<double>(islandWidthBits * supportingFrameCount) * std::log10(2.0);
        }

        if (groupLog10Contribution <= 0.0) {
            continue;
        }

        log10InverseRandomProbability += groupLog10Contribution;
        ++contributingGroupCount;
    }

    if (contributingGroupCount <= 0) {
        return 0.0;
    }

    const double averageLog10InverseProbability =
        log10InverseRandomProbability / static_cast<double>(contributingGroupCount);
    return qMin(30.0, 10.0 * averageLog10InverseProbability);
}

[[nodiscard]] std::optional<PreanalyzedCandidate> preanalyzePatternRecord(
    const data::ByteDataSource& dataSource,
    const PatternOccurrenceRecord& patternRecord,
    const BitstreamSyncDiscoverySettings& settings,
    OccurrenceSearchCache* occurrenceSearchCache,
    const std::atomic_bool* cancelRequested
) {
    const WidthAdjustedPattern adjustedPattern = determineAdjustedPattern(
        dataSource,
        patternRecord,
        settings,
        occurrenceSearchCache,
        cancelRequested
    );
    if (isCanceled(cancelRequested) || adjustedPattern.bitWidth <= 0 || adjustedPattern.matchStartBits.isEmpty()) {
        return std::nullopt;
    }

    PreanalyzedCandidate preanalyzedCandidate;
    preanalyzedCandidate.patternRecord = patternRecord;
    preanalyzedCandidate.adjustedPattern = adjustedPattern;
    preanalyzedCandidate.adjustedGapStatistics = summarizeGaps(adjustedPattern.matchStartBits);
    preanalyzedCandidate.frameSpans = buildFrameSpans(
        dataSource.bitCount(),
        adjustedPattern.matchStartBits,
        adjustedPattern.bitWidth
    );
    if (preanalyzedCandidate.frameSpans.size() < settings.minimumMatchCount) {
        return std::nullopt;
    }
    preanalyzedCandidate.frameLengthSummary = summarizeFrameLengths(preanalyzedCandidate.frameSpans);
    preanalyzedCandidate.familyKey = preanalyzedFamilyKey(
        adjustedPattern.matchStartBits,
        preanalyzedCandidate.adjustedGapStatistics.medianGapBits
    );
    preanalyzedCandidate.quickScore = quickPreanalysisScore(dataSource, preanalyzedCandidate, settings);
    return preanalyzedCandidate;
}

[[nodiscard]] std::optional<CandidateAnalysis> analyzePatternRecord(
    const data::ByteDataSource& dataSource,
    const PreanalyzedCandidate& preanalyzedCandidate,
    const BitstreamSyncDiscoverySettings& settings,
    const std::atomic_bool* cancelRequested
) {
    CandidateAnalysis analysis;
    BitstreamSyncDiscoveryCandidate& candidate = analysis.candidate;
    candidate.rawPattern.bitValue = preanalyzedCandidate.patternRecord.patternValue;
    candidate.rawPattern.bitWidth = preanalyzedCandidate.patternRecord.bitWidth;
    candidate.refinedPattern.bitValue = preanalyzedCandidate.adjustedPattern.patternValue;
    candidate.refinedPattern.bitWidth = preanalyzedCandidate.adjustedPattern.bitWidth;
    candidate.matchStartBits = preanalyzedCandidate.adjustedPattern.matchStartBits;
    candidate.medianGapBits = preanalyzedCandidate.adjustedGapStatistics.medianGapBits;
    candidate.gapCoefficientVariation = preanalyzedCandidate.adjustedGapStatistics.coefficientVariation;
    candidate.leftNeighborStability = qBound(
        0.0,
        1.0 - preanalyzedCandidate.adjustedPattern.leftBoundaryStatistics.entropy,
        1.0
    );
    candidate.rightNeighborStability = qBound(
        0.0,
        1.0 - preanalyzedCandidate.adjustedPattern.rightBoundaryStatistics.entropy,
        1.0
    );
    candidate.cliffSharpnessScore = preanalyzedCandidate.adjustedPattern.cliffSharpnessScore;
    candidate.frameSpans = preanalyzedCandidate.frameSpans;
    candidate.frameLengthSummary = preanalyzedCandidate.frameLengthSummary;

    QVector<FrameLengthGroup> frameLengthGroups = buildInitialFrameLengthGroups(candidate.frameSpans);
    mergeFrameLengthGroupsByEntropy(dataSource, candidate.frameSpans, &frameLengthGroups);
    const double entropySampleReliability = sampleReliabilityScore(candidate.frameSpans.size());

    int maximumStrongCliffCount = 0;
    int totalStrongCliffCount = 0;
    int totalRawCliffCount = 0;
    int totalValidationCliffCount = 0;
    double accumulatedStrongCliffWidthBits = 0.0;
    for (FrameLengthGroup& frameLengthGroup : frameLengthGroups) {
        ensureEntropyProfileForFrameGroup(dataSource, candidate.frameSpans, &frameLengthGroup);
        frameLengthGroup.lowEntropyIslands = findLowEntropyIslands(
            frameLengthGroup.entropyProfile,
            frameLengthGroup.frameIndexes,
            settings.entropyThreshold
        );
        const int groupStrongCliffCount = strongEntropyIslandCount(
            frameLengthGroup.lowEntropyIslands,
            settings.entropyThreshold
        );
        maximumStrongCliffCount = qMax(maximumStrongCliffCount, groupStrongCliffCount);
        totalStrongCliffCount += groupStrongCliffCount;
        totalRawCliffCount += frameLengthGroup.lowEntropyIslands.size();
        totalValidationCliffCount += validationEntropyIslandCount(
            frameLengthGroup.lowEntropyIslands,
            candidate.refinedPattern.bitWidth,
            settings.entropyThreshold
        );
        accumulatedStrongCliffWidthBits += averageStrongEntropyIslandWidth(
            frameLengthGroup.lowEntropyIslands,
            settings.entropyThreshold
        ) * static_cast<double>(groupStrongCliffCount);
    }

    candidate.detectedGroupCount = frameLengthGroups.size();
    candidate.detectedEntropyCliffCount = maximumStrongCliffCount;
    candidate.sharedHeaderCliffCount = sharedHeaderCliffCount(frameLengthGroups, settings.entropyThreshold);
    candidate.protocolClassification = classifyFrameDistribution(candidate.frameLengthSummary, frameLengthGroups);
    const double classificationScore = protocolClassificationScore(candidate.protocolClassification);

    double withinGroupTightness = 0.0;
    int totalGroupedFrameCount = 0;
    for (const FrameLengthGroup& frameLengthGroup : frameLengthGroups) {
        const double medianLengthBits = static_cast<double>(qMax<qsizetype>(1, frameLengthGroup.summary.medianLengthBits));
        const double spreadBits = static_cast<double>(
            frameLengthGroup.summary.maximumLengthBits - frameLengthGroup.summary.minimumLengthBits
        );
        const double groupTightness = qBound(0.0, 1.0 - (spreadBits / medianLengthBits), 1.0);
        withinGroupTightness += groupTightness * static_cast<double>(frameLengthGroup.frameIndexes.size());
        totalGroupedFrameCount += frameLengthGroup.frameIndexes.size();
    }
    withinGroupTightness = totalGroupedFrameCount > 0
        ? withinGroupTightness / static_cast<double>(totalGroupedFrameCount)
        : 0.0;

    const double baseSpacingScore = candidate.gapCoefficientVariation <= 0.0
        ? 1.0
        : qBound(0.0, 1.0 - qMin(candidate.gapCoefficientVariation, 1.0), 1.0);
    candidate.spacingRegularityScore = qBound(
        0.0,
        (0.55 * baseSpacingScore + 0.45 * withinGroupTightness) * (0.65 + 0.35 * classificationScore),
        1.0
    );

    candidate.patternUniquenessScore = preanalyzedCandidate.patternRecord.matchCount > 0
        ? qBound(
              0.0,
              1.0
                  - (static_cast<double>(preanalyzedCandidate.patternRecord.strongestNearMissCount)
                     / static_cast<double>(preanalyzedCandidate.patternRecord.matchCount)),
              1.0
          )
        : 0.0;

    const double expectedFrameCount = candidate.frameLengthSummary.medianLengthBits > 0
        ? static_cast<double>(dataSource.bitCount()) / static_cast<double>(candidate.frameLengthSummary.medianLengthBits)
        : static_cast<double>(candidate.frameSpans.size());
    candidate.occurrenceCountScore = expectedFrameCount > 0.0
        ? qBound(0.0, static_cast<double>(candidate.frameSpans.size()) / expectedFrameCount, 1.0) * entropySampleReliability
        : 0.0;

    const double representativeStrongCliffCount = frameLengthGroups.isEmpty()
        ? 0.0
        : static_cast<double>(totalStrongCliffCount) / static_cast<double>(frameLengthGroups.size());
    const double fragmentationScore = totalRawCliffCount > 0
        ? static_cast<double>(totalStrongCliffCount) / static_cast<double>(totalRawCliffCount)
        : 0.0;
    const double averageStrongCliffWidthBits = totalStrongCliffCount > 0
        ? accumulatedStrongCliffWidthBits / static_cast<double>(totalStrongCliffCount)
        : 0.0;
    const double widthQualityScore = qBound(
        0.0,
        (averageStrongCliffWidthBits - 1.0) / 7.0,
        1.0
    );
    candidate.distributedConstantsScore = qBound(
        0.0,
        0.45 * qBound(0.0, (representativeStrongCliffCount - 1.0) / 4.0, 1.0)
            + 0.35 * widthQualityScore
            + 0.20 * fragmentationScore,
        1.0
    ) * entropySampleReliability;

    if (frameLengthGroups.size() <= 1) {
        candidate.crossGroupAgreementScore = 1.0;
    } else {
        int sharedHeaderLimitBits = std::numeric_limits<int>::max();
        for (const FrameLengthGroup& frameLengthGroup : frameLengthGroups) {
            sharedHeaderLimitBits = qMin(
                sharedHeaderLimitBits,
                static_cast<int>(frameLengthGroup.summary.medianLengthBits)
            );
        }

        int referenceSharedHeaderIslands = 0;
        for (const EntropyIsland& entropyIsland : frameLengthGroups.first().lowEntropyIslands) {
            if (isStrongEntropyIsland(entropyIsland, settings.entropyThreshold)
                && entropyIsland.startBit < sharedHeaderLimitBits) {
                ++referenceSharedHeaderIslands;
            }
        }

        candidate.crossGroupAgreementScore = referenceSharedHeaderIslands > 0
            ? qBound(
                  0.0,
                  static_cast<double>(candidate.sharedHeaderCliffCount)
                      / static_cast<double>(referenceSharedHeaderIslands),
                  1.0
              )
            : 0.0;
    }
    candidate.crossGroupAgreementScore *= entropySampleReliability;

    const double validationCoverageScore = frameLengthGroups.isEmpty()
        ? 0.0
        : qBound(
              0.0,
              static_cast<double>(totalValidationCliffCount) / static_cast<double>(frameLengthGroups.size() * 3),
              1.0
          );
    candidate.validationBoostScore = validationBoostFromEntropyIslands(
        frameLengthGroups,
        candidate.refinedPattern.bitWidth,
        settings.entropyThreshold
    );
    candidate.validationBoostScore *=
        classificationScore
        * qBound(0.15, fragmentationScore, 1.0)
        * qBound(0.20, validationCoverageScore, 1.0)
        * entropySampleReliability;

    const double baseConfidenceScore = 100.0 * (
        0.20 * candidate.spacingRegularityScore
        + 0.25 * candidate.cliffSharpnessScore
        + 0.15 * candidate.patternUniquenessScore
        + 0.10 * candidate.occurrenceCountScore
        + 0.20 * candidate.distributedConstantsScore
        + 0.10 * candidate.crossGroupAgreementScore
    ) * classificationScore * entropySampleReliability;
    candidate.confidenceScore = qBound(0.0, baseConfidenceScore + candidate.validationBoostScore, 100.0);

    return analysis;
}

void emitProgress(
    const BitstreamSyncDiscoveryEngine::ProgressCallback& progressCallback,
    const QString& phaseLabel,
    const QString& detailLabel,
    int percentComplete,
    const BitstreamSyncDiscoveryCandidateList& partialCandidates
) {
    if (!progressCallback) {
        return;
    }

    BitstreamSyncDiscoveryProgressUpdate progressUpdate;
    progressUpdate.phaseLabel = phaseLabel;
    progressUpdate.detailLabel = detailLabel;
    progressUpdate.percentComplete = qBound(0, percentComplete, 100);
    progressCallback(progressUpdate, partialCandidates);
}

}  // namespace

BitstreamSyncDiscoveryCandidateList BitstreamSyncDiscoveryEngine::discover(
    const data::ByteDataSource& dataSource,
    const BitstreamSyncDiscoverySettings& settings,
    const std::atomic_bool* cancelRequested,
    const ProgressCallback& progressCallback,
    QString* errorMessage
) {
    if (errorMessage != nullptr) {
        errorMessage->clear();
    }

    if (!dataSource.hasData()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Load a file before running bitstream sync discovery.");
        }
        return {};
    }

    const BitstreamSyncDiscoverySettings effectiveSettings = normalizedSettings(settings);
    const qsizetype totalBitCount = dataSource.bitCount();
    if (totalBitCount < effectiveSettings.minimumPatternBits) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("The loaded file is too small for the requested sync width range.");
        }
        return {};
    }

    QVector<PatternOccurrenceRecord> patternRecords;
    const int widthCount =
        effectiveSettings.maximumPatternBits - effectiveSettings.minimumPatternBits + 1;
    const int retainedCandidateLimit = internalCandidateRetentionLimit(effectiveSettings);
    patternRecords.reserve(widthCount * retainedCandidateLimit);
    QSet<int> bitAlignedScannedBitWidths;
    bitAlignedScannedBitWidths.reserve(widthCount);
    QVector<int> refinementBitWidths;
    refinementBitWidths.reserve(widthCount);

    emitProgress(
        progressCallback,
        QStringLiteral("Phase 1/6: Candidate extraction"),
        QStringLiteral("Running byte-aligned seed scan for widths %1 to %2.")
            .arg(effectiveSettings.minimumPatternBits)
            .arg(effectiveSettings.maximumPatternBits),
        0,
        {}
    );

    int byteAlignedWidthCount = 0;
    for (int bitWidth = effectiveSettings.minimumPatternBits;
         bitWidth <= effectiveSettings.maximumPatternBits;
         ++bitWidth) {
        if (bitWidth % 8 == 0) {
            ++byteAlignedWidthCount;
        }
    }

    int scannedByteAlignedWidths = 0;
    for (int bitWidth = effectiveSettings.minimumPatternBits;
         bitWidth <= effectiveSettings.maximumPatternBits;
         ++bitWidth) {
        if (bitWidth % 8 != 0) {
            continue;
        }
        if (isCanceled(cancelRequested)) {
            return {};
        }

        const QVector<PatternOccurrenceRecord> widthRecords = collectCandidatePatternRecordsForWidth(
            dataSource,
            bitWidth,
            effectiveSettings,
            8,
            cancelRequested
        );
        patternRecords += widthRecords;
        ++scannedByteAlignedWidths;

        const int widthProgress = byteAlignedWidthCount > 0
            ? (scannedByteAlignedWidths * 18) / byteAlignedWidthCount
            : 18;
        emitProgress(
            progressCallback,
            QStringLiteral("Phase 1/6: Byte-aligned seed scan"),
            QStringLiteral("Scanned byte-aligned width %1 of %2 (%3-bit patterns, %4 surviving seeds).")
                .arg(scannedByteAlignedWidths)
                .arg(byteAlignedWidthCount)
                .arg(bitWidth)
                .arg(widthRecords.size()),
            widthProgress,
            {}
        );
    }

    QVector<PatternOccurrenceRecord> sortedByteAlignedRecords = patternRecords;
    std::sort(sortedByteAlignedRecords.begin(), sortedByteAlignedRecords.end(), patternRecordRanksBefore);

    bool hasStrongByteAlignedSeed = false;
    const qsizetype strongSeedMatchThreshold = strongSeedMatchThresholdForRefinement(effectiveSettings);
    for (const PatternOccurrenceRecord& patternRecord : sortedByteAlignedRecords) {
        if (patternRecord.matchCount >= strongSeedMatchThreshold
            && patternRecord.gapCoefficientVariation <= qMin(0.9, effectiveSettings.maximumGapCoefficientVariation)) {
            hasStrongByteAlignedSeed = true;
            break;
        }
    }

    if (hasStrongByteAlignedSeed) {
        const int seedLimit = qMin(sortedByteAlignedRecords.size(), retainedCandidateLimit * 2);
        for (int seedIndex = 0; seedIndex < seedLimit; ++seedIndex) {
            const int seedBitWidth = sortedByteAlignedRecords.at(seedIndex).bitWidth;
            for (int delta = -kRefinementHaloBits; delta <= kRefinementHaloBits; ++delta) {
                const int refinedBitWidth = seedBitWidth + delta;
                if (refinedBitWidth < effectiveSettings.minimumPatternBits
                    || refinedBitWidth > effectiveSettings.maximumPatternBits
                    || bitAlignedScannedBitWidths.contains(refinedBitWidth)) {
                    continue;
                }
                if (!refinementBitWidths.contains(refinedBitWidth)) {
                    refinementBitWidths.append(refinedBitWidth);
                }
            }
        }
    } else {
        for (int bitWidth = effectiveSettings.minimumPatternBits;
             bitWidth <= effectiveSettings.maximumPatternBits;
             ++bitWidth) {
            if (!bitAlignedScannedBitWidths.contains(bitWidth)) {
                refinementBitWidths.append(bitWidth);
            }
        }
    }

    emitProgress(
        progressCallback,
        QStringLiteral("Phase 1/6: Bit-aligned refinement"),
        hasStrongByteAlignedSeed
            ? QStringLiteral("Refining around %1 strong byte-aligned seed width%2.")
                  .arg(refinementBitWidths.size())
                  .arg(refinementBitWidths.size() == 1 ? QString() : QStringLiteral("s"))
            : QStringLiteral("No strong byte-aligned seed found; scanning the full bit-width range."),
        18,
        {}
    );

    for (int refinementIndex = 0; refinementIndex < refinementBitWidths.size(); ++refinementIndex) {
        const int bitWidth = refinementBitWidths.at(refinementIndex);
        if (isCanceled(cancelRequested)) {
            return {};
        }

        const QVector<PatternOccurrenceRecord> widthRecords = collectCandidatePatternRecordsForWidth(
            dataSource,
            bitWidth,
            effectiveSettings,
            1,
            cancelRequested
        );
        patternRecords += widthRecords;
        bitAlignedScannedBitWidths.insert(bitWidth);

        const int widthProgress = 18 + ((refinementIndex + 1) * 17) / qMax(1, refinementBitWidths.size());
        emitProgress(
            progressCallback,
            QStringLiteral("Phase 1/6: Bit-aligned refinement"),
            QStringLiteral("Refined width %1 of %2 (%3-bit patterns, %4 surviving seeds).")
                .arg(refinementIndex + 1)
                .arg(refinementBitWidths.size())
                .arg(bitWidth)
                .arg(widthRecords.size()),
            widthProgress,
            {}
        );
    }

    if (patternRecords.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral(
                "No pattern seeds survived the v2 extraction filters. "
                "Try lowering Min Frame Bits, Min matches, or Max Gap CV."
            );
        }
        return {};
    }

    std::sort(patternRecords.begin(), patternRecords.end(), patternRecordRanksBefore);
    patternRecords = screenedPatternRecordsForAnalysis(
        patternRecords,
        widthCount,
        effectiveSettings
    );
    OccurrenceSearchCache occurrenceSearchCache;
    occurrenceSearchCache.reserve(patternRecords.size() * 4);
    QVector<PreanalyzedCandidate> preanalyzedCandidates;
    preanalyzedCandidates.reserve(patternRecords.size());

    emitProgress(
        progressCallback,
        QStringLiteral("Phase 2/6: Candidate refinement screening"),
        QStringLiteral("Refining %1 shortlisted candidate%2.")
            .arg(patternRecords.size())
            .arg(patternRecords.size() == 1 ? QString() : QStringLiteral("s")),
        35,
        {}
    );

    for (int candidateIndex = 0; candidateIndex < patternRecords.size(); ++candidateIndex) {
        if (isCanceled(cancelRequested)) {
            return {};
        }

        const std::optional<PreanalyzedCandidate> preanalyzedCandidate = preanalyzePatternRecord(
            dataSource,
            patternRecords.at(candidateIndex),
            effectiveSettings,
            &occurrenceSearchCache,
            cancelRequested
        );
        if (isCanceled(cancelRequested)) {
            return {};
        }

        if (preanalyzedCandidate.has_value()) {
            preanalyzedCandidates.append(*preanalyzedCandidate);
        }

        const bool shouldRefreshPreanalysis =
            candidateIndex == 0
            || candidateIndex + 1 == patternRecords.size()
            || ((candidateIndex + 1) % kPreanalysisUpdateStride == 0);
        if (shouldRefreshPreanalysis) {
            const int preanalysisProgress = 35 + ((candidateIndex + 1) * 20) / qMax(1, patternRecords.size());
            const PatternOccurrenceRecord& patternRecord = patternRecords.at(candidateIndex);
            emitProgress(
                progressCallback,
                QStringLiteral("Phase 2/6: Candidate refinement screening"),
                QStringLiteral("Refined candidate %1 of %2 (%3 bits, %4 raw matches).")
                    .arg(candidateIndex + 1)
                    .arg(patternRecords.size())
                    .arg(patternRecord.bitWidth)
                    .arg(patternRecord.matchCount),
                preanalysisProgress,
                {}
            );
        }
    }

    int familyWinnerCount = 0;
    QVector<PreanalyzedCandidate> screenedPreanalyzedCandidates = screenedPreanalyzedCandidatesForFullAnalysis(
        preanalyzedCandidates,
        widthCount,
        effectiveSettings,
        &familyWinnerCount
    );
    QVector<CandidateAnalysis> analyses;
    analyses.reserve(screenedPreanalyzedCandidates.size());

    emitProgress(
        progressCallback,
        QStringLiteral("Phase 3-6/6: Candidate analysis"),
        QStringLiteral("Running full entropy analysis on %1 survivor%2.")
            .arg(screenedPreanalyzedCandidates.size())
            .arg(screenedPreanalyzedCandidates.size() == 1 ? QString() : QStringLiteral("s")),
        55,
        {}
    );

    bool stoppedFullAnalysisEarly = false;
    for (int candidateIndex = 0; candidateIndex < screenedPreanalyzedCandidates.size(); ++candidateIndex) {
        if (isCanceled(cancelRequested)) {
            return {};
        }

        const std::optional<CandidateAnalysis> analysis = analyzePatternRecord(
            dataSource,
            screenedPreanalyzedCandidates.at(candidateIndex),
            effectiveSettings,
            cancelRequested
        );
        if (isCanceled(cancelRequested)) {
            return {};
        }

        if (analysis.has_value()) {
            analyses.append(*analysis);
        }

        const bool shouldRefreshPartialRanking =
            candidateIndex == 0
            || candidateIndex + 1 == screenedPreanalyzedCandidates.size()
            || candidateIndex < 7
            || ((candidateIndex + 1) % kPartialRankingUpdateStride == 0);
        if (shouldRefreshPartialRanking) {
            BitstreamSyncDiscoveryCandidateList partialCandidates = deduplicatedCandidates(
                analyses,
                effectiveSettings.maximumResultCount
            );
            BitstreamSyncDiscoveryCandidateList filteredPartialCandidates;
            filteredPartialCandidates.reserve(partialCandidates.size());
            for (const BitstreamSyncDiscoveryCandidate& candidate : partialCandidates) {
                if (candidate.confidenceScore >= 20.0) {
                    filteredPartialCandidates.append(candidate);
                }
            }

            const int analysisProgress =
                55 + ((candidateIndex + 1) * 40) / qMax(1, screenedPreanalyzedCandidates.size());
            const PreanalyzedCandidate& preanalyzedCandidate = screenedPreanalyzedCandidates.at(candidateIndex);
            emitProgress(
                progressCallback,
                QStringLiteral("Phase 3-6/6: Candidate analysis"),
                QStringLiteral("Analyzed survivor %1 of %2 (%3 bits, %4 refined matches).")
                    .arg(candidateIndex + 1)
                    .arg(screenedPreanalyzedCandidates.size())
                    .arg(preanalyzedCandidate.adjustedPattern.bitWidth)
                    .arg(preanalyzedCandidate.frameSpans.size()),
                analysisProgress,
                filteredPartialCandidates
            );

            if (shouldStopFullAnalysisEarly(
                    filteredPartialCandidates,
                    screenedPreanalyzedCandidates,
                    candidateIndex + 1,
                    familyWinnerCount,
                    effectiveSettings)) {
                stoppedFullAnalysisEarly = true;
                break;
            }
        }
    }

    BitstreamSyncDiscoveryCandidateList finalCandidates = deduplicatedCandidates(
        analyses,
        effectiveSettings.maximumResultCount
    );
    BitstreamSyncDiscoveryCandidateList filteredFinalCandidates;
    filteredFinalCandidates.reserve(finalCandidates.size());
    for (const BitstreamSyncDiscoveryCandidate& candidate : finalCandidates) {
        if (candidate.confidenceScore >= 20.0) {
            filteredFinalCandidates.append(candidate);
        }
    }
    finalCandidates = filteredFinalCandidates;

    emitProgress(
        progressCallback,
        QStringLiteral("Phase 6/6: Final ranking"),
        stoppedFullAnalysisEarly
            ? QStringLiteral("Ranked %1 confident candidate%2 after early convergence.")
                  .arg(finalCandidates.size())
                  .arg(finalCandidates.size() == 1 ? QString() : QStringLiteral("s"))
            : QStringLiteral("Ranked %1 confident candidate%2.")
                  .arg(finalCandidates.size())
                  .arg(finalCandidates.size() == 1 ? QString() : QStringLiteral("s")),
        100,
        finalCandidates
    );

    if (finalCandidates.isEmpty() && errorMessage != nullptr) {
        *errorMessage = QStringLiteral(
            "No candidate exceeded the v2 confidence threshold. "
            "Try lowering Min Frame Bits or broadening the sync width range."
        );
    }

    return finalCandidates;
}

}  // namespace bitabyte::features::bitstream_sync_discovery
