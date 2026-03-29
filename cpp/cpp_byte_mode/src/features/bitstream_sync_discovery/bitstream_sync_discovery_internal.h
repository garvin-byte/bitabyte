#pragma once

#include <QHash>
#include <QSet>
#include <QtGlobal>

#include <array>
#include <limits>

#include "features/bitstream_sync_discovery/bitstream_sync_discovery_types.h"

namespace bitabyte::features::bitstream_sync_discovery::detail {

inline constexpr int kEntropyWindowPaddingBits = 16;
inline constexpr int kLengthClusterGapBits = 8;
inline constexpr double kEntropyHighThreshold = 0.9;
inline constexpr double kSharedIslandToleranceBits = 2.0;
inline constexpr int kStrongIslandMinimumWidthBits = 2;
inline constexpr int kValidationIslandMinimumWidthBits = 4;
inline constexpr int kRefinementHaloBits = 8;
inline constexpr int kMinimumInternalCandidatesPerWidth = 24;
inline constexpr int kMaximumInternalCandidatesPerWidth = 48;
inline constexpr int kPartialRankingUpdateStride = 16;
inline constexpr int kPreanalysisUpdateStride = 24;
inline constexpr qsizetype kDominanceMaximumShiftBits = 12;
inline constexpr double kDominanceMinimumSmallerOverlapRatio = 0.95;
inline constexpr double kDominanceMinimumLargerOverlapRatio = 0.80;
inline constexpr int kPreanalysisAlternatesPerFamily = 5;
inline constexpr qsizetype kSampleReliabilityFloorFrameCount = 4;
inline constexpr double kSampleReliabilityRampFrameCount = 32.0;
inline constexpr int kPreanalyzedFamilySignatureDeltaCount = 7;
inline constexpr int kDeduplicationSignatureDeltaCount = 5;
inline constexpr qsizetype kCancelPollStrideBits = 1024;

using PatternValue = BitPatternValue;

struct PatternOccurrenceRecord {
    PatternValue patternValue;
    int bitWidth = 0;
    QVector<qsizetype> matchStartBits;
    qsizetype matchCount = 0;
    qsizetype medianGapBits = 0;
    double gapCoefficientVariation = 0.0;
    double preliminaryScore = 0.0;
    int strongestNearMissCount = 0;
    int outerLeftNearMissCount = 0;
    int outerRightNearMissCount = 0;
};

struct PatternAccumulator {
    qsizetype lastAcceptedStartBit = std::numeric_limits<qsizetype>::min() / 4;
    QVector<qsizetype> acceptedStartBits;
};

struct OccurrenceSearchCacheKey {
    PatternValue patternValue;
    int bitWidth = 0;
    int minimumGapBits = 0;

    [[nodiscard]] bool operator==(const OccurrenceSearchCacheKey& other) const noexcept {
        return patternValue == other.patternValue
            && bitWidth == other.bitWidth
            && minimumGapBits == other.minimumGapBits;
    }
};

using OccurrenceSearchCache = QHash<OccurrenceSearchCacheKey, QVector<qsizetype>>;

struct GapStatistics {
    qsizetype medianGapBits = 0;
    double meanGapBits = 0.0;
    double coefficientVariation = std::numeric_limits<double>::infinity();
};

struct RunLengthStatistics {
    int runCount = 0;
    int distinctRunLengthCount = 0;
    int minimumRunLength = 0;
    int maximumRunLength = 0;
};

struct SingleBitStatistics {
    double entropy = 1.0;
    double dominantFraction = 0.5;
    int sampleCount = 0;
};

struct EntropyIsland {
    int startBit = 0;
    int endBit = 0;
    double averageEntropy = 0.0;
    int supportingFrameCount = 0;
};

struct FrameLengthGroup {
    QVector<int> frameIndexes;
    FrameLengthSummary summary;
    QVector<double> entropyProfile;
    QVector<EntropyIsland> lowEntropyIslands;
};

struct WidthAdjustedPattern {
    PatternValue patternValue;
    int bitWidth = 0;
    QVector<qsizetype> matchStartBits;
    int leftTrimBits = 0;
    int rightTrimBits = 0;
    SingleBitStatistics leftBoundaryStatistics;
    SingleBitStatistics rightBoundaryStatistics;
    int leftTransitionWidth = 1;
    int rightTransitionWidth = 1;
    double cliffSharpnessScore = 0.0;
};

struct CandidateAnalysis {
    BitstreamSyncDiscoveryCandidate candidate;
};

struct PreanalyzedFamilyKey {
    qsizetype medianGapBits = 0;
    int signatureLength = 0;
    std::array<qsizetype, kPreanalyzedFamilySignatureDeltaCount> gapSignature{};

    [[nodiscard]] bool operator==(const PreanalyzedFamilyKey& other) const noexcept = default;
};

struct PreanalyzedCandidate {
    PatternOccurrenceRecord patternRecord;
    WidthAdjustedPattern adjustedPattern;
    GapStatistics adjustedGapStatistics;
    QVector<framing::FrameSpan> frameSpans;
    FrameLengthSummary frameLengthSummary;
    PreanalyzedFamilyKey familyKey;
    double quickScore = 0.0;
};

enum class TrimSide {
    Left,
    Right,
};

struct TrimEvaluation {
    bool shouldTrim = false;
    PatternValue patternValue;
    int bitWidth = 0;
    QVector<qsizetype> matchStartBits;
    SingleBitStatistics boundaryStatistics;
    qsizetype newMatchCount = 0;
    double newMatchRatio = 0.0;
    bool newBitsUniform = false;
    bool superset = false;
    double confidence = 0.0;
};

struct DominanceRelation {
    bool related = false;
    qsizetype deltaBits = 0;
    qsizetype sharedCount = 0;
    double smallerOverlapRatio = 0.0;
    double largerOverlapRatio = 0.0;
};

struct PatternContainmentRelation {
    bool related = false;
    bool leftContainsRight = false;
    bool rightContainsLeft = false;
    int offsetBits = 0;
};

struct BitCursor {
    const unsigned char* byteData = nullptr;
    qsizetype byteIndex = 0;
    int bitOffsetInByte = 0;

    [[nodiscard]] unsigned char currentBit() const {
        return static_cast<unsigned char>((byteData[byteIndex] >> (7 - bitOffsetInByte)) & 0x01);
    }

    void advance() {
        ++bitOffsetInByte;
        if (bitOffsetInByte >= 8) {
            bitOffsetInByte = 0;
            ++byteIndex;
        }
    }
};

struct PatternRecordKey {
    PatternValue patternValue;
    int bitWidth = 0;

    [[nodiscard]] bool operator==(const PatternRecordKey& other) const noexcept = default;
};

struct DeduplicationKey {
    qsizetype matchCount = 0;
    qsizetype firstStartBit = -1;
    qsizetype medianGapBits = 0;
    int signatureLength = 0;
    std::array<qsizetype, kDeduplicationSignatureDeltaCount> gapSignature{};

    [[nodiscard]] bool operator==(const DeduplicationKey& other) const noexcept = default;
};

[[nodiscard]] inline size_t qHash(const OccurrenceSearchCacheKey& key, size_t seed = 0) noexcept {
    seed ^= qHash(key.patternValue, seed);
    seed ^= ::qHash(key.bitWidth, seed << 1);
    seed ^= ::qHash(key.minimumGapBits, seed << 2);
    return seed;
}

[[nodiscard]] inline size_t qHash(const PatternRecordKey& key, size_t seed = 0) noexcept {
    seed ^= qHash(key.patternValue, seed);
    seed ^= ::qHash(key.bitWidth, seed << 1);
    return seed;
}

[[nodiscard]] inline size_t qHash(const PreanalyzedFamilyKey& key, size_t seed = 0) noexcept {
    seed ^= ::qHash(key.medianGapBits, seed);
    seed ^= ::qHash(key.signatureLength, seed << 1);
    for (int index = 0; index < key.signatureLength; ++index) {
        seed ^= ::qHash(key.gapSignature[static_cast<size_t>(index)], seed + static_cast<size_t>(index + 2));
    }
    return seed;
}

[[nodiscard]] inline size_t qHash(const DeduplicationKey& key, size_t seed = 0) noexcept {
    seed ^= ::qHash(key.matchCount, seed);
    seed ^= ::qHash(key.firstStartBit, seed << 1);
    seed ^= ::qHash(key.medianGapBits, seed << 2);
    seed ^= ::qHash(key.signatureLength, seed << 3);
    for (int index = 0; index < key.signatureLength; ++index) {
        seed ^= ::qHash(key.gapSignature[static_cast<size_t>(index)], seed + static_cast<size_t>(index + 4));
    }
    return seed;
}

[[nodiscard]] inline PatternValue patternValueMask(int bitWidth) {
    if (bitWidth <= 0) {
        return {};
    }
    if (bitWidth >= kMaximumSupportedSyncPatternBits) {
        return {std::numeric_limits<quint64>::max(), std::numeric_limits<quint64>::max()};
    }
    if (bitWidth > 64) {
        const int upperBitCount = bitWidth - 64;
        return {
            upperBitCount >= 64
                ? std::numeric_limits<quint64>::max()
                : ((quint64{1} << upperBitCount) - 1),
            std::numeric_limits<quint64>::max(),
        };
    }
    if (bitWidth == 64) {
        return {0, std::numeric_limits<quint64>::max()};
    }
    return {0, (quint64{1} << bitWidth) - 1};
}

[[nodiscard]] inline PatternValue patternValueShiftLeft(PatternValue patternValue, int shiftBits) {
    if (shiftBits <= 0) {
        return patternValue;
    }
    if (shiftBits >= kMaximumSupportedSyncPatternBits) {
        return {};
    }
    if (shiftBits >= 64) {
        return {patternValue.lower << (shiftBits - 64), 0};
    }
    return {
        (patternValue.upper << shiftBits) | (patternValue.lower >> (64 - shiftBits)),
        patternValue.lower << shiftBits,
    };
}

[[nodiscard]] inline PatternValue patternValueShiftRight(PatternValue patternValue, int shiftBits) {
    if (shiftBits <= 0) {
        return patternValue;
    }
    if (shiftBits >= kMaximumSupportedSyncPatternBits) {
        return {};
    }
    if (shiftBits >= 64) {
        return {0, patternValue.upper >> (shiftBits - 64)};
    }
    return {
        patternValue.upper >> shiftBits,
        (patternValue.lower >> shiftBits) | (patternValue.upper << (64 - shiftBits)),
    };
}

[[nodiscard]] inline PatternValue patternValueAnd(PatternValue leftValue, PatternValue rightValue) {
    return {leftValue.upper & rightValue.upper, leftValue.lower & rightValue.lower};
}

[[nodiscard]] inline PatternValue patternValueMasked(PatternValue patternValue, int bitWidth) {
    return patternValueAnd(patternValue, patternValueMask(bitWidth));
}

[[nodiscard]] inline PatternValue patternValueAppendBit(PatternValue patternValue, unsigned char bitValue) {
    patternValue = patternValueShiftLeft(patternValue, 1);
    patternValue.lower |= static_cast<quint64>(bitValue & 0x01);
    return patternValue;
}

[[nodiscard]] inline bool patternValueBitAt(const PatternValue& patternValue, int bitIndex) {
    if (bitIndex < 0 || bitIndex >= kMaximumSupportedSyncPatternBits) {
        return false;
    }
    if (bitIndex >= 64) {
        return ((patternValue.upper >> (bitIndex - 64)) & quint64{1}) != 0;
    }
    return ((patternValue.lower >> bitIndex) & quint64{1}) != 0;
}

[[nodiscard]] inline PatternValue patternValueWithFlippedBit(PatternValue patternValue, int bitIndex) {
    if (bitIndex < 0 || bitIndex >= kMaximumSupportedSyncPatternBits) {
        return patternValue;
    }
    if (bitIndex >= 64) {
        patternValue.upper ^= (quint64{1} << (bitIndex - 64));
    } else {
        patternValue.lower ^= (quint64{1} << bitIndex);
    }
    return patternValue;
}

[[nodiscard]] inline bool patternValueLess(const PatternValue& leftValue, const PatternValue& rightValue) {
    if (leftValue.upper != rightValue.upper) {
        return leftValue.upper < rightValue.upper;
    }
    return leftValue.lower < rightValue.lower;
}

[[nodiscard]] inline PatternValue extractPatternSegment(
    PatternValue patternValue,
    int patternBitWidth,
    int offsetBits,
    int segmentBitWidth
) {
    const int shiftBits = patternBitWidth - (offsetBits + segmentBitWidth);
    return patternValueMasked(patternValueShiftRight(patternValue, shiftBits), segmentBitWidth);
}

[[nodiscard]] inline PatternContainmentRelation patternContainmentRelation(
    PatternValue leftPatternValue,
    int leftPatternBitWidth,
    PatternValue rightPatternValue,
    int rightPatternBitWidth
) {
    PatternContainmentRelation relation;
    if (leftPatternBitWidth <= 0
        || rightPatternBitWidth <= 0
        || leftPatternBitWidth == rightPatternBitWidth) {
        return relation;
    }

    const bool leftIsLonger = leftPatternBitWidth > rightPatternBitWidth;
    const PatternValue longerPatternValue = leftIsLonger ? leftPatternValue : rightPatternValue;
    const int longerPatternBitWidth = leftIsLonger ? leftPatternBitWidth : rightPatternBitWidth;
    const PatternValue shorterPatternValue = leftIsLonger ? rightPatternValue : leftPatternValue;
    const int shorterPatternBitWidth = leftIsLonger ? rightPatternBitWidth : leftPatternBitWidth;

    for (int offsetBits = 0; offsetBits <= longerPatternBitWidth - shorterPatternBitWidth; ++offsetBits) {
        if (extractPatternSegment(longerPatternValue, longerPatternBitWidth, offsetBits, shorterPatternBitWidth)
            != shorterPatternValue) {
            continue;
        }

        relation.related = true;
        relation.offsetBits = offsetBits;
        relation.leftContainsRight = leftIsLonger;
        relation.rightContainsLeft = !leftIsLonger;
        return relation;
    }

    return relation;
}

[[nodiscard]] inline bool supportsAreComparable(
    qsizetype leftCount,
    qsizetype rightCount,
    qsizetype leftMedianGapBits,
    qsizetype rightMedianGapBits
) {
    const qsizetype largerCount = qMax(leftCount, rightCount);
    const qsizetype smallerCount = qMin(leftCount, rightCount);
    if (smallerCount <= 0) {
        return false;
    }

    const qsizetype countDifference = largerCount - smallerCount;
    const qsizetype allowedCountDifference = qMax<qsizetype>(4, largerCount / 50);
    if (countDifference > allowedCountDifference) {
        return false;
    }

    const qsizetype largerGapBits = qMax(leftMedianGapBits, rightMedianGapBits);
    const qsizetype smallerGapBits = qMin(leftMedianGapBits, rightMedianGapBits);
    const qsizetype gapDifference = largerGapBits - smallerGapBits;
    const qsizetype allowedGapDifference = qMax<qsizetype>(16, largerGapBits / 20);
    return gapDifference <= allowedGapDifference;
}

[[nodiscard]] inline int canonicalWidthPreferenceScore(int bitWidth) {
    if (bitWidth % 8 == 0) {
        return 3;
    }
    if (bitWidth % 4 == 0) {
        return 2;
    }
    if (bitWidth % 2 == 0) {
        return 1;
    }
    return 0;
}

[[nodiscard]] inline bool candidatesShareExactMatchFamily(
    const BitstreamSyncDiscoveryCandidate& leftCandidate,
    const BitstreamSyncDiscoveryCandidate& rightCandidate
) {
    return leftCandidate.matchStartBits == rightCandidate.matchStartBits
        && leftCandidate.medianGapBits == rightCandidate.medianGapBits;
}

[[nodiscard]] inline int canonicalWidthPreferenceScore(const BitstreamSyncDiscoveryCandidate& candidate) {
    return canonicalWidthPreferenceScore(candidate.refinedPattern.bitWidth);
}

[[nodiscard]] inline int internalCandidateRetentionLimit(const BitstreamSyncDiscoverySettings& settings) {
    switch (settings.searchEffort) {
    case BitstreamSyncDiscoverySearchEffort::Fast:
        return qBound(
            settings.maximumCandidatesPerWidth,
            qMax(settings.maximumCandidatesPerWidth * 4, 16),
            32
        );
    case BitstreamSyncDiscoverySearchEffort::Exhaustive:
        return qBound(
            settings.maximumCandidatesPerWidth + 2,
            qMax(settings.maximumCandidatesPerWidth * 8, 32),
            72
        );
    case BitstreamSyncDiscoverySearchEffort::Balanced:
    default:
        return qBound(
            settings.maximumCandidatesPerWidth,
            qMax(settings.maximumCandidatesPerWidth * 6, kMinimumInternalCandidatesPerWidth),
            kMaximumInternalCandidatesPerWidth
        );
    }
}

[[nodiscard]] inline int screenedPatternPerWidthKeepCount(const BitstreamSyncDiscoverySettings& settings) {
    switch (settings.searchEffort) {
    case BitstreamSyncDiscoverySearchEffort::Fast:
        return qBound(2, settings.maximumCandidatesPerWidth - 1, 3);
    case BitstreamSyncDiscoverySearchEffort::Exhaustive:
        return qBound(3, settings.maximumCandidatesPerWidth + 2, 8);
    case BitstreamSyncDiscoverySearchEffort::Balanced:
    default:
        return qBound(2, settings.maximumCandidatesPerWidth, 4);
    }
}

[[nodiscard]] inline int screenedPatternGlobalAnalysisBudget(
    int widthCount,
    const BitstreamSyncDiscoverySettings& settings,
    int candidateCount
) {
    switch (settings.searchEffort) {
    case BitstreamSyncDiscoverySearchEffort::Fast:
        return qMin(candidateCount, qMax(widthCount * 4, settings.maximumResultCount * 3));
    case BitstreamSyncDiscoverySearchEffort::Exhaustive:
        return qMin(candidateCount, qMax(widthCount * 10, settings.maximumResultCount * 8));
    case BitstreamSyncDiscoverySearchEffort::Balanced:
    default:
        return qMin(candidateCount, qMax(widthCount * 6, settings.maximumResultCount * 4));
    }
}

[[nodiscard]] inline int screenedPreanalysisPerWidthKeepCount(const BitstreamSyncDiscoverySettings& settings) {
    switch (settings.searchEffort) {
    case BitstreamSyncDiscoverySearchEffort::Fast:
        return qBound(2, settings.maximumCandidatesPerWidth, 4);
    case BitstreamSyncDiscoverySearchEffort::Exhaustive:
        return qBound(4, settings.maximumCandidatesPerWidth + 3, 10);
    case BitstreamSyncDiscoverySearchEffort::Balanced:
    default:
        return qBound(3, settings.maximumCandidatesPerWidth + 1, 6);
    }
}

[[nodiscard]] inline int screenedPreanalysisGlobalBudget(
    int widthCount,
    const BitstreamSyncDiscoverySettings& settings,
    int candidateCount
) {
    switch (settings.searchEffort) {
    case BitstreamSyncDiscoverySearchEffort::Fast:
        return qMin(candidateCount, qMax(widthCount * 4, settings.maximumResultCount * 4));
    case BitstreamSyncDiscoverySearchEffort::Exhaustive:
        return qMin(candidateCount, qMax(widthCount * 10, settings.maximumResultCount * 9));
    case BitstreamSyncDiscoverySearchEffort::Balanced:
    default:
        return qMin(candidateCount, qMax(widthCount * 5, settings.maximumResultCount * 5));
    }
}

[[nodiscard]] inline qsizetype strongSeedMatchThresholdForRefinement(const BitstreamSyncDiscoverySettings& settings) {
    switch (settings.searchEffort) {
    case BitstreamSyncDiscoverySearchEffort::Fast:
        return 24;
    case BitstreamSyncDiscoverySearchEffort::Exhaustive:
        return 12;
    case BitstreamSyncDiscoverySearchEffort::Balanced:
    default:
        return 16;
    }
}

[[nodiscard]] inline bool allowsEarlyStop(const BitstreamSyncDiscoverySettings& settings) {
    return settings.searchEffort != BitstreamSyncDiscoverySearchEffort::Exhaustive;
}

[[nodiscard]] inline int minimumAnalyzedCountForEarlyStop(
    int familyWinnerCount,
    const BitstreamSyncDiscoverySettings& settings
) {
    switch (settings.searchEffort) {
    case BitstreamSyncDiscoverySearchEffort::Fast:
        return qMin(familyWinnerCount, qMax(4, settings.maximumResultCount / 10));
    case BitstreamSyncDiscoverySearchEffort::Exhaustive:
        return qMin(familyWinnerCount, qMax(8, settings.maximumResultCount / 6));
    case BitstreamSyncDiscoverySearchEffort::Balanced:
    default:
        return qMin(familyWinnerCount, qMax(6, settings.maximumResultCount / 8));
    }
}

[[nodiscard]] inline qsizetype requiredFrameCountForEarlyStop(const BitstreamSyncDiscoverySettings& settings) {
    switch (settings.searchEffort) {
    case BitstreamSyncDiscoverySearchEffort::Fast:
        return 48;
    case BitstreamSyncDiscoverySearchEffort::Exhaustive:
        return 96;
    case BitstreamSyncDiscoverySearchEffort::Balanced:
    default:
        return 64;
    }
}

[[nodiscard]] bool patternRecordRanksBefore(
    const PatternOccurrenceRecord& leftRecord,
    const PatternOccurrenceRecord& rightRecord
);

[[nodiscard]] PreanalyzedFamilyKey preanalyzedFamilyKey(
    const QVector<qsizetype>& matchStartBits,
    qsizetype medianGapBits
);

[[nodiscard]] QVector<PreanalyzedCandidate> screenedPreanalyzedCandidatesForFullAnalysis(
    const QVector<PreanalyzedCandidate>& preanalyzedCandidates,
    int widthCount,
    const BitstreamSyncDiscoverySettings& settings,
    int* familyWinnerCount
);

[[nodiscard]] QVector<PatternOccurrenceRecord> screenedPatternRecordsForAnalysis(
    const QVector<PatternOccurrenceRecord>& patternRecords,
    int widthCount,
    const BitstreamSyncDiscoverySettings& settings
);

[[nodiscard]] bool shouldStopFullAnalysisEarly(
    const BitstreamSyncDiscoveryCandidateList& partialCandidates,
    const QVector<PreanalyzedCandidate>& screenedPreanalyzedCandidates,
    int analyzedCount,
    int familyWinnerCount,
    const BitstreamSyncDiscoverySettings& settings
);

[[nodiscard]] BitstreamSyncDiscoveryCandidateList deduplicatedCandidates(
    const QVector<CandidateAnalysis>& analyses,
    int maximumResultCount
);

}  // namespace bitabyte::features::bitstream_sync_discovery::detail
