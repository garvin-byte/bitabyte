#pragma once

#include <QHashFunctions>
#include <QMetaType>
#include <QString>
#include <QtTypes>
#include <QVector>

#include "features/framing/frame_layout.h"

namespace bitabyte::features::bitstream_sync_discovery {

inline constexpr int kMaximumSupportedSyncPatternBits = 128;

enum class BitstreamSyncDiscoverySearchEffort {
    Fast = 0,
    Balanced = 1,
    Exhaustive = 2,
};

struct BitstreamSyncDiscoverySettings {
    int minimumPatternBits = 3;
    int maximumPatternBits = 32;
    int minimumMatchCount = 4;
    int maximumCandidatesPerWidth = 3;
    int maximumResultCount = 40;
    int minimumExpectedFrameBits = 64;
    double entropyThreshold = 0.3;
    double maximumGapCoefficientVariation = 2.0;
    BitstreamSyncDiscoverySearchEffort searchEffort = BitstreamSyncDiscoverySearchEffort::Balanced;
};

struct BitstreamSyncDiscoveryProgressUpdate {
    QString phaseLabel;
    QString detailLabel;
    int percentComplete = 0;
};

struct BitPatternValue {
    quint64 upper = 0;
    quint64 lower = 0;

    [[nodiscard]] bool operator==(const BitPatternValue& other) const noexcept = default;
};

[[nodiscard]] inline size_t qHash(const BitPatternValue& bitPatternValue, size_t seed = 0) noexcept {
    seed ^= ::qHash(bitPatternValue.upper, seed);
    seed ^= ::qHash(bitPatternValue.lower, seed << 1);
    return seed;
}

struct DiscoveredBitPattern {
    BitPatternValue bitValue;
    int bitWidth = 0;
};

struct FrameLengthSummary {
    qsizetype minimumLengthBits = 0;
    qsizetype medianLengthBits = 0;
    qsizetype maximumLengthBits = 0;
    double averageLengthBits = 0.0;
};

struct BitstreamSyncDiscoveryCandidate {
    DiscoveredBitPattern rawPattern;
    DiscoveredBitPattern refinedPattern;
    QVector<qsizetype> matchStartBits;
    QVector<framing::FrameSpan> frameSpans;
    FrameLengthSummary frameLengthSummary;
    qsizetype medianGapBits = 0;
    double gapCoefficientVariation = 0.0;
    double leftNeighborStability = 1.0;
    double rightNeighborStability = 1.0;
    double spacingRegularityScore = 0.0;
    double cliffSharpnessScore = 0.0;
    double patternUniquenessScore = 0.0;
    double occurrenceCountScore = 0.0;
    double distributedConstantsScore = 0.0;
    double crossGroupAgreementScore = 0.0;
    double validationBoostScore = 0.0;
    double confidenceScore = 0.0;
    int detectedEntropyCliffCount = 0;
    int detectedGroupCount = 0;
    int sharedHeaderCliffCount = 0;
    QString displayFormat = QStringLiteral("binary");
    QString displayHexPattern;
    QString trailingBitsText;
    QString displayPattern;
    QString protocolClassification;
};

using BitstreamSyncDiscoveryCandidateList = QVector<BitstreamSyncDiscoveryCandidate>;

}  // namespace bitabyte::features::bitstream_sync_discovery

Q_DECLARE_METATYPE(bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryProgressUpdate)
Q_DECLARE_METATYPE(bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate)
Q_DECLARE_METATYPE(bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidateList)
