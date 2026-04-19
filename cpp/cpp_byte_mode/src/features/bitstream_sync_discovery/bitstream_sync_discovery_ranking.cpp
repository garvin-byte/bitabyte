#include "features/bitstream_sync_discovery/bitstream_sync_discovery_internal.h"

#include <algorithm>
#include <cmath>

namespace bitabyte::features::bitstream_sync_discovery::detail {
namespace {

// Thresholds for deciding when a longer contained alias should be preferred
// over a shorter one with a higher raw confidence score.
constexpr double kAliasMinimumLongerCoverageRatio  = 0.98;
constexpr double kAliasMinimumShorterCoverageRatio = 0.95;
constexpr double kAliasMaximumAllowableScoreGap    = 15.0;

// When comparing a wider pattern against a narrower one that dominates it,
// suppress the width-preference bias if the narrower pattern leads by more
// than this amount in confidence score.
constexpr double kDominanceMaximumScoreGapForWidthBias = 4.0;

[[nodiscard]] PatternRecordKey patternRecordKey(const PatternOccurrenceRecord& patternRecord) {
    return {
        patternRecord.patternValue,
        patternRecord.bitWidth,
    };
}

[[nodiscard]] double quickScreenScore(const PatternOccurrenceRecord& patternRecord) {
    const double uniquenessScore = patternRecord.matchCount > 0
        ? qBound(
              0.0,
              1.0 - (static_cast<double>(patternRecord.strongestNearMissCount) / static_cast<double>(patternRecord.matchCount)),
              1.0
          )
        : 0.0;
    const double widthPreferenceScore =
        0.20 * static_cast<double>(canonicalWidthPreferenceScore(patternRecord.bitWidth))
        + 0.20 * qBound(0.0, static_cast<double>(patternRecord.bitWidth) / 32.0, 1.0);
    return patternRecord.preliminaryScore * (0.75 + 0.25 * uniquenessScore) + widthPreferenceScore;
}

[[nodiscard]] bool quickScreenRanksBefore(
    const PatternOccurrenceRecord& leftRecord,
    const PatternOccurrenceRecord& rightRecord
) {
    const double leftScore = quickScreenScore(leftRecord);
    const double rightScore = quickScreenScore(rightRecord);
    if (!qFuzzyCompare(leftScore, rightScore)) {
        return leftScore > rightScore;
    }
    return patternRecordRanksBefore(leftRecord, rightRecord);
}

[[nodiscard]] qsizetype sharedShiftedStartCount(
    const QVector<qsizetype>& leftStartBits,
    const QVector<qsizetype>& rightStartBits,
    qsizetype deltaBits
);

struct ContainedAliasRelation {
    bool related = false;
    bool leftIsLonger = false;
    qsizetype sharedCount = 0;
    double longerCoverageRatio = 0.0;
    double shorterCoverageRatio = 0.0;
};

[[nodiscard]] ContainedAliasRelation containedAliasRelation(
    const BitstreamSyncDiscoveryCandidate& leftCandidate,
    const BitstreamSyncDiscoveryCandidate& rightCandidate
) {
    ContainedAliasRelation relation;
    if (leftCandidate.matchStartBits.isEmpty() || rightCandidate.matchStartBits.isEmpty()) {
        return relation;
    }

    const PatternContainmentRelation containmentRelation = patternContainmentRelation(
        leftCandidate.refinedPattern.bitValue,
        leftCandidate.refinedPattern.bitWidth,
        rightCandidate.refinedPattern.bitValue,
        rightCandidate.refinedPattern.bitWidth
    );
    if (!containmentRelation.related) {
        return relation;
    }

    qsizetype sharedCount = 0;
    qsizetype longerCount = 0;
    qsizetype shorterCount = 0;
    if (containmentRelation.leftContainsRight) {
        sharedCount = sharedShiftedStartCount(
            leftCandidate.matchStartBits,
            rightCandidate.matchStartBits,
            containmentRelation.offsetBits
        );
        longerCount = leftCandidate.matchStartBits.size();
        shorterCount = rightCandidate.matchStartBits.size();
        relation.leftIsLonger = true;
    } else if (containmentRelation.rightContainsLeft) {
        sharedCount = sharedShiftedStartCount(
            rightCandidate.matchStartBits,
            leftCandidate.matchStartBits,
            containmentRelation.offsetBits
        );
        longerCount = rightCandidate.matchStartBits.size();
        shorterCount = leftCandidate.matchStartBits.size();
        relation.leftIsLonger = false;
    } else {
        return relation;
    }

    if (sharedCount <= 0 || longerCount <= 0 || shorterCount <= 0) {
        return relation;
    }

    relation.related = true;
    relation.sharedCount = sharedCount;
    relation.longerCoverageRatio =
        static_cast<double>(sharedCount) / static_cast<double>(longerCount);
    relation.shorterCoverageRatio =
        static_cast<double>(sharedCount) / static_cast<double>(shorterCount);
    return relation;
}

[[nodiscard]] bool shouldPreferLongerContainedAlias(
    const BitstreamSyncDiscoveryCandidate& leftCandidate,
    const BitstreamSyncDiscoveryCandidate& rightCandidate,
    const ContainedAliasRelation& aliasRelation
) {
    if (!aliasRelation.related) {
        return false;
    }

    const BitstreamSyncDiscoveryCandidate& longerCandidate =
        aliasRelation.leftIsLonger ? leftCandidate : rightCandidate;
    const BitstreamSyncDiscoveryCandidate& shorterCandidate =
        aliasRelation.leftIsLonger ? rightCandidate : leftCandidate;

    const bool bothHaveCliffEvidence =
        longerCandidate.detectedEntropyCliffCount > 0
        && shorterCandidate.detectedEntropyCliffCount > 0;
    if (!bothHaveCliffEvidence) {
        return false;
    }

    if (aliasRelation.longerCoverageRatio < kAliasMinimumLongerCoverageRatio
        || aliasRelation.shorterCoverageRatio < kAliasMinimumShorterCoverageRatio) {
        return false;
    }

    const double scoreGap = shorterCandidate.confidenceScore - longerCandidate.confidenceScore;
    if (scoreGap > kAliasMaximumAllowableScoreGap) {
        return false;
    }

    return true;
}

[[nodiscard]] bool shouldRescueWiderPatternRecord(
    const PatternOccurrenceRecord& candidate,
    const PatternOccurrenceRecord& keptRecord
) {
    if (candidate.bitWidth <= keptRecord.bitWidth) {
        return false;
    }

    if (!supportsAreComparable(
            candidate.matchCount,
            keptRecord.matchCount,
            candidate.medianGapBits,
            keptRecord.medianGapBits)) {
        return false;
    }

    const PatternContainmentRelation containmentRelation = patternContainmentRelation(
        candidate.patternValue,
        candidate.bitWidth,
        keptRecord.patternValue,
        keptRecord.bitWidth
    );
    if (!containmentRelation.related) {
        return false;
    }

    return quickScreenScore(candidate) >= quickScreenScore(keptRecord) - 8.0;
}

[[nodiscard]] bool preanalyzedCandidateRanksBefore(
    const PreanalyzedCandidate& leftCandidate,
    const PreanalyzedCandidate& rightCandidate
) {
    const PatternContainmentRelation containmentRelation = patternContainmentRelation(
        leftCandidate.adjustedPattern.patternValue,
        leftCandidate.adjustedPattern.bitWidth,
        rightCandidate.adjustedPattern.patternValue,
        rightCandidate.adjustedPattern.bitWidth
    );
    const bool comparableContainedFamily =
        containmentRelation.related
        && supportsAreComparable(
            leftCandidate.frameSpans.size(),
            rightCandidate.frameSpans.size(),
            leftCandidate.adjustedGapStatistics.medianGapBits,
            rightCandidate.adjustedGapStatistics.medianGapBits
        );
    if (comparableContainedFamily && std::abs(leftCandidate.quickScore - rightCandidate.quickScore) < 8.0) {
        return leftCandidate.adjustedPattern.bitWidth > rightCandidate.adjustedPattern.bitWidth;
    }

    if (leftCandidate.familyKey == rightCandidate.familyKey
        && std::abs(leftCandidate.quickScore - rightCandidate.quickScore) < 8.0) {
        const int leftCanonicalPreference = canonicalWidthPreferenceScore(leftCandidate.adjustedPattern.bitWidth);
        const int rightCanonicalPreference = canonicalWidthPreferenceScore(rightCandidate.adjustedPattern.bitWidth);
        if (leftCanonicalPreference != rightCanonicalPreference) {
            return leftCanonicalPreference > rightCanonicalPreference;
        }
        if (leftCandidate.adjustedPattern.bitWidth != rightCandidate.adjustedPattern.bitWidth) {
            return leftCandidate.adjustedPattern.bitWidth > rightCandidate.adjustedPattern.bitWidth;
        }
    }

    if (std::abs(leftCandidate.quickScore - rightCandidate.quickScore) > 0.001) {
        return leftCandidate.quickScore > rightCandidate.quickScore;
    }
    if (leftCandidate.frameSpans.size() != rightCandidate.frameSpans.size()) {
        return leftCandidate.frameSpans.size() > rightCandidate.frameSpans.size();
    }
    if (leftCandidate.adjustedPattern.bitWidth != rightCandidate.adjustedPattern.bitWidth) {
        return leftCandidate.adjustedPattern.bitWidth > rightCandidate.adjustedPattern.bitWidth;
    }
    return patternRecordRanksBefore(leftCandidate.patternRecord, rightCandidate.patternRecord);
}

[[nodiscard]] bool shouldReplacePreanalyzedFamilyWinner(
    const PreanalyzedCandidate& candidate,
    const PreanalyzedCandidate& currentWinner
) {
    if (preanalyzedCandidateRanksBefore(candidate, currentWinner)) {
        return true;
    }

    if (candidate.familyKey != currentWinner.familyKey) {
        return false;
    }

    if (std::abs(candidate.quickScore - currentWinner.quickScore) >= 8.0) {
        return false;
    }

    const int widthDeltaBits = candidate.adjustedPattern.bitWidth - currentWinner.adjustedPattern.bitWidth;
    if (widthDeltaBits > 0) {
        return true;
    }
    return candidate.frameSpans.size() > currentWinner.frameSpans.size()
        && candidate.quickScore >= currentWinner.quickScore - 2.0;
}

[[nodiscard]] qsizetype sharedShiftedStartCount(
    const QVector<qsizetype>& leftStartBits,
    const QVector<qsizetype>& rightStartBits,
    qsizetype deltaBits
) {
    qsizetype sharedCount = 0;
    int leftIndex = 0;
    int rightIndex = 0;
    while (leftIndex < leftStartBits.size() && rightIndex < rightStartBits.size()) {
        const qsizetype shiftedLeftStartBit = leftStartBits.at(leftIndex) + deltaBits;
        const qsizetype rightStartBit = rightStartBits.at(rightIndex);
        if (shiftedLeftStartBit == rightStartBit) {
            ++sharedCount;
            ++leftIndex;
            ++rightIndex;
            continue;
        }
        if (shiftedLeftStartBit < rightStartBit) {
            ++leftIndex;
        } else {
            ++rightIndex;
        }
    }
    return sharedCount;
}

[[nodiscard]] DominanceRelation dominanceRelation(
    const BitstreamSyncDiscoveryCandidate& leftCandidate,
    const BitstreamSyncDiscoveryCandidate& rightCandidate
) {
    DominanceRelation relation;
    if (leftCandidate.matchStartBits.isEmpty() || rightCandidate.matchStartBits.isEmpty()) {
        return relation;
    }

    qsizetype bestSharedCount = 0;
    qsizetype bestDeltaBits = 0;
    for (qsizetype deltaBits = -kDominanceMaximumShiftBits;
         deltaBits <= kDominanceMaximumShiftBits;
         ++deltaBits) {
        const qsizetype sharedCount = sharedShiftedStartCount(
            leftCandidate.matchStartBits,
            rightCandidate.matchStartBits,
            deltaBits
        );
        if (sharedCount > bestSharedCount) {
            bestSharedCount = sharedCount;
            bestDeltaBits = deltaBits;
        }
    }

    if (bestSharedCount <= 0) {
        return relation;
    }

    const qsizetype smallerCount = qMin(leftCandidate.matchStartBits.size(), rightCandidate.matchStartBits.size());
    const qsizetype largerCount = qMax(leftCandidate.matchStartBits.size(), rightCandidate.matchStartBits.size());
    relation.sharedCount = bestSharedCount;
    relation.deltaBits = bestDeltaBits;
    relation.smallerOverlapRatio = smallerCount > 0
        ? static_cast<double>(bestSharedCount) / static_cast<double>(smallerCount)
        : 0.0;
    relation.largerOverlapRatio = largerCount > 0
        ? static_cast<double>(bestSharedCount) / static_cast<double>(largerCount)
        : 0.0;
    relation.related =
        relation.smallerOverlapRatio >= kDominanceMinimumSmallerOverlapRatio
        && relation.largerOverlapRatio >= kDominanceMinimumLargerOverlapRatio;
    return relation;
}

[[nodiscard]] bool candidateRanksBefore(
    const BitstreamSyncDiscoveryCandidate& leftCandidate,
    const BitstreamSyncDiscoveryCandidate& rightCandidate
) {
    const ContainedAliasRelation aliasRelation = containedAliasRelation(leftCandidate, rightCandidate);
    if (shouldPreferLongerContainedAlias(leftCandidate, rightCandidate, aliasRelation)) {
        return aliasRelation.leftIsLonger;
    }

    const PatternContainmentRelation containmentRelation = patternContainmentRelation(
        leftCandidate.refinedPattern.bitValue,
        leftCandidate.refinedPattern.bitWidth,
        rightCandidate.refinedPattern.bitValue,
        rightCandidate.refinedPattern.bitWidth
    );
    const bool comparableContainedFamily =
        containmentRelation.related
        && supportsAreComparable(
            leftCandidate.matchStartBits.size(),
            rightCandidate.matchStartBits.size(),
            leftCandidate.medianGapBits,
            rightCandidate.medianGapBits
        );
    if (comparableContainedFamily && std::abs(leftCandidate.confidenceScore - rightCandidate.confidenceScore) < 8.0) {
        return leftCandidate.refinedPattern.bitWidth > rightCandidate.refinedPattern.bitWidth;
    }

    const bool sameExactMatchFamily = candidatesShareExactMatchFamily(leftCandidate, rightCandidate);
    const DominanceRelation relatedFamilyRelation = dominanceRelation(leftCandidate, rightCandidate);
    const bool sameRelatedFamily = relatedFamilyRelation.related;
    if (sameExactMatchFamily && std::abs(leftCandidate.confidenceScore - rightCandidate.confidenceScore) < 8.0) {
        const int leftCanonicalPreference = canonicalWidthPreferenceScore(leftCandidate);
        const int rightCanonicalPreference = canonicalWidthPreferenceScore(rightCandidate);
        if (leftCanonicalPreference != rightCanonicalPreference) {
            return leftCanonicalPreference > rightCanonicalPreference;
        }

        const int widthDeltaBits = std::abs(leftCandidate.refinedPattern.bitWidth - rightCandidate.refinedPattern.bitWidth);
        if (widthDeltaBits <= 2 && leftCandidate.refinedPattern.bitWidth != rightCandidate.refinedPattern.bitWidth) {
            return leftCandidate.refinedPattern.bitWidth > rightCandidate.refinedPattern.bitWidth;
        }
    }

    if (sameRelatedFamily && std::abs(leftCandidate.confidenceScore - rightCandidate.confidenceScore) < 8.0) {
        const int leftCanonicalPreference = canonicalWidthPreferenceScore(leftCandidate);
        const int rightCanonicalPreference = canonicalWidthPreferenceScore(rightCandidate);
        if (leftCanonicalPreference != rightCanonicalPreference) {
            return leftCanonicalPreference > rightCanonicalPreference;
        }
        if (leftCandidate.refinedPattern.bitWidth != rightCandidate.refinedPattern.bitWidth) {
            return leftCandidate.refinedPattern.bitWidth > rightCandidate.refinedPattern.bitWidth;
        }
    }

    if (std::abs(leftCandidate.confidenceScore - rightCandidate.confidenceScore) > 0.001) {
        return leftCandidate.confidenceScore > rightCandidate.confidenceScore;
    }
    if (leftCandidate.detectedEntropyCliffCount != rightCandidate.detectedEntropyCliffCount) {
        return leftCandidate.detectedEntropyCliffCount > rightCandidate.detectedEntropyCliffCount;
    }
    if (leftCandidate.sharedHeaderCliffCount != rightCandidate.sharedHeaderCliffCount) {
        return leftCandidate.sharedHeaderCliffCount > rightCandidate.sharedHeaderCliffCount;
    }
    if (leftCandidate.matchStartBits.size() != rightCandidate.matchStartBits.size()) {
        return leftCandidate.matchStartBits.size() > rightCandidate.matchStartBits.size();
    }
    const qsizetype leftStartBit = leftCandidate.matchStartBits.isEmpty() ? 0 : leftCandidate.matchStartBits.first();
    const qsizetype rightStartBit = rightCandidate.matchStartBits.isEmpty() ? 0 : rightCandidate.matchStartBits.first();
    if (leftStartBit == rightStartBit
        && leftCandidate.matchStartBits.size() == rightCandidate.matchStartBits.size()
        && leftCandidate.medianGapBits == rightCandidate.medianGapBits
        && std::abs(leftCandidate.confidenceScore - rightCandidate.confidenceScore) < 5.0) {
        const int widthDeltaBits = std::abs(leftCandidate.refinedPattern.bitWidth - rightCandidate.refinedPattern.bitWidth);
        if (widthDeltaBits <= 2
            && std::abs(leftCandidate.cliffSharpnessScore - rightCandidate.cliffSharpnessScore) > 0.10) {
            return leftCandidate.cliffSharpnessScore > rightCandidate.cliffSharpnessScore;
        }
        if (leftCandidate.refinedPattern.bitWidth != rightCandidate.refinedPattern.bitWidth) {
            return leftCandidate.refinedPattern.bitWidth > rightCandidate.refinedPattern.bitWidth;
        }
    }
    if (leftCandidate.refinedPattern.bitWidth != rightCandidate.refinedPattern.bitWidth) {
        return leftCandidate.refinedPattern.bitWidth > rightCandidate.refinedPattern.bitWidth;
    }
    return leftStartBit < rightStartBit;
}

[[nodiscard]] bool shouldReplaceDeduplicatedFamilyWinner(
    const BitstreamSyncDiscoveryCandidate& candidate,
    const BitstreamSyncDiscoveryCandidate& currentWinner
) {
    if (candidateRanksBefore(candidate, currentWinner)) {
        return true;
    }

    const bool sameMatchFamily = candidatesShareExactMatchFamily(candidate, currentWinner);
    if (!sameMatchFamily) {
        return false;
    }

    if (std::abs(candidate.confidenceScore - currentWinner.confidenceScore) >= 5.0) {
        return false;
    }

    const int widthDeltaBits = candidate.refinedPattern.bitWidth - currentWinner.refinedPattern.bitWidth;
    if (widthDeltaBits > 0) {
        return true;
    }
    if (std::abs(widthDeltaBits) <= 2
        && candidate.cliffSharpnessScore > currentWinner.cliffSharpnessScore + 0.10) {
        return true;
    }
    return false;
}

[[nodiscard]] bool candidatesBelongToShiftedFamily(
    const BitstreamSyncDiscoveryCandidate& leftCandidate,
    const BitstreamSyncDiscoveryCandidate& rightCandidate
) {
    if (leftCandidate.matchStartBits.size() != rightCandidate.matchStartBits.size()
        || leftCandidate.matchStartBits.isEmpty()
        || leftCandidate.medianGapBits != rightCandidate.medianGapBits) {
        return false;
    }

    const qsizetype familyDeltaBits = rightCandidate.matchStartBits.first() - leftCandidate.matchStartBits.first();
    if (std::abs(familyDeltaBits) > 2) {
        return false;
    }

    const int signatureCount = qMin(8, static_cast<int>(leftCandidate.matchStartBits.size()));
    for (int signatureIndex = 1; signatureIndex < signatureCount; ++signatureIndex) {
        const qsizetype deltaBits =
            rightCandidate.matchStartBits.at(signatureIndex) - leftCandidate.matchStartBits.at(signatureIndex);
        if (std::abs(deltaBits - familyDeltaBits) > 1) {
            return false;
        }
    }

    return true;
}

[[nodiscard]] DeduplicationKey deduplicationKey(const BitstreamSyncDiscoveryCandidate& candidate) {
    DeduplicationKey key;
    key.matchCount = candidate.matchStartBits.size();
    key.firstStartBit = candidate.matchStartBits.isEmpty() ? -1 : candidate.matchStartBits.first();
    key.medianGapBits = candidate.medianGapBits;
    key.signatureLength = qMax(
        0,
        qMin(
            kDeduplicationSignatureDeltaCount,
            static_cast<int>(candidate.matchStartBits.size()) - 1
        )
    );
    for (int signatureIndex = 0; signatureIndex < key.signatureLength; ++signatureIndex) {
        key.gapSignature[static_cast<size_t>(signatureIndex)] =
            candidate.matchStartBits.at(signatureIndex + 1) - candidate.matchStartBits.at(signatureIndex);
    }
    return key;
}

}  // namespace

bool patternRecordRanksBefore(
    const PatternOccurrenceRecord& leftRecord,
    const PatternOccurrenceRecord& rightRecord
) {
    if (!qFuzzyCompare(leftRecord.preliminaryScore, rightRecord.preliminaryScore)) {
        return leftRecord.preliminaryScore > rightRecord.preliminaryScore;
    }
    if (leftRecord.matchCount != rightRecord.matchCount) {
        return leftRecord.matchCount > rightRecord.matchCount;
    }
    if (!qFuzzyCompare(leftRecord.gapCoefficientVariation, rightRecord.gapCoefficientVariation)) {
        return leftRecord.gapCoefficientVariation < rightRecord.gapCoefficientVariation;
    }
    const qsizetype leftFirstStartBit =
        leftRecord.matchStartBits.isEmpty() ? std::numeric_limits<qsizetype>::max() : leftRecord.matchStartBits.first();
    const qsizetype rightFirstStartBit =
        rightRecord.matchStartBits.isEmpty() ? std::numeric_limits<qsizetype>::max() : rightRecord.matchStartBits.first();
    if (leftFirstStartBit != rightFirstStartBit) {
        return leftFirstStartBit < rightFirstStartBit;
    }
    return patternValueLess(leftRecord.patternValue, rightRecord.patternValue);
}

PreanalyzedFamilyKey preanalyzedFamilyKey(
    const QVector<qsizetype>& matchStartBits,
    qsizetype medianGapBits
) {
    PreanalyzedFamilyKey key;
    key.medianGapBits = medianGapBits;
    key.signatureLength = qMax(
        0,
        qMin(
            kPreanalyzedFamilySignatureDeltaCount,
            static_cast<int>(matchStartBits.size()) - 1
        )
    );
    for (int signatureIndex = 0; signatureIndex < key.signatureLength; ++signatureIndex) {
        key.gapSignature[static_cast<size_t>(signatureIndex)] =
            matchStartBits.at(signatureIndex + 1) - matchStartBits.at(signatureIndex);
    }
    return key;
}

QVector<PreanalyzedCandidate> screenedPreanalyzedCandidatesForFullAnalysis(
    const QVector<PreanalyzedCandidate>& preanalyzedCandidates,
    int widthCount,
    const BitstreamSyncDiscoverySettings& settings,
    int* familyWinnerCount
) {
    if (preanalyzedCandidates.isEmpty()) {
        if (familyWinnerCount != nullptr) {
            *familyWinnerCount = 0;
        }
        return {};
    }

    const int globalAnalysisBudget = screenedPreanalysisGlobalBudget(
        widthCount,
        settings,
        preanalyzedCandidates.size()
    );

    QHash<PreanalyzedFamilyKey, PreanalyzedCandidate> bestCandidateByFamily;
    bestCandidateByFamily.reserve(preanalyzedCandidates.size());
    for (const PreanalyzedCandidate& candidate : preanalyzedCandidates) {
        const auto existingIterator = bestCandidateByFamily.constFind(candidate.familyKey);
        if (existingIterator == bestCandidateByFamily.constEnd()
            || shouldReplacePreanalyzedFamilyWinner(candidate, existingIterator.value())) {
            bestCandidateByFamily.insert(candidate.familyKey, candidate);
        }
    }

    QVector<PreanalyzedCandidate> familyWinners;
    familyWinners.reserve(bestCandidateByFamily.size());
    for (auto it = bestCandidateByFamily.cbegin(); it != bestCandidateByFamily.cend(); ++it) {
        familyWinners.append(it.value());
    }
    std::sort(familyWinners.begin(), familyWinners.end(), preanalyzedCandidateRanksBefore);
    if (familyWinnerCount != nullptr) {
        *familyWinnerCount = familyWinners.size();
    }

    QVector<PreanalyzedCandidate> quickSortedCandidates = preanalyzedCandidates;
    std::sort(quickSortedCandidates.begin(), quickSortedCandidates.end(), preanalyzedCandidateRanksBefore);

    QVector<PreanalyzedCandidate> screenedCandidates;
    screenedCandidates.reserve(globalAnalysisBudget);
    QSet<PatternRecordKey> keptCandidateKeys;
    keptCandidateKeys.reserve(globalAnalysisBudget * 2);
    QHash<PreanalyzedFamilyKey, int> keptCountByFamily;
    keptCountByFamily.reserve(bestCandidateByFamily.size());
    QHash<int, int> keptCountByWidth;
    keptCountByWidth.reserve(widthCount);

    const int perWidthKeepCount = screenedPreanalysisPerWidthKeepCount(settings);

    const auto trackScreenedCandidate = [&](const PreanalyzedCandidate& candidate) {
        const PatternRecordKey recordKey = patternRecordKey(candidate.patternRecord);
        screenedCandidates.append(candidate);
        keptCandidateKeys.insert(recordKey);
        keptCountByFamily.insert(candidate.familyKey, keptCountByFamily.value(candidate.familyKey, 0) + 1);
        keptCountByWidth.insert(
            candidate.adjustedPattern.bitWidth,
            keptCountByWidth.value(candidate.adjustedPattern.bitWidth, 0) + 1
        );
    };

    for (const PreanalyzedCandidate& candidate : familyWinners) {
        if (screenedCandidates.size() >= globalAnalysisBudget) {
            break;
        }
        const PatternRecordKey candidateKey = patternRecordKey(candidate.patternRecord);
        if (!keptCandidateKeys.contains(candidateKey)) {
            trackScreenedCandidate(candidate);
        }
    }

    for (const PreanalyzedCandidate& candidate : quickSortedCandidates) {
        if (screenedCandidates.size() >= globalAnalysisBudget) {
            break;
        }
        const PatternRecordKey candidateKey = patternRecordKey(candidate.patternRecord);
        if (keptCandidateKeys.contains(candidateKey)) {
            continue;
        }
        if (keptCountByFamily.value(candidate.familyKey, 0) >= kPreanalysisAlternatesPerFamily) {
            continue;
        }
        if (keptCountByWidth.value(candidate.adjustedPattern.bitWidth, 0) >= perWidthKeepCount) {
            continue;
        }
        trackScreenedCandidate(candidate);
    }

    for (const PreanalyzedCandidate& candidate : quickSortedCandidates) {
        if (screenedCandidates.size() >= globalAnalysisBudget) {
            break;
        }
        const PatternRecordKey candidateKey = patternRecordKey(candidate.patternRecord);
        if (keptCandidateKeys.contains(candidateKey)) {
            continue;
        }
        if (keptCountByFamily.value(candidate.familyKey, 0) >= kPreanalysisAlternatesPerFamily) {
            continue;
        }
        trackScreenedCandidate(candidate);
    }

    std::sort(screenedCandidates.begin(), screenedCandidates.end(), preanalyzedCandidateRanksBefore);
    return screenedCandidates;
}

QVector<PatternOccurrenceRecord> screenedPatternRecordsForAnalysis(
    const QVector<PatternOccurrenceRecord>& patternRecords,
    int widthCount,
    const BitstreamSyncDiscoverySettings& settings
) {
    if (patternRecords.isEmpty()) {
        return {};
    }

    const int perWidthKeepCount = screenedPatternPerWidthKeepCount(settings);
    const int globalAnalysisBudget = screenedPatternGlobalAnalysisBudget(
        widthCount,
        settings,
        patternRecords.size()
    );
    if (patternRecords.size() <= globalAnalysisBudget) {
        return patternRecords;
    }

    QVector<PatternOccurrenceRecord> quickSortedRecords = patternRecords;
    std::sort(quickSortedRecords.begin(), quickSortedRecords.end(), quickScreenRanksBefore);

    QVector<PatternOccurrenceRecord> screenedPatternRecords;
    screenedPatternRecords.reserve(globalAnalysisBudget);
    QSet<PatternRecordKey> keptRecordKeys;
    keptRecordKeys.reserve(globalAnalysisBudget * 2);
    QHash<int, int> keptCountByWidth;
    keptCountByWidth.reserve(widthCount);
    QHash<PreanalyzedFamilyKey, PatternOccurrenceRecord> bestKeptRecordByFamily;
    bestKeptRecordByFamily.reserve(globalAnalysisBudget);

    for (const PatternOccurrenceRecord& patternRecord : quickSortedRecords) {
        if (screenedPatternRecords.size() >= globalAnalysisBudget) {
            break;
        }

        if (keptCountByWidth.value(patternRecord.bitWidth, 0) >= perWidthKeepCount) {
            continue;
        }

        const PatternRecordKey recordKey = patternRecordKey(patternRecord);
        if (keptRecordKeys.contains(recordKey)) {
            continue;
        }

        screenedPatternRecords.append(patternRecord);
        keptRecordKeys.insert(recordKey);
        keptCountByWidth.insert(patternRecord.bitWidth, keptCountByWidth.value(patternRecord.bitWidth, 0) + 1);
        const PreanalyzedFamilyKey familyKey = preanalyzedFamilyKey(patternRecord.matchStartBits, patternRecord.medianGapBits);
        const auto existingFamilyIterator = bestKeptRecordByFamily.constFind(familyKey);
        if (existingFamilyIterator == bestKeptRecordByFamily.constEnd()
            || quickScreenRanksBefore(patternRecord, existingFamilyIterator.value())
            || shouldRescueWiderPatternRecord(patternRecord, existingFamilyIterator.value())) {
            bestKeptRecordByFamily.insert(familyKey, patternRecord);
        }
    }

    QSet<PreanalyzedFamilyKey> rescuedFamilies;
    rescuedFamilies.reserve(bestKeptRecordByFamily.size());
    for (const PatternOccurrenceRecord& patternRecord : quickSortedRecords) {
        if (screenedPatternRecords.size() >= globalAnalysisBudget) {
            break;
        }

        const PatternRecordKey recordKey = patternRecordKey(patternRecord);
        if (keptRecordKeys.contains(recordKey)) {
            continue;
        }

        const PreanalyzedFamilyKey familyKey = preanalyzedFamilyKey(patternRecord.matchStartBits, patternRecord.medianGapBits);
        if (rescuedFamilies.contains(familyKey)) {
            continue;
        }

        const auto existingFamilyIterator = bestKeptRecordByFamily.constFind(familyKey);
        if (existingFamilyIterator == bestKeptRecordByFamily.constEnd()
            || !shouldRescueWiderPatternRecord(patternRecord, existingFamilyIterator.value())) {
            continue;
        }

        screenedPatternRecords.append(patternRecord);
        keptRecordKeys.insert(recordKey);
        keptCountByWidth.insert(patternRecord.bitWidth, keptCountByWidth.value(patternRecord.bitWidth, 0) + 1);
        rescuedFamilies.insert(familyKey);
    }

    for (const PatternOccurrenceRecord& patternRecord : quickSortedRecords) {
        if (screenedPatternRecords.size() >= globalAnalysisBudget) {
            break;
        }

        const PatternRecordKey recordKey = patternRecordKey(patternRecord);
        if (keptRecordKeys.contains(recordKey)) {
            continue;
        }

        screenedPatternRecords.append(patternRecord);
        keptRecordKeys.insert(recordKey);
    }

    std::sort(screenedPatternRecords.begin(), screenedPatternRecords.end(), patternRecordRanksBefore);
    return screenedPatternRecords;
}

bool shouldStopFullAnalysisEarly(
    const BitstreamSyncDiscoveryCandidateList& partialCandidates,
    const QVector<PreanalyzedCandidate>& screenedPreanalyzedCandidates,
    int analyzedCount,
    int familyWinnerCount,
    const BitstreamSyncDiscoverySettings& settings
) {
    if (!allowsEarlyStop(settings)) {
        return false;
    }

    if (partialCandidates.isEmpty() || analyzedCount >= screenedPreanalyzedCandidates.size()) {
        return false;
    }

    const int minimumAnalyzedCount = minimumAnalyzedCountForEarlyStop(familyWinnerCount, settings);
    if (analyzedCount < minimumAnalyzedCount) {
        return false;
    }

    const BitstreamSyncDiscoveryCandidate& leader = partialCandidates.first();
    const double leaderQuickScore = screenedPreanalyzedCandidates.first().quickScore;
    const double runnerUpScore = partialCandidates.size() >= 2
        ? partialCandidates.at(1).confidenceScore
        : 0.0;
    const double scoreGap = partialCandidates.size() >= 2
        ? leader.confidenceScore - runnerUpScore
        : leader.confidenceScore;
    const double nextQuickScore = screenedPreanalyzedCandidates.at(analyzedCount).quickScore;
    const qsizetype requiredFrameCount = requiredFrameCountForEarlyStop(settings);

    const bool strongLeader =
        leader.confidenceScore >= 28.0
        && scoreGap >= 3.0
        && leader.matchStartBits.size() >= requiredFrameCount
        && canonicalWidthPreferenceScore(leader) >= 2;
    if (!strongLeader) {
        return false;
    }

    const bool allFamilyWinnersCovered = analyzedCount >= familyWinnerCount;
    const bool remainingPreanalysisLooksWeak =
        nextQuickScore <= leader.confidenceScore - 2.5
        || (leaderQuickScore > 0.0 && (nextQuickScore / leaderQuickScore) <= 0.82);
    return allFamilyWinnersCovered || remainingPreanalysisLooksWeak;
}

BitstreamSyncDiscoveryCandidateList deduplicatedCandidates(
    const QVector<CandidateAnalysis>& analyses,
    int maximumResultCount
) {
    QHash<DeduplicationKey, BitstreamSyncDiscoveryCandidate> bestCandidateByKey;
    bestCandidateByKey.reserve(analyses.size());

    for (const CandidateAnalysis& analysis : analyses) {
        const BitstreamSyncDiscoveryCandidate& candidate = analysis.candidate;
        const DeduplicationKey key = deduplicationKey(candidate);
        const auto existingIterator = bestCandidateByKey.constFind(key);
        if (existingIterator == bestCandidateByKey.constEnd()
            || shouldReplaceDeduplicatedFamilyWinner(candidate, existingIterator.value())) {
            bestCandidateByKey.insert(key, candidate);
        }
    }

    BitstreamSyncDiscoveryCandidateList candidates;
    candidates.reserve(bestCandidateByKey.size());
    for (auto it = bestCandidateByKey.cbegin(); it != bestCandidateByKey.cend(); ++it) {
        candidates.append(it.value());
    }
    std::sort(candidates.begin(), candidates.end(), candidateRanksBefore);

    BitstreamSyncDiscoveryCandidateList familyFilteredCandidates;
    familyFilteredCandidates.reserve(candidates.size());
    for (const BitstreamSyncDiscoveryCandidate& candidate : candidates) {
        bool shouldSuppressCandidate = false;
        for (const BitstreamSyncDiscoveryCandidate& keptCandidate : familyFilteredCandidates) {
            const ContainedAliasRelation aliasRelation = containedAliasRelation(keptCandidate, candidate);
            if (shouldPreferLongerContainedAlias(keptCandidate, candidate, aliasRelation)
                && aliasRelation.leftIsLonger) {
                shouldSuppressCandidate = true;
                break;
            }

            if (candidatesBelongToShiftedFamily(keptCandidate, candidate)
                && std::abs(keptCandidate.confidenceScore - candidate.confidenceScore) < 3.0) {
                shouldSuppressCandidate = true;
                break;
            }
        }

        if (!shouldSuppressCandidate) {
            familyFilteredCandidates.append(candidate);
        }
    }

    if (familyFilteredCandidates.size() >= 2) {
        QVector<double> dominanceAdjustments(familyFilteredCandidates.size(), 0.0);
        for (int leftIndex = 0; leftIndex < familyFilteredCandidates.size(); ++leftIndex) {
            for (int rightIndex = leftIndex + 1; rightIndex < familyFilteredCandidates.size(); ++rightIndex) {
                const BitstreamSyncDiscoveryCandidate& leftCandidate = familyFilteredCandidates.at(leftIndex);
                const BitstreamSyncDiscoveryCandidate& rightCandidate = familyFilteredCandidates.at(rightIndex);
                const DominanceRelation relation = dominanceRelation(leftCandidate, rightCandidate);
                if (!relation.related) {
                    continue;
                }

                int widerIndex = leftIndex;
                int narrowerIndex = rightIndex;
                if (rightCandidate.refinedPattern.bitWidth > leftCandidate.refinedPattern.bitWidth) {
                    widerIndex = rightIndex;
                    narrowerIndex = leftIndex;
                }
                const BitstreamSyncDiscoveryCandidate& widerCandidate = familyFilteredCandidates.at(widerIndex);
                const BitstreamSyncDiscoveryCandidate& narrowerCandidate = familyFilteredCandidates.at(narrowerIndex);
                if (widerCandidate.refinedPattern.bitWidth <= narrowerCandidate.refinedPattern.bitWidth) {
                    continue;
                }

                const int widerCanonicalPreference = canonicalWidthPreferenceScore(widerCandidate);
                const int narrowerCanonicalPreference = canonicalWidthPreferenceScore(narrowerCandidate);
                if (widerCanonicalPreference < narrowerCanonicalPreference) {
                    continue;
                }

                const double scoreGap = narrowerCandidate.confidenceScore - widerCandidate.confidenceScore;
                if (scoreGap > kDominanceMaximumScoreGapForWidthBias) {
                    continue;
                }

                const int widthDeltaBits = widerCandidate.refinedPattern.bitWidth - narrowerCandidate.refinedPattern.bitWidth;
                if (widthDeltaBits < 4 && widerCanonicalPreference == narrowerCanonicalPreference) {
                    continue;
                }

                const double baseDominanceBonus = qMin(
                    2.5,
                    (0.75 + (static_cast<double>(widthDeltaBits) * 0.125)) * relation.largerOverlapRatio
                );
                const double requiredLead =
                    0.25
                    + 0.05 * static_cast<double>(qMin(widthDeltaBits, 12))
                    + 0.30 * static_cast<double>(qMax(0, widerCanonicalPreference - narrowerCanonicalPreference));
                const double requiredSwing = qMax(0.0, scoreGap + requiredLead);

                const double widerBonus = qMin(
                    5.0,
                    qMax(baseDominanceBonus, requiredSwing * 0.60)
                );
                const double narrowerPenalty = qMin(
                    3.5,
                    qMax(baseDominanceBonus * 0.65, requiredSwing - widerBonus)
                );

                dominanceAdjustments[widerIndex] += widerBonus;
                dominanceAdjustments[narrowerIndex] -= narrowerPenalty;
            }
        }

        for (int candidateIndex = 0; candidateIndex < familyFilteredCandidates.size(); ++candidateIndex) {
            if (qFuzzyIsNull(dominanceAdjustments.at(candidateIndex))) {
                continue;
            }
            familyFilteredCandidates[candidateIndex].confidenceScore = qBound(
                0.0,
                familyFilteredCandidates.at(candidateIndex).confidenceScore + dominanceAdjustments.at(candidateIndex),
                100.0
            );
        }
        std::sort(familyFilteredCandidates.begin(), familyFilteredCandidates.end(), candidateRanksBefore);
    }

    if (familyFilteredCandidates.size() > maximumResultCount) {
        familyFilteredCandidates.resize(maximumResultCount);
    }
    return familyFilteredCandidates;
}

}  // namespace bitabyte::features::bitstream_sync_discovery::detail
