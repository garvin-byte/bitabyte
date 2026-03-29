#include "features/bitstream_sync_discovery/bitstream_sync_discovery_formatter.h"

#include "features/bitstream_sync_discovery/bitstream_sync_discovery_internal.h"

#include <utility>

namespace bitabyte::features::bitstream_sync_discovery {
namespace {

using namespace detail;

[[nodiscard]] QString formatHexPrefixPattern(PatternValue bitValue, int bitWidth) {
    if (bitWidth <= 0) {
        return {};
    }

    const int trailingBitCount = bitWidth % 4;
    const int hexBitCount = bitWidth - trailingBitCount;
    if (hexBitCount <= 0) {
        return QStringLiteral("0x0");
    }

    const int requiredHexDigits = qMax(1, hexBitCount / 4);
    const PatternValue hexValue =
        trailingBitCount > 0 ? patternValueShiftRight(bitValue, trailingBitCount) : bitValue;
    QString hexText;
    hexText.reserve(requiredHexDigits);
    for (int digitIndex = 0; digitIndex < requiredHexDigits; ++digitIndex) {
        const PatternValue nibbleValue = extractPatternSegment(hexValue, hexBitCount, digitIndex * 4, 4);
        const int nibble = static_cast<int>(nibbleValue.lower & 0x0F);
        hexText.append(QString::number(nibble, 16).toUpper());
    }
    return QStringLiteral("0x%1").arg(hexText);
}

[[nodiscard]] QString formatTrailingBits(PatternValue bitValue, int trailingBitCount) {
    if (trailingBitCount <= 0) {
        return {};
    }

    QString trailingBitsText;
    trailingBitsText.reserve(trailingBitCount);
    for (int bitOffset = trailingBitCount - 1; bitOffset >= 0; --bitOffset) {
        trailingBitsText.append(patternValueBitAt(bitValue, bitOffset) ? QLatin1Char('1') : QLatin1Char('0'));
    }
    return trailingBitsText;
}

}  // namespace

void formatCandidateForDisplay(BitstreamSyncDiscoveryCandidate* candidate) {
    if (candidate == nullptr) {
        return;
    }

    const int trailingBitCount = candidate->refinedPattern.bitWidth % 4;
    const int hexBitCount = candidate->refinedPattern.bitWidth - trailingBitCount;
    candidate->displayHexPattern = hexBitCount > 0
        ? formatHexPrefixPattern(candidate->refinedPattern.bitValue, candidate->refinedPattern.bitWidth)
        : QString();
    candidate->trailingBitsText = formatTrailingBits(candidate->refinedPattern.bitValue, trailingBitCount);
    if (!candidate->displayHexPattern.isEmpty() && !candidate->trailingBitsText.isEmpty()) {
        candidate->displayPattern =
            QStringLiteral("%1 %2").arg(candidate->displayHexPattern, candidate->trailingBitsText);
        candidate->displayFormat = QStringLiteral("hex+bits");
    } else if (!candidate->displayHexPattern.isEmpty()) {
        candidate->displayPattern = candidate->displayHexPattern;
        candidate->displayFormat = QStringLiteral("hex");
    } else {
        candidate->displayPattern = candidate->trailingBitsText;
        candidate->displayFormat = QStringLiteral("binary");
    }
}

BitstreamSyncDiscoveryCandidateList formatCandidatesForDisplay(const BitstreamSyncDiscoveryCandidateList& candidates) {
    BitstreamSyncDiscoveryCandidateList formattedCandidates;
    formattedCandidates.reserve(candidates.size());
    for (BitstreamSyncDiscoveryCandidate candidate : candidates) {
        formatCandidateForDisplay(&candidate);
        formattedCandidates.append(std::move(candidate));
    }
    return formattedCandidates;
}

}  // namespace bitabyte::features::bitstream_sync_discovery
