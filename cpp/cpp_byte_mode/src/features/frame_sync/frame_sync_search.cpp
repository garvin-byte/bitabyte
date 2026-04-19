#include "features/frame_sync/frame_sync_search.h"

#include "data/byte_data_source.h"

#include <QByteArray>
#include <QChar>
#include <QString>
#include <QStringView>
#include <QVector>

#include <cstring>

namespace bitabyte::features::frame_sync {
namespace {

struct ParsedPatternBits {
    QByteArray packedBytes;
    qsizetype bitCount = 0;

    [[nodiscard]] bool isByteAligned() const {
        return bitCount > 0 && (bitCount % 8) == 0;
    }
};

bool isPatternSeparator(QChar character) {
    return character.isSpace() || character == QLatin1Char(',') || character == QLatin1Char('_');
}

int hexDigitValue(QChar character) {
    const ushort codePoint = character.unicode();
    if (codePoint >= u'0' && codePoint <= u'9') {
        return static_cast<int>(codePoint - u'0');
    }
    if (codePoint >= u'a' && codePoint <= u'f') {
        return 10 + static_cast<int>(codePoint - u'a');
    }
    if (codePoint >= u'A' && codePoint <= u'F') {
        return 10 + static_cast<int>(codePoint - u'A');
    }
    return -1;
}

bool containsOnlyBinaryDigits(QStringView text) {
    for (QChar character : text) {
        if (character != QLatin1Char('0') && character != QLatin1Char('1')) {
            return false;
        }
    }
    return !text.isEmpty();
}

std::optional<ParsedPatternBits> parsePatternBits(const QString& patternText, QString* errorMessage) {
    QString normalizedPatternText;
    normalizedPatternText.reserve(patternText.size());

    for (QChar character : patternText) {
        if (isPatternSeparator(character)) {
            continue;
        }

        normalizedPatternText.append(character);
    }

    bool interpretAsBinary = false;
    if (normalizedPatternText.startsWith(QStringLiteral("0b"), Qt::CaseInsensitive)) {
        interpretAsBinary = true;
        normalizedPatternText.remove(0, 2);
    } else if (normalizedPatternText.startsWith(QStringLiteral("0x"), Qt::CaseInsensitive)) {
        normalizedPatternText.remove(0, 2);
    }

    if (normalizedPatternText.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Enter a sync pattern like 0x1ACF, 1011011, or 0b1011011.");
        }
        return std::nullopt;
    }

    const QStringView normalizedPatternView{normalizedPatternText};
    if (!interpretAsBinary) {
        interpretAsBinary = containsOnlyBinaryDigits(normalizedPatternView);
    }

    ParsedPatternBits parsedPattern;
    if (interpretAsBinary) {
        parsedPattern.bitCount = normalizedPatternText.size();
        parsedPattern.packedBytes.fill('\0', static_cast<int>((parsedPattern.bitCount + 7) / 8));
        for (int bitIndex = 0; bitIndex < normalizedPatternText.size(); ++bitIndex) {
            const QChar character = normalizedPatternText.at(bitIndex);
            if (character != QLatin1Char('0') && character != QLatin1Char('1')) {
                if (errorMessage != nullptr) {
                    *errorMessage = QStringLiteral("Binary sync pattern may only contain 0 and 1.");
                }
                return std::nullopt;
            }
            if (character == QLatin1Char('1')) {
                const int byteIndex = bitIndex / 8;
                const int bitOffset = bitIndex % 8;
                parsedPattern.packedBytes[byteIndex] = static_cast<char>(
                    static_cast<unsigned char>(parsedPattern.packedBytes.at(byteIndex))
                    | static_cast<unsigned char>(1u << (7 - bitOffset))
                );
            }
        }
        return parsedPattern;
    }

    parsedPattern.bitCount = static_cast<qsizetype>(normalizedPatternText.size()) * 4;
    parsedPattern.packedBytes.fill('\0', static_cast<int>((parsedPattern.bitCount + 7) / 8));
    for (int digitIndex = 0; digitIndex < normalizedPatternText.size(); ++digitIndex) {
        const int nibbleValue = hexDigitValue(normalizedPatternText.at(digitIndex));
        if (nibbleValue < 0) {
            if (errorMessage != nullptr) {
                *errorMessage = QStringLiteral("Sync pattern contains non-hex characters.");
            }
            return std::nullopt;
        }

        const int byteIndex = digitIndex / 2;
        const int shift = digitIndex % 2 == 0 ? 4 : 0;
        parsedPattern.packedBytes[byteIndex] = static_cast<char>(
            static_cast<unsigned char>(parsedPattern.packedBytes.at(byteIndex))
            | static_cast<unsigned char>(nibbleValue << shift)
        );
    }
    return parsedPattern;
}

bool bytesMatchAtOffset(
    const data::ByteDataSource& dataSource,
    const QByteArray& patternBytes,
    qsizetype startBit
) {
    if (patternBytes.isEmpty()) {
        return false;
    }

    const int patternByteCount = patternBytes.size();
    const int bitOffsetInByte = static_cast<int>(startBit % 8);
    if (bitOffsetInByte == 0) {
        const char* dataPointer = dataSource.rawBytes().constData() + (startBit / 8);
        return std::memcmp(dataPointer, patternBytes.constData(), static_cast<size_t>(patternByteCount)) == 0;
    }

    const unsigned char firstPatternByte = static_cast<unsigned char>(patternBytes.at(0));
    if (dataSource.byteValueAtBitOffset(startBit) != firstPatternByte) {
        return false;
    }

    if (patternByteCount == 1) {
        return true;
    }

    const unsigned char lastPatternByte = static_cast<unsigned char>(patternBytes.at(patternByteCount - 1));
    if (dataSource.byteValueAtBitOffset(startBit + ((patternByteCount - 1) * 8)) != lastPatternByte) {
        return false;
    }

    for (int patternByteIndex = 1; patternByteIndex < patternByteCount - 1; ++patternByteIndex) {
        if (dataSource.byteValueAtBitOffset(startBit + (patternByteIndex * 8))
            != static_cast<unsigned char>(patternBytes.at(patternByteIndex))) {
            return false;
        }
    }

    return true;
}

bool patternMatchesAtOffset(
    const data::ByteDataSource& dataSource,
    const ParsedPatternBits& patternBits,
    qsizetype startBit
) {
    if (patternBits.bitCount <= 0 || patternBits.packedBytes.isEmpty()) {
        return false;
    }

    if (patternBits.isByteAligned()) {
        return bytesMatchAtOffset(dataSource, patternBits.packedBytes, startBit);
    }

    const int fullByteCount = static_cast<int>(patternBits.bitCount / 8);
    const int trailingBitCount = static_cast<int>(patternBits.bitCount % 8);
    for (int byteIndex = 0; byteIndex < fullByteCount; ++byteIndex) {
        if (dataSource.byteValueAtBitOffset(startBit + (byteIndex * 8))
            != static_cast<unsigned char>(patternBits.packedBytes.at(byteIndex))) {
            return false;
        }
    }

    if (trailingBitCount <= 0) {
        return true;
    }

    const unsigned char candidateByte = dataSource.byteValueAtBitOffset(startBit + (fullByteCount * 8));
    const unsigned char expectedByte = static_cast<unsigned char>(patternBits.packedBytes.at(fullByteCount));
    const unsigned char trailingMask = static_cast<unsigned char>(0xFFu << (8 - trailingBitCount));
    return (candidateByte & trailingMask) == (expectedByte & trailingMask);
}

std::optional<PatternSearchResult> findPatternMatchesInternal(
    const data::ByteDataSource& dataSource,
    const QString& patternText,
    bool allowOverlappingMatches,
    QString* errorMessage
) {
    if (!dataSource.hasData()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Load a file first.");
        }
        return std::nullopt;
    }

    const std::optional<ParsedPatternBits> patternBits = parsePatternBits(patternText, errorMessage);
    if (!patternBits.has_value()) {
        return std::nullopt;
    }

    if (patternBits->bitCount <= 0 || patternBits->packedBytes.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Pattern must contain at least one bit.");
        }
        return std::nullopt;
    }

    PatternSearchResult searchResult;
    searchResult.patternBitCount = patternBits->bitCount;
    if (dataSource.bitCount() < searchResult.patternBitCount) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Pattern is longer than the loaded file.");
        }
        return std::nullopt;
    }

    const qsizetype maxStartBit = dataSource.bitCount() - searchResult.patternBitCount;
    searchResult.matchStartBits.reserve(static_cast<int>(qMin<qsizetype>(maxStartBit + 1, 4096)));

    for (qsizetype startBit = 0; startBit <= maxStartBit; ++startBit) {
        if (!patternMatchesAtOffset(dataSource, *patternBits, startBit)) {
            continue;
        }

        searchResult.matchStartBits.append(startBit);
        if (!allowOverlappingMatches && searchResult.patternBitCount > 0) {
            startBit += searchResult.patternBitCount - 1;
        }
    }

    if (searchResult.matchStartBits.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Pattern was not found in the loaded file.");
        }
        return std::nullopt;
    }

    return searchResult;
}

}  // namespace

std::optional<PatternSearchResult> FrameSyncSearch::findPatternMatches(
    const data::ByteDataSource& dataSource,
    const QString& patternText,
    QString* errorMessage
) {
    return findPatternMatchesInternal(dataSource, patternText, true, errorMessage);
}

std::optional<FrameSyncSearchResult> FrameSyncSearch::findBitAccurateMatches(
    const data::ByteDataSource& dataSource,
    const QString& patternText,
    QString* errorMessage
) {
    const std::optional<PatternSearchResult> patternSearchResult =
        findPatternMatchesInternal(dataSource, patternText, false, errorMessage);
    if (!patternSearchResult.has_value()) {
        return std::nullopt;
    }

    FrameSyncSearchResult searchResult;
    searchResult.firstMatchBit = patternSearchResult->matchStartBits.first();
    searchResult.matchCount = patternSearchResult->matchStartBits.size();
    const qsizetype patternBitCount = patternSearchResult->patternBitCount;
    const QVector<qsizetype>& matchBits = patternSearchResult->matchStartBits;
    for (int matchIndex = 0; matchIndex < matchBits.size(); ++matchIndex) {
        const qsizetype startBit = matchBits[matchIndex];
        const qsizetype nextStartBit =
            matchIndex + 1 < matchBits.size() ? matchBits[matchIndex + 1] : dataSource.bitCount();
        const qsizetype lengthBits = qMax<qsizetype>(0, nextStartBit - startBit);
        if (lengthBits <= 0) {
            continue;
        }

        framing::FrameSpan frameSpan;
        frameSpan.startBit = startBit;
        frameSpan.lengthBits = lengthBits;
        searchResult.frameSpans.append(frameSpan);
    }

    if (searchResult.frameSpans.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Sync pattern matches did not produce any usable frames.");
        }
        return std::nullopt;
    }

    return searchResult;
}

}  // namespace bitabyte::features::frame_sync
