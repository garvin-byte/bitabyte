#pragma once

#include <optional>

#include <QVector>
#include <QtTypes>

#include "features/framing/frame_layout.h"

class QString;

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::features::frame_sync {

struct PatternSearchResult {
    qsizetype patternBitCount = 0;
    QVector<qsizetype> matchStartBits;
};

struct FrameSyncSearchResult {
    qsizetype firstMatchBit = -1;
    qsizetype matchCount = 0;
    QVector<framing::FrameSpan> frameSpans;
};

class FrameSyncSearch {
public:
    [[nodiscard]] static std::optional<PatternSearchResult> findPatternMatches(
        const data::ByteDataSource& dataSource,
        const QString& patternText,
        QString* errorMessage = nullptr
    );

    [[nodiscard]] static std::optional<FrameSyncSearchResult> findBitAccurateMatches(
        const data::ByteDataSource& dataSource,
        const QString& patternText,
        QString* errorMessage = nullptr
    );
};

}  // namespace bitabyte::features::frame_sync
