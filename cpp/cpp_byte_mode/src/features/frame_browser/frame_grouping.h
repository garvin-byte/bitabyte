#pragma once

#include <QByteArray>
#include <QString>
#include <QVector>
#include <QtTypes>

#include "data/byte_data_source.h"
#include "features/framing/frame_layout.h"

namespace bitabyte::features::frame_browser {

enum class FrameGroupingKeyKind {
    FieldValue,
    FrameLength,
};

struct FrameGroupingKey {
    FrameGroupingKeyKind kind = FrameGroupingKeyKind::FieldValue;
    QString label;
    QString displayFormat = QStringLiteral("hex");
    int startBit = 0;
    int bitWidth = 0;

    [[nodiscard]] bool operator==(const FrameGroupingKey& other) const noexcept = default;
};

struct FrameGroupValue {
    bool missing = false;
    QByteArray rawBits;
    QString displayText;
    quint64 numericValue = 0;
    bool hasNumericValue = false;

    [[nodiscard]] bool operator==(const FrameGroupValue& other) const noexcept = default;
};

struct FrameFilterClause {
    FrameGroupingKey key;
    FrameGroupValue value;
};

[[nodiscard]] FrameGroupValue evaluateFrameGroupValue(
    const data::ByteDataSource& dataSource,
    const framing::FrameSpan& frameSpan,
    const FrameGroupingKey& key
);

[[nodiscard]] int compareFrameGroupValues(const FrameGroupValue& leftValue, const FrameGroupValue& rightValue);
[[nodiscard]] bool frameMatchesFilterClause(
    const data::ByteDataSource& dataSource,
    const framing::FrameSpan& frameSpan,
    const FrameFilterClause& filterClause
);
[[nodiscard]] QString appendScopeKey(
    const QString& parentScopeKey,
    const FrameGroupingKey& groupingKey,
    const FrameGroupValue& groupValue
);
[[nodiscard]] QString parentScopeKey(const QString& scopeKey);

}  // namespace bitabyte::features::frame_browser
