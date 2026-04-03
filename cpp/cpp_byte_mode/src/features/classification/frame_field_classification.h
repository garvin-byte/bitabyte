#pragma once

#include <QSet>
#include <QString>
#include <QVector>

#include "features/columns/visible_byte_column.h"

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::features::framing {
class FrameLayout;
}

namespace bitabyte::features::classification {

struct FrameFieldColumnSnapshot {
    int visibleColumnIndex = -1;
    features::columns::VisibleByteColumn visibleColumn;
    QString label;
    QString displayFormat = QStringLiteral("hex");
};

struct FrameFieldHint {
    int visibleColumnIndex = -1;
    int absoluteStartBit = -1;
    int absoluteEndBit = -1;
    QVector<int> visibleColumnIndices;
    double confidenceScore = 0.0;
    QString label;
    QString valueText;
    QString summaryText;
};

struct FrameFieldClassificationResult {
    QVector<FrameFieldHint> counterHints;
    QVector<FrameFieldHint> constantHints;
    QSet<int> counterVisibleColumnIndices;
    QSet<int> constantVisibleColumnIndices;
};

[[nodiscard]] FrameFieldClassificationResult classifyFramedVisibleColumns(
    const data::ByteDataSource& dataSource,
    const features::framing::FrameLayout& frameLayout,
    const QVector<FrameFieldColumnSnapshot>& columnSnapshots
);

[[nodiscard]] QVector<FrameFieldHint> discoverCounterHintsByBitWindow(
    const data::ByteDataSource& dataSource,
    const features::framing::FrameLayout& frameLayout,
    const QVector<FrameFieldColumnSnapshot>& displayColumns,
    int startBit,
    int endBit
);

}  // namespace bitabyte::features::classification
