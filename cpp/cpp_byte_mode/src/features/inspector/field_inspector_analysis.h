#pragma once

#include <QVector>

#include <QString>

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::features::framing {
class FrameLayout;
}

namespace bitabyte::features::inspector {

struct FieldSelection {
    QString label;
    int startBit = 0;
    int endBit = 0;
};

struct HistogramBin {
    QString label;
    int count = 0;
    double fraction = 0.0;
};

struct FieldInspectorAnalysis {
    bool hasField = false;
    QString fieldLabel;
    QString positionText;
    QString currentRowText;
    int bitWidth = 0;
    int analyzedFrameCount = 0;
    int missingFrameCount = 0;
    int uniqueValueCount = 0;
    QString currentHexValue;
    QString currentBinaryValue;
    QString currentAsciiValue;
    QString currentUnsignedBigEndianValue;
    QString currentUnsignedLittleEndianValue;
    QString currentSignedBigEndianValue;
    QString currentSignedLittleEndianValue;
    QString currentFloatBigEndianValue;
    QString currentFloatLittleEndianValue;
    QString minValueText;
    QString maxValueText;
    QString meanValueText;
    QString modeValueText;
    QString entropyText;
    bool isConstant = false;
    bool isMonotonicIncreasing = false;
    bool isHighEntropy = false;
    QVector<HistogramBin> histogramBins;
};

[[nodiscard]] FieldInspectorAnalysis analyzeField(
    const data::ByteDataSource& dataSource,
    const features::framing::FrameLayout& frameLayout,
    const FieldSelection& fieldSelection,
    int currentRow
);

}  // namespace bitabyte::features::inspector
