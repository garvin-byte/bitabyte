#pragma once

#include <QWidget>

#include "features/inspector/field_inspector_analysis.h"

class QLabel;

namespace bitabyte::ui {

class FieldInspectorPanel final : public QWidget {
    Q_OBJECT

public:
    explicit FieldInspectorPanel(QWidget* parent = nullptr);

    void clearAnalysis();
    void setPendingAnalysis(const QString& fieldLabel, const QString& positionText);
    void setAnalysis(const features::inspector::FieldInspectorAnalysis& analysis);

private:
    class HistogramWidget;

    static void setLabelText(QLabel* label, const QString& text);

    QLabel* fieldLabelValue_ = nullptr;
    QLabel* positionValue_ = nullptr;
    QLabel* currentRowValue_ = nullptr;
    QLabel* analyzedFramesValue_ = nullptr;
    QLabel* missingFramesValue_ = nullptr;
    QLabel* uniqueValuesValue_ = nullptr;
    QLabel* currentHexValue_ = nullptr;
    QLabel* currentBinaryValue_ = nullptr;
    QLabel* currentAsciiValue_ = nullptr;
    QLabel* currentUnsignedBigEndianValue_ = nullptr;
    QLabel* currentUnsignedLittleEndianValue_ = nullptr;
    QLabel* currentSignedBigEndianValue_ = nullptr;
    QLabel* currentSignedLittleEndianValue_ = nullptr;
    QLabel* currentFloatBigEndianValue_ = nullptr;
    QLabel* currentFloatLittleEndianValue_ = nullptr;
    QLabel* minimumValue_ = nullptr;
    QLabel* maximumValue_ = nullptr;
    QLabel* meanValue_ = nullptr;
    QLabel* modeValue_ = nullptr;
    QLabel* entropyValue_ = nullptr;
    QLabel* constantFlagValue_ = nullptr;
    QLabel* monotonicFlagValue_ = nullptr;
    QLabel* highEntropyFlagValue_ = nullptr;
    HistogramWidget* histogramWidget_ = nullptr;
};

}  // namespace bitabyte::ui
