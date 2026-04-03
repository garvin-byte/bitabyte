#pragma once

#include <QWidget>

#include "features/inspector/field_inspector_analysis.h"

namespace bitabyte::ui {

class FieldInspectorPanel final : public QWidget {
    Q_OBJECT

public:
    explicit FieldInspectorPanel(QWidget* parent = nullptr);

    void clearAnalysis();
    void showUnavailable();
    void setPendingAnalysis(const QString& fieldLabel, const QString& positionText);
    void setAnalysis(const features::inspector::FieldInspectorAnalysis& analysis);

private:
    class HistogramWidget;

    HistogramWidget* histogramWidget_ = nullptr;
};

}  // namespace bitabyte::ui
