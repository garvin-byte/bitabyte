#pragma once

#include <QWidget>

#include "features/inspector/field_inspector_analysis.h"

class QButtonGroup;
class QLabel;

namespace bitabyte::ui {

class FieldCurrentValuePanel final : public QWidget {
    Q_OBJECT

public:
    explicit FieldCurrentValuePanel(QWidget* parent = nullptr);

    void clearAnalysis();
    void setPendingAnalysis(const QString& fieldLabel, const QString& positionText);
    void setAnalysis(const features::inspector::FieldInspectorAnalysis& analysis);

private:
    void refresh();
    static void setLabelText(QLabel* label, const QString& text);

    features::inspector::FieldInspectorAnalysis lastAnalysis_;
    bool hasAnalysis_ = false;
    bool isPending_ = false;

    QButtonGroup* byteOrderGroup_ = nullptr;
    QLabel* hexValue_ = nullptr;
    QLabel* binaryValue_ = nullptr;
    QLabel* asciiValue_ = nullptr;
    QLabel* unsignedValue_ = nullptr;
    QLabel* signedValue_ = nullptr;
    QLabel* floatValue_ = nullptr;
};

}  // namespace bitabyte::ui
