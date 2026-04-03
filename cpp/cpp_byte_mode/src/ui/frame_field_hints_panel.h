#pragma once

#include <QSet>
#include <QWidget>

#include "features/classification/frame_field_classification.h"

class QColor;
class QLabel;
class QTreeWidget;
class QTreeWidgetItem;

namespace bitabyte::ui {

class FrameFieldHintsPanel final : public QWidget {
    Q_OBJECT

public:
    explicit FrameFieldHintsPanel(QWidget* parent = nullptr);

    void showUnavailable();
    void setPendingAnalysis();
    void setHints(const features::classification::FrameFieldClassificationResult& classificationResult);

signals:
    void bitRangeRequested(int startBit, int endBit);

private:
    void maybeEmitRequestedColumn(QTreeWidgetItem* treeItem);

    QLabel* summaryLabel_ = nullptr;
    QTreeWidget* treeWidget_ = nullptr;
};

}  // namespace bitabyte::ui
