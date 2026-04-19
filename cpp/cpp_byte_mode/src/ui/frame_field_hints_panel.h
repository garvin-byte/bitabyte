#pragma once

#include <QSet>
#include <QWidget>

#include "features/classification/frame_field_classification.h"
#include "features/columns/byte_column_definition.h"

class QColor;
class QLabel;
class QTreeWidget;
class QTreeWidgetItem;

namespace bitabyte::ui {

class FrameFieldHintsPanel final : public QWidget {
    Q_OBJECT

public:
    enum class Mode {
        BrowseOnly,
        AddColumnSelection,
    };

    explicit FrameFieldHintsPanel(Mode mode = Mode::BrowseOnly, QWidget* parent = nullptr);

    void showUnavailable();
    void setPendingAnalysis();
    void setHints(
        const features::classification::FrameFieldClassificationResult& classificationResult,
        const QVector<features::columns::ByteColumnDefinition>& existingDefinitions = {}
    );
    [[nodiscard]] QVector<features::classification::FrameFieldHint> selectedColumnHints() const;

signals:
    void bitRangeRequested(int startBit, int endBit);
    void addColumnRequested(int startBit, int endBit, bool isConstant, const QString& label, const QString& valueText);

private:
    void maybeEmitRequestedColumn(QTreeWidgetItem* treeItem, int column);

    Mode mode_ = Mode::BrowseOnly;
    QLabel* summaryLabel_ = nullptr;
    QTreeWidget* treeWidget_ = nullptr;
};

}  // namespace bitabyte::ui
