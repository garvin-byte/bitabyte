#pragma once

#include <QWidget>

#include "features/columns/byte_column_definition.h"
#include "models/byte_table_model.h"

class QPushButton;
class QTreeWidget;
class QTreeWidgetItem;

namespace bitabyte::ui {

class ColumnDefinitionsPanel final : public QWidget {
    Q_OBJECT

public:
    explicit ColumnDefinitionsPanel(QWidget* parent = nullptr);

    void setEntries(
        const QVector<features::columns::ByteColumnDefinition>& definitions,
        const QVector<bitabyte::models::SplitPanelEntry>& splitEntries
    );
    [[nodiscard]] int currentDefinitionIndex() const;
    void setSelectionEnabled(bool hasSelectableColumns);

signals:
    void addRequested();
    void defineSelectionRequested();
    void editRequested(int definitionIndex);
    void editSplitRequested(int startBit, int endBit);
    void removeRequested(int definitionIndex);
    void removeSplitRequested(int byteIndex);

private:
    [[nodiscard]] int definitionIndexForItem(const QTreeWidgetItem* item) const;
    [[nodiscard]] int splitByteIndexForItem(const QTreeWidgetItem* item) const;
    [[nodiscard]] int splitStartBitForItem(const QTreeWidgetItem* item) const;
    [[nodiscard]] int splitEndBitForItem(const QTreeWidgetItem* item) const;
    void updateButtonState();

    QTreeWidget* treeWidget_ = nullptr;
    QPushButton* addButton_ = nullptr;
    QPushButton* defineSelectionButton_ = nullptr;
    QPushButton* editButton_ = nullptr;
    QPushButton* removeButton_ = nullptr;
};

}  // namespace bitabyte::ui
