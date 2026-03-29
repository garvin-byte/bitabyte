#include "ui/column_definitions_panel.h"

#include <QAbstractItemView>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QPushButton>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>
#include <QtGlobal>

#include <algorithm>

namespace bitabyte::ui {
namespace {

constexpr int kRowKindRole = Qt::UserRole;
constexpr int kRowPayloadRole = Qt::UserRole + 1;
constexpr int kSplitStartBitRole = Qt::UserRole + 2;
constexpr int kSplitEndBitRole = Qt::UserRole + 3;

QString definitionPositionLabel(const features::columns::ByteColumnDefinition& definition) {
    if (definition.unit == QStringLiteral("bit")) {
        const int endBit = definition.startBit + qMax(1, definition.totalBits) - 1;
        if (definition.startBit == endBit) {
            return QString::number(definition.startBit);
        }
        return QStringLiteral("%1-%2").arg(definition.startBit).arg(endBit);
    }

    if (definition.startByte == definition.endByte) {
        return QString::number(definition.startByte);
    }

    return QStringLiteral("%1-%2").arg(definition.startByte).arg(definition.endByte);
}

QString definitionFormatLabel(const QString& displayFormat) {
    if (displayFormat == QStringLiteral("binary")) {
        return QStringLiteral("Binary");
    }
    if (displayFormat == QStringLiteral("decimal")) {
        return QStringLiteral("Decimal");
    }
    if (displayFormat == QStringLiteral("ascii")) {
        return QStringLiteral("ASCII");
    }
    return QStringLiteral("Hex");
}

QString splitPositionLabel(const bitabyte::models::SplitPanelEntry& splitEntry) {
    if (splitEntry.startBit == splitEntry.endBit) {
        return QString::number(splitEntry.startBit);
    }
    return QStringLiteral("%1-%2").arg(splitEntry.startBit).arg(splitEntry.endBit);
}

}  // namespace

ColumnDefinitionsPanel::ColumnDefinitionsPanel(QWidget* parent)
    : QWidget(parent) {
    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(8, 8, 8, 8);
    rootLayout->setSpacing(8);

    treeWidget_ = new QTreeWidget(this);
    treeWidget_->setColumnCount(4);
    treeWidget_->setHeaderLabels({
        QStringLiteral("Name"),
        QStringLiteral("Position"),
        QStringLiteral("Format"),
        QStringLiteral("Color"),
    });
    treeWidget_->setRootIsDecorated(false);
    treeWidget_->setItemsExpandable(false);
    treeWidget_->setIndentation(0);
    treeWidget_->setUniformRowHeights(true);
    treeWidget_->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
    treeWidget_->setSelectionMode(QAbstractItemView::SelectionMode::SingleSelection);
    treeWidget_->header()->setStretchLastSection(false);
    treeWidget_->header()->setSectionResizeMode(0, QHeaderView::ResizeMode::Stretch);
    treeWidget_->header()->setSectionResizeMode(1, QHeaderView::ResizeMode::ResizeToContents);
    treeWidget_->header()->setSectionResizeMode(2, QHeaderView::ResizeMode::ResizeToContents);
    treeWidget_->header()->setSectionResizeMode(3, QHeaderView::ResizeMode::ResizeToContents);
    rootLayout->addWidget(treeWidget_, 1);

    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(6);

    addButton_ = new QPushButton(QStringLiteral("Add"), this);
    connect(addButton_, &QPushButton::clicked, this, &ColumnDefinitionsPanel::addRequested);
    buttonLayout->addWidget(addButton_);

    defineSelectionButton_ = new QPushButton(QStringLiteral("Define Selection"), this);
    connect(defineSelectionButton_, &QPushButton::clicked, this, &ColumnDefinitionsPanel::defineSelectionRequested);
    buttonLayout->addWidget(defineSelectionButton_);

    editButton_ = new QPushButton(QStringLiteral("Edit"), this);
    connect(editButton_, &QPushButton::clicked, this, [this]() {
        const int definitionIndex = currentDefinitionIndex();
        if (definitionIndex >= 0) {
            emit editRequested(definitionIndex);
            return;
        }

        const int splitStartBit = splitStartBitForItem(treeWidget_->currentItem());
        const int splitEndBit = splitEndBitForItem(treeWidget_->currentItem());
        if (splitStartBit >= 0 && splitEndBit >= splitStartBit) {
            emit editSplitRequested(splitStartBit, splitEndBit);
        }
    });
    buttonLayout->addWidget(editButton_);

    removeButton_ = new QPushButton(QStringLiteral("Remove"), this);
    connect(removeButton_, &QPushButton::clicked, this, [this]() {
        const int definitionIndex = currentDefinitionIndex();
        if (definitionIndex >= 0) {
            emit removeRequested(definitionIndex);
            return;
        }

        const int splitByteIndex = splitByteIndexForItem(treeWidget_->currentItem());
        if (splitByteIndex >= 0) {
            emit removeSplitRequested(splitByteIndex);
        }
    });
    buttonLayout->addWidget(removeButton_);

    rootLayout->addLayout(buttonLayout);

    connect(treeWidget_, &QTreeWidget::itemSelectionChanged, this, &ColumnDefinitionsPanel::updateButtonState);
    connect(treeWidget_, &QTreeWidget::itemDoubleClicked, this, [this](QTreeWidgetItem* item, int) {
        const int definitionIndex = definitionIndexForItem(item);
        if (definitionIndex >= 0) {
            emit editRequested(definitionIndex);
            return;
        }

        const int splitStartBit = splitStartBitForItem(item);
        const int splitEndBit = splitEndBitForItem(item);
        if (splitStartBit >= 0 && splitEndBit >= splitStartBit) {
            emit editSplitRequested(splitStartBit, splitEndBit);
        }
    });

    updateButtonState();
}

void ColumnDefinitionsPanel::setEntries(
    const QVector<features::columns::ByteColumnDefinition>& definitions,
    const QVector<bitabyte::models::SplitPanelEntry>& splitEntries
) {
    treeWidget_->clear();

    struct OrderedPanelEntry {
        int startBit = 0;
        int endBit = 0;
        int rowKindOrder = 0;
        int payloadIndex = 0;
        bool isDefinition = true;
    };

    QVector<OrderedPanelEntry> orderedEntries;
    for (int definitionIndex = 0; definitionIndex < definitions.size(); ++definitionIndex) {
        const features::columns::ByteColumnDefinition& definition = definitions[definitionIndex];
        OrderedPanelEntry orderedEntry;
        orderedEntry.isDefinition = true;
        orderedEntry.payloadIndex = definitionIndex;
        if (definition.unit == QStringLiteral("bit")) {
            orderedEntry.startBit = definition.startBit;
            orderedEntry.endBit = definition.startBit + qMax(1, definition.totalBits) - 1;
        } else {
            orderedEntry.startBit = definition.startByte * 8;
            orderedEntry.endBit = (definition.endByte + 1) * 8 - 1;
        }
        orderedEntries.append(orderedEntry);
    }

    for (int splitIndex = 0; splitIndex < splitEntries.size(); ++splitIndex) {
        const bitabyte::models::SplitPanelEntry& splitEntry = splitEntries[splitIndex];
        OrderedPanelEntry orderedEntry;
        orderedEntry.isDefinition = false;
        orderedEntry.payloadIndex = splitIndex;
        orderedEntry.startBit = splitEntry.startBit;
        orderedEntry.endBit = splitEntry.endBit;
        orderedEntry.rowKindOrder = 1;
        orderedEntries.append(orderedEntry);
    }

    std::sort(orderedEntries.begin(), orderedEntries.end(), [](const OrderedPanelEntry& leftEntry, const OrderedPanelEntry& rightEntry) {
        if (leftEntry.startBit != rightEntry.startBit) {
            return leftEntry.startBit < rightEntry.startBit;
        }
        if (leftEntry.endBit != rightEntry.endBit) {
            return leftEntry.endBit < rightEntry.endBit;
        }
        if (leftEntry.rowKindOrder != rightEntry.rowKindOrder) {
            return leftEntry.rowKindOrder < rightEntry.rowKindOrder;
        }
        return leftEntry.payloadIndex < rightEntry.payloadIndex;
    });

    for (const OrderedPanelEntry& orderedEntry : orderedEntries) {
        if (orderedEntry.isDefinition) {
            const features::columns::ByteColumnDefinition& definition = definitions[orderedEntry.payloadIndex];
            QTreeWidgetItem* item = new QTreeWidgetItem(treeWidget_);
            item->setText(0, definition.label.isEmpty() ? QStringLiteral("(unnamed)") : definition.label);
            item->setText(1, definitionPositionLabel(definition));
            item->setText(2, definitionFormatLabel(definition.displayFormat));
            item->setText(3, definition.colorName);
            item->setData(0, kRowKindRole, QStringLiteral("definition"));
            item->setData(0, kRowPayloadRole, orderedEntry.payloadIndex);
            continue;
        }

        const bitabyte::models::SplitPanelEntry& splitEntry = splitEntries[orderedEntry.payloadIndex];
        QTreeWidgetItem* item = new QTreeWidgetItem(treeWidget_);
        item->setText(0, splitEntry.label.isEmpty() ? QStringLiteral("(unnamed split)") : splitEntry.label);
        item->setText(1, splitPositionLabel(splitEntry));
        item->setText(2, definitionFormatLabel(splitEntry.displayFormat));
        item->setText(3, splitEntry.colorName);
        item->setData(0, kRowKindRole, QStringLiteral("split"));
        item->setData(0, kRowPayloadRole, splitEntry.payloadByteIndex);
        item->setData(0, kSplitStartBitRole, splitEntry.startBit);
        item->setData(0, kSplitEndBitRole, splitEntry.endBit);
    }

    updateButtonState();
}

int ColumnDefinitionsPanel::currentDefinitionIndex() const {
    return definitionIndexForItem(treeWidget_->currentItem());
}

void ColumnDefinitionsPanel::setSelectionEnabled(bool hasSelectableColumns) {
    defineSelectionButton_->setEnabled(hasSelectableColumns);
}

int ColumnDefinitionsPanel::definitionIndexForItem(const QTreeWidgetItem* item) const {
    if (item == nullptr) {
        return -1;
    }

    if (item->data(0, kRowKindRole).toString() != QStringLiteral("definition")) {
        return -1;
    }

    return item->data(0, kRowPayloadRole).toInt();
}

int ColumnDefinitionsPanel::splitByteIndexForItem(const QTreeWidgetItem* item) const {
    if (item == nullptr) {
        return -1;
    }

    if (item->data(0, kRowKindRole).toString() != QStringLiteral("split")) {
        return -1;
    }

    return item->data(0, kRowPayloadRole).toInt();
}

int ColumnDefinitionsPanel::splitStartBitForItem(const QTreeWidgetItem* item) const {
    if (item == nullptr || item->data(0, kRowKindRole).toString() != QStringLiteral("split")) {
        return -1;
    }

    return item->data(0, kSplitStartBitRole).toInt();
}

int ColumnDefinitionsPanel::splitEndBitForItem(const QTreeWidgetItem* item) const {
    if (item == nullptr || item->data(0, kRowKindRole).toString() != QStringLiteral("split")) {
        return -1;
    }

    return item->data(0, kSplitEndBitRole).toInt();
}

void ColumnDefinitionsPanel::updateButtonState() {
    const bool hasCurrentDefinition = currentDefinitionIndex() >= 0;
    const bool hasCurrentSplit = splitByteIndexForItem(treeWidget_->currentItem()) >= 0;
    editButton_->setEnabled(hasCurrentDefinition || hasCurrentSplit);
    removeButton_->setEnabled(hasCurrentDefinition || hasCurrentSplit);
}

}  // namespace bitabyte::ui
