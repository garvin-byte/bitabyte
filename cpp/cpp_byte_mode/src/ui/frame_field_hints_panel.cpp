#include "ui/frame_field_hints_panel.h"

#include <QColor>
#include <QHeaderView>
#include <QLabel>
#include <QList>
#include <QPushButton>
#include <QStringList>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVariant>
#include <QVBoxLayout>

#include <algorithm>

namespace bitabyte::ui {
namespace {

const QColor kCounterAccentColor(212, 232, 255);
const QColor kConstantAccentColor(255, 247, 196);
constexpr int kStartBitRole = Qt::UserRole;
constexpr int kEndBitRole = Qt::UserRole + 1;
constexpr int kKindRole = Qt::UserRole + 2;
constexpr int kValueTextRole = Qt::UserRole + 3;
constexpr int kFieldColumn = 0;
constexpr int kHintColumn = 1;
constexpr int kAddColumnColumn = 2;

struct DisplayHintEntry {
    enum class Kind {
        Counter,
        Constant,
    };

    QString label;
    QString summaryText;
    QString valueText;
    QColor accentColor;
    int startBit = -1;
    int endBit = -1;
    Kind kind = Kind::Counter;
};

bool overlapsDefinitionRange(
    int startBit,
    int endBit,
    const QVector<features::columns::ByteColumnDefinition>& existingDefinitions
) {
    if (startBit < 0 || endBit < startBit) {
        return false;
    }

    for (const features::columns::ByteColumnDefinition& definition : existingDefinitions) {
        if (!(endBit < definition.startAbsoluteBit() || startBit > definition.endAbsoluteBit())) {
            return true;
        }
    }
    return false;
}

QString formatRangeLabel(int startBit, int endBit) {
    if (startBit < 0 || endBit < startBit) {
        return QStringLiteral("-");
    }

    if ((startBit % 8) == 0 && (endBit % 8) == 7) {
        const int startByte = startBit / 8;
        const int endByte = endBit / 8;
        if (startByte == endByte) {
            return QStringLiteral("Byte %1").arg(startByte);
        }
        return QStringLiteral("Bytes %1-%2").arg(startByte).arg(endByte);
    }

    if (startBit == endBit) {
        return QStringLiteral("Bit %1").arg(startBit);
    }
    return QStringLiteral("Bits %1-%2").arg(startBit).arg(endBit);
}

QString joinConstantValues(const QStringList& valueTexts) {
    if (valueTexts.isEmpty()) {
        return QStringLiteral("-");
    }

    bool allHexPrefixed = true;
    for (const QString& valueText : valueTexts) {
        if (!valueText.startsWith(QStringLiteral("0x"), Qt::CaseInsensitive)) {
            allHexPrefixed = false;
            break;
        }
    }

    if (!allHexPrefixed) {
        return valueTexts.join(QLatin1Char(' '));
    }

    QStringList normalizedValues;
    normalizedValues.reserve(valueTexts.size());
    for (int valueIndex = 0; valueIndex < valueTexts.size(); ++valueIndex) {
        normalizedValues.append(
            valueIndex == 0 ? valueTexts.at(valueIndex) : valueTexts.at(valueIndex).mid(2)
        );
    }
    return normalizedValues.join(QLatin1Char(' '));
}

QVector<DisplayHintEntry> groupedConstantEntries(
    const QVector<features::classification::FrameFieldHint>& constantHints
) {
    QVector<DisplayHintEntry> entries;
    int constantIndex = 0;
    while (constantIndex < constantHints.size()) {
        const features::classification::FrameFieldHint& firstHint = constantHints.at(constantIndex);
        DisplayHintEntry entry;
        entry.accentColor = kConstantAccentColor;
        entry.kind = DisplayHintEntry::Kind::Constant;
        entry.startBit = firstHint.absoluteStartBit;
        int groupedEndBit = firstHint.absoluteEndBit;
        QStringList valueTexts{firstHint.valueText};

        int nextConstantIndex = constantIndex + 1;
        while (nextConstantIndex < constantHints.size()) {
            const features::classification::FrameFieldHint& nextHint = constantHints.at(nextConstantIndex);
            if (nextHint.absoluteStartBit != groupedEndBit + 1) {
                break;
            }
            groupedEndBit = nextHint.absoluteEndBit;
            valueTexts.append(nextHint.valueText);
            ++nextConstantIndex;
        }

        entry.label = formatRangeLabel(firstHint.absoluteStartBit, groupedEndBit);
        entry.valueText = joinConstantValues(valueTexts);
        entry.summaryText = QStringLiteral("Constant %1").arg(entry.valueText);
        entry.endBit = groupedEndBit;
        entries.append(entry);
        constantIndex = nextConstantIndex;
    }
    return entries;
}

}  // namespace

FrameFieldHintsPanel::FrameFieldHintsPanel(Mode mode, QWidget* parent)
    : QWidget(parent),
      mode_(mode) {
    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(6);

    summaryLabel_ = new QLabel(this);
    summaryLabel_->setWordWrap(true);
    rootLayout->addWidget(summaryLabel_);

    treeWidget_ = new QTreeWidget(this);
    treeWidget_->setColumnCount(3);
    treeWidget_->setHeaderLabels({
        QStringLiteral("Field"),
        QStringLiteral("Hint"),
        QStringLiteral("Add"),
    });
    treeWidget_->header()->setStretchLastSection(true);
    treeWidget_->header()->setSectionResizeMode(kFieldColumn, QHeaderView::ResizeToContents);
    treeWidget_->header()->setSectionResizeMode(kHintColumn, QHeaderView::Stretch);
    treeWidget_->header()->setSectionResizeMode(kAddColumnColumn, QHeaderView::ResizeToContents);
    treeWidget_->setColumnHidden(kAddColumnColumn, false);
    treeWidget_->setRootIsDecorated(false);
    treeWidget_->setIndentation(0);
    treeWidget_->setUniformRowHeights(true);
    rootLayout->addWidget(treeWidget_, 1);

    connect(treeWidget_, &QTreeWidget::itemClicked, this, [this](QTreeWidgetItem* treeItem, int column) {
        maybeEmitRequestedColumn(treeItem, column);
    });
    connect(treeWidget_, &QTreeWidget::itemActivated, this, [this](QTreeWidgetItem* treeItem, int column) {
        maybeEmitRequestedColumn(treeItem, column);
    });

    showUnavailable();
}

void FrameFieldHintsPanel::showUnavailable() {
    summaryLabel_->setText(QStringLiteral("Framing hints appear after framing is active."));
    treeWidget_->clear();
}

void FrameFieldHintsPanel::setPendingAnalysis() {
    summaryLabel_->setText(QStringLiteral("Analyzing framed columns for counters and constants..."));
    treeWidget_->clear();
}

void FrameFieldHintsPanel::setHints(
    const features::classification::FrameFieldClassificationResult& classificationResult,
    const QVector<features::columns::ByteColumnDefinition>& existingDefinitions
) {
    treeWidget_->clear();
    const QVector<DisplayHintEntry> constantEntries = groupedConstantEntries(classificationResult.constantHints);
    QVector<DisplayHintEntry> displayEntries;
    displayEntries.reserve(constantEntries.size() + classificationResult.counterHints.size());

    for (const DisplayHintEntry& constantEntry : constantEntries) {
        displayEntries.append(constantEntry);
    }

    for (const features::classification::FrameFieldHint& counterHint : classificationResult.counterHints) {
        DisplayHintEntry entry;
        entry.label = counterHint.label;
        entry.summaryText = counterHint.summaryText;
        entry.accentColor = kCounterAccentColor;
        entry.kind = DisplayHintEntry::Kind::Counter;
        entry.startBit = counterHint.absoluteStartBit;
        entry.endBit = counterHint.absoluteEndBit;
        displayEntries.append(entry);
    }

    std::sort(
        displayEntries.begin(),
        displayEntries.end(),
        [](const DisplayHintEntry& leftEntry, const DisplayHintEntry& rightEntry) {
            return leftEntry.startBit < rightEntry.startBit;
        }
    );

    if (displayEntries.isEmpty()) {
        summaryLabel_->setText(QStringLiteral("No counters or constants detected."));
        QTreeWidgetItem* emptyItem = new QTreeWidgetItem(treeWidget_);
        emptyItem->setText(kFieldColumn, QStringLiteral("None"));
        emptyItem->setFlags(emptyItem->flags() & ~Qt::ItemIsSelectable);
        return;
    }

    int visibleCounterFieldCount = 0;
    int visibleConstantFieldCount = 0;
    for (const DisplayHintEntry& entry : displayEntries) {
        if (overlapsDefinitionRange(entry.startBit, entry.endBit, existingDefinitions)) {
            continue;
        }

        QTreeWidgetItem* hintItem = new QTreeWidgetItem(treeWidget_);
        hintItem->setText(kFieldColumn, entry.label);
        hintItem->setText(kHintColumn, entry.summaryText);
        hintItem->setData(kFieldColumn, kStartBitRole, entry.startBit);
        hintItem->setData(kFieldColumn, kEndBitRole, entry.endBit);
        hintItem->setData(kFieldColumn, kKindRole, entry.kind == DisplayHintEntry::Kind::Constant);
        hintItem->setData(kFieldColumn, kValueTextRole, entry.valueText);
        hintItem->setBackground(kFieldColumn, entry.accentColor);
        hintItem->setBackground(kHintColumn, entry.accentColor);
        if (mode_ == Mode::AddColumnSelection) {
            hintItem->setFlags(hintItem->flags() | Qt::ItemIsUserCheckable);
            hintItem->setCheckState(kAddColumnColumn, Qt::Checked);
            hintItem->setBackground(kAddColumnColumn, entry.accentColor);
        } else {
            QPushButton* addButton = new QPushButton(QStringLiteral("Add"), treeWidget_);
            addButton->setAutoDefault(false);
            addButton->setDefault(false);
            connect(addButton, &QPushButton::clicked, this, [this, entry]() {
                emit addColumnRequested(
                    entry.startBit,
                    entry.endBit,
                    entry.kind == DisplayHintEntry::Kind::Constant,
                    entry.label,
                    entry.valueText
                );
            });
            treeWidget_->setItemWidget(hintItem, kAddColumnColumn, addButton);
        }

        if (entry.kind == DisplayHintEntry::Kind::Constant) {
            ++visibleConstantFieldCount;
        } else {
            ++visibleCounterFieldCount;
        }
    }

    if (treeWidget_->topLevelItemCount() == 0) {
        summaryLabel_->setText(QStringLiteral("All detected fields already have column definitions."));
        QTreeWidgetItem* emptyItem = new QTreeWidgetItem(treeWidget_);
        emptyItem->setText(kFieldColumn, QStringLiteral("Already defined"));
        emptyItem->setFlags(emptyItem->flags() & ~Qt::ItemIsSelectable);
        return;
    }

    summaryLabel_->setText(
        QStringLiteral("Detected %1 possible counter%2 and %3 constant field%4.")
            .arg(visibleCounterFieldCount)
            .arg(visibleCounterFieldCount == 1 ? QString() : QStringLiteral("s"))
            .arg(visibleConstantFieldCount)
            .arg(visibleConstantFieldCount == 1 ? QString() : QStringLiteral("s"))
    );
}

QVector<features::classification::FrameFieldHint> FrameFieldHintsPanel::selectedColumnHints() const {
    QVector<features::classification::FrameFieldHint> selectedHints;
    if (treeWidget_ == nullptr || mode_ != Mode::AddColumnSelection) {
        return selectedHints;
    }

    selectedHints.reserve(treeWidget_->topLevelItemCount());
    for (int itemIndex = 0; itemIndex < treeWidget_->topLevelItemCount(); ++itemIndex) {
        QTreeWidgetItem* hintItem = treeWidget_->topLevelItem(itemIndex);
        if (hintItem == nullptr || hintItem->checkState(kAddColumnColumn) != Qt::Checked) {
            continue;
        }

        const QVariant startBitData = hintItem->data(kFieldColumn, kStartBitRole);
        const QVariant endBitData = hintItem->data(kFieldColumn, kEndBitRole);
        if (!startBitData.isValid() || !endBitData.isValid()) {
            continue;
        }

        features::classification::FrameFieldHint selectedHint;
        selectedHint.absoluteStartBit = startBitData.toInt();
        selectedHint.absoluteEndBit = endBitData.toInt();
        selectedHint.label = hintItem->text(kFieldColumn);
        selectedHint.summaryText = hintItem->text(kHintColumn);
        if (selectedHint.summaryText.startsWith(QStringLiteral("Constant "))) {
            selectedHint.valueText = selectedHint.summaryText.mid(QStringLiteral("Constant ").size());
        }
        selectedHints.append(selectedHint);
    }
    return selectedHints;
}

void FrameFieldHintsPanel::maybeEmitRequestedColumn(QTreeWidgetItem* treeItem, int column) {
    if (treeItem == nullptr) {
        return;
    }
    if (mode_ == Mode::AddColumnSelection && column == kAddColumnColumn) {
        return;
    }

    const QVariant startBitData = treeItem->data(kFieldColumn, kStartBitRole);
    const QVariant endBitData = treeItem->data(kFieldColumn, kEndBitRole);
    if (!startBitData.isValid() || !endBitData.isValid()) {
        return;
    }

    const int startBit = startBitData.toInt();
    const int endBit = endBitData.toInt();
    if (mode_ == Mode::BrowseOnly && column == kAddColumnColumn) {
        emit addColumnRequested(
            startBit,
            endBit,
            treeItem->data(kFieldColumn, kKindRole).toBool(),
            treeItem->text(kFieldColumn),
            treeItem->data(kFieldColumn, kValueTextRole).toString()
        );
        return;
    }

    if (startBit >= 0 && endBit >= startBit) {
        emit bitRangeRequested(startBit, endBit);
    }
}

}  // namespace bitabyte::ui
