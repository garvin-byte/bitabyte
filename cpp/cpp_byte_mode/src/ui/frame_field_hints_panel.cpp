#include "ui/frame_field_hints_panel.h"

#include <QColor>
#include <QHeaderView>
#include <QLabel>
#include <QList>
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

struct DisplayHintEntry {
    QString label;
    QString summaryText;
    QColor accentColor;
    int startBit = -1;
    int endBit = -1;
};

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
        entry.summaryText = QStringLiteral("Constant %1").arg(joinConstantValues(valueTexts));
        entry.endBit = groupedEndBit;
        entries.append(entry);
        constantIndex = nextConstantIndex;
    }
    return entries;
}

}  // namespace

FrameFieldHintsPanel::FrameFieldHintsPanel(QWidget* parent)
    : QWidget(parent) {
    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(6);

    summaryLabel_ = new QLabel(this);
    summaryLabel_->setWordWrap(true);
    rootLayout->addWidget(summaryLabel_);

    treeWidget_ = new QTreeWidget(this);
    treeWidget_->setColumnCount(2);
    treeWidget_->setHeaderLabels({QStringLiteral("Field"), QStringLiteral("Hint")});
    treeWidget_->header()->setStretchLastSection(true);
    treeWidget_->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    treeWidget_->setRootIsDecorated(false);
    treeWidget_->setIndentation(0);
    treeWidget_->setUniformRowHeights(true);
    rootLayout->addWidget(treeWidget_, 1);

    connect(treeWidget_, &QTreeWidget::itemClicked, this, [this](QTreeWidgetItem* treeItem, int) {
        maybeEmitRequestedColumn(treeItem);
    });
    connect(treeWidget_, &QTreeWidget::itemActivated, this, [this](QTreeWidgetItem* treeItem, int) {
        maybeEmitRequestedColumn(treeItem);
    });

    showUnavailable();
}

void FrameFieldHintsPanel::showUnavailable() {
    summaryLabel_->setText(QStringLiteral("Find Frames hints appear after framing is active."));
    treeWidget_->clear();
}

void FrameFieldHintsPanel::setPendingAnalysis() {
    summaryLabel_->setText(QStringLiteral("Analyzing framed columns for counters and constants..."));
}

void FrameFieldHintsPanel::setHints(
    const features::classification::FrameFieldClassificationResult& classificationResult
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
        entry.startBit = counterHint.absoluteStartBit;
        entry.endBit = counterHint.absoluteEndBit;
        displayEntries.append(entry);
    }

    std::sort(displayEntries.begin(), displayEntries.end(), [](const DisplayHintEntry& leftEntry, const DisplayHintEntry& rightEntry) {
        return leftEntry.startBit < rightEntry.startBit;
    });

    const int constantFieldCount = classificationResult.constantHints.size();
    const int counterFieldCount = classificationResult.counterHints.size();
    if (displayEntries.isEmpty()) {
        summaryLabel_->setText(QStringLiteral("No counters or constants detected."));
        QTreeWidgetItem* emptyItem = new QTreeWidgetItem(treeWidget_);
        emptyItem->setText(0, QStringLiteral("None"));
        emptyItem->setFlags(emptyItem->flags() & ~Qt::ItemIsSelectable);
        return;
    }

    summaryLabel_->setText(
        QStringLiteral("Detected %1 possible counter%2 and %3 constant field%4.")
            .arg(counterFieldCount)
            .arg(counterFieldCount == 1 ? QString() : QStringLiteral("s"))
            .arg(constantFieldCount)
            .arg(constantFieldCount == 1 ? QString() : QStringLiteral("s"))
    );

    for (const DisplayHintEntry& entry : displayEntries) {
        QTreeWidgetItem* hintItem = new QTreeWidgetItem(treeWidget_);
        hintItem->setText(0, entry.label);
        hintItem->setText(1, entry.summaryText);
        hintItem->setData(0, kStartBitRole, entry.startBit);
        hintItem->setData(0, kEndBitRole, entry.endBit);
        hintItem->setBackground(0, entry.accentColor);
        hintItem->setBackground(1, entry.accentColor);
    }
}

void FrameFieldHintsPanel::maybeEmitRequestedColumn(QTreeWidgetItem* treeItem) {
    if (treeItem == nullptr) {
        return;
    }

    const QVariant startBitData = treeItem->data(0, kStartBitRole);
    const QVariant endBitData = treeItem->data(0, kEndBitRole);
    if (!startBitData.isValid() || !endBitData.isValid()) {
        return;
    }

    const int startBit = startBitData.toInt();
    const int endBit = endBitData.toInt();
    if (startBit >= 0 && endBit >= startBit) {
        emit bitRangeRequested(startBit, endBit);
    }
}

}  // namespace bitabyte::ui
