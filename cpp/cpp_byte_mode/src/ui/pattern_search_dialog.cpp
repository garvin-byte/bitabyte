#include "ui/pattern_search_dialog.h"

#include "core/byte_format_utils.h"
#include "data/byte_data_source.h"
#include "features/frame_sync/frame_sync_search.h"

#include <QAbstractItemView>
#include <QCloseEvent>
#include <QComboBox>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include <algorithm>
#include <QHash>

namespace bitabyte::ui {
namespace {

constexpr int kMatchIndexColumn = 0;
constexpr int kByteOffsetColumn = 1;
constexpr int kBitOffsetColumn = 2;
constexpr int kStartColumn = 3;
constexpr int kDistanceBitsColumn = 4;
constexpr int kRepeatedPatternRankColumn = 0;
constexpr int kRepeatedPatternValueColumn = 1;
constexpr int kRepeatedPatternRepeatCountColumn = 2;
constexpr int kRepeatedPatternFirstStartColumn = 3;
constexpr qsizetype kMaximumRepeatedPatternWindows = 500000;

struct RepeatedPatternEntry {
    QByteArray bits;
    int repeatCount = 0;
    qsizetype firstStartBit = 0;
};

QString matchStartLabel(qsizetype startBit) {
    const qsizetype byteOffset = startBit / 8;
    const int bitOffset = static_cast<int>(startBit % 8);
    if (bitOffset == 0) {
        return core::formatByteOffset(byteOffset);
    }

    return QStringLiteral("%1+%2b").arg(core::formatByteOffset(byteOffset)).arg(bitOffset);
}

QString bitsAsBinaryText(const QByteArray& bits) {
    QString binaryText;
    binaryText.reserve(bits.size());
    for (char bitValue : bits) {
        binaryText.append(bitValue != 0 ? QLatin1Char('1') : QLatin1Char('0'));
    }
    return binaryText;
}

QString patternDisplayText(const QByteArray& bits) {
    return core::formatBitsAsHexWithTrailingBits(bits, true);
}

QString searchTextForBits(const QByteArray& bits) {
    if (bits.isEmpty()) {
        return {};
    }

    return (bits.size() % 4) == 0
        ? core::formatBitsAsHexWithTrailingBits(bits, true)
        : bitsAsBinaryText(bits);
}

}  // namespace

PatternSearchDialog::PatternSearchDialog(
    data::ByteDataSource* dataSource,
    const QString& initialPatternText,
    QWidget* parent
)
    : QDialog(parent),
      dataSource_(dataSource) {
    setModal(true);
    setWindowTitle(QStringLiteral("Find Pattern"));
    resize(900, 720);

    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(10, 10, 10, 10);
    rootLayout->setSpacing(8);

    QHBoxLayout* searchLayout = new QHBoxLayout();
    searchLayout->setSpacing(8);
    searchLayout->addWidget(new QLabel(QStringLiteral("Pattern:"), this));

    patternLineEdit_ = new QLineEdit(this);
    patternLineEdit_->setPlaceholderText(QStringLiteral("0x1ACF, 1011011, or 0b1011011"));
    patternLineEdit_->setText(initialPatternText);
    searchLayout->addWidget(patternLineEdit_, 1);

    findButton_ = new QPushButton(QStringLiteral("Find"), this);
    connect(findButton_, &QPushButton::clicked, this, &PatternSearchDialog::runSearch);
    searchLayout->addWidget(findButton_);

    rootLayout->addLayout(searchLayout);

    QSplitter* contentSplitter = new QSplitter(Qt::Vertical, this);
    contentSplitter->setChildrenCollapsible(false);

    QWidget* matchesPanel = new QWidget(contentSplitter);
    QVBoxLayout* matchesLayout = new QVBoxLayout(matchesPanel);
    matchesLayout->setContentsMargins(0, 0, 0, 0);
    matchesLayout->setSpacing(8);

    summaryLabel_ = new QLabel(QStringLiteral("Search for a hex or bit pattern at any bit offset."), this);
    summaryLabel_->setWordWrap(true);
    matchesLayout->addWidget(summaryLabel_);

    resultsTableWidget_ = new QTableWidget(matchesPanel);
    resultsTableWidget_->setColumnCount(5);
    resultsTableWidget_->setHorizontalHeaderLabels({
        QStringLiteral("#"),
        QStringLiteral("Byte Offset"),
        QStringLiteral("Bit"),
        QStringLiteral("Start"),
        QStringLiteral("Distance (Bits)"),
    });
    resultsTableWidget_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    resultsTableWidget_->setSelectionBehavior(QAbstractItemView::SelectRows);
    resultsTableWidget_->setSelectionMode(QAbstractItemView::SingleSelection);
    resultsTableWidget_->setAlternatingRowColors(false);
    resultsTableWidget_->verticalHeader()->setVisible(false);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kMatchIndexColumn, QHeaderView::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kByteOffsetColumn, QHeaderView::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kBitOffsetColumn, QHeaderView::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kStartColumn, QHeaderView::Stretch);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kDistanceBitsColumn, QHeaderView::ResizeToContents);
    connect(
        resultsTableWidget_,
        &QTableWidget::itemSelectionChanged,
        this,
        &PatternSearchDialog::handleCurrentMatchChanged
    );
    matchesLayout->addWidget(resultsTableWidget_, 1);
    contentSplitter->addWidget(matchesPanel);

    QGroupBox* repeatedPatternsGroup = new QGroupBox(QStringLiteral("Find Patterns"), contentSplitter);
    QVBoxLayout* repeatedPatternsLayout = new QVBoxLayout(repeatedPatternsGroup);
    repeatedPatternsLayout->setContentsMargins(8, 8, 8, 8);
    repeatedPatternsLayout->setSpacing(8);

    QHBoxLayout* repeatedPatternControlsLayout = new QHBoxLayout();
    repeatedPatternControlsLayout->setSpacing(8);
    repeatedPatternControlsLayout->addWidget(new QLabel(QStringLiteral("Size:"), repeatedPatternsGroup));

    repeatedPatternSizeSpinBox_ = new QSpinBox(repeatedPatternsGroup);
    repeatedPatternSizeSpinBox_->setRange(1, 4096);
    repeatedPatternSizeSpinBox_->setValue(2);
    repeatedPatternControlsLayout->addWidget(repeatedPatternSizeSpinBox_);

    repeatedPatternUnitComboBox_ = new QComboBox(repeatedPatternsGroup);
    repeatedPatternUnitComboBox_->addItem(QStringLiteral("Bits"));
    repeatedPatternUnitComboBox_->addItem(QStringLiteral("Bytes"));
    repeatedPatternUnitComboBox_->setCurrentText(QStringLiteral("Bytes"));
    repeatedPatternControlsLayout->addWidget(repeatedPatternUnitComboBox_);

    discoverPatternsButton_ = new QPushButton(QStringLiteral("Find Patterns"), repeatedPatternsGroup);
    connect(discoverPatternsButton_, &QPushButton::clicked, this, &PatternSearchDialog::discoverRepeatedPatterns);
    repeatedPatternControlsLayout->addWidget(discoverPatternsButton_);

    repeatedPatternControlsLayout->addStretch();
    repeatedPatternsLayout->addLayout(repeatedPatternControlsLayout);

    repeatedPatternSummaryLabel_ =
        new QLabel(QStringLiteral("Set a size and click Find Patterns to rank repeated patterns."), repeatedPatternsGroup);
    repeatedPatternSummaryLabel_->setWordWrap(true);
    repeatedPatternsLayout->addWidget(repeatedPatternSummaryLabel_);

    repeatedPatternsTableWidget_ = new QTableWidget(repeatedPatternsGroup);
    repeatedPatternsTableWidget_->setColumnCount(4);
    repeatedPatternsTableWidget_->setHorizontalHeaderLabels({
        QStringLiteral("#"),
        QStringLiteral("Pattern"),
        QStringLiteral("Repeats"),
        QStringLiteral("First Start"),
    });
    repeatedPatternsTableWidget_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    repeatedPatternsTableWidget_->setSelectionBehavior(QAbstractItemView::SelectRows);
    repeatedPatternsTableWidget_->setSelectionMode(QAbstractItemView::SingleSelection);
    repeatedPatternsTableWidget_->setAlternatingRowColors(false);
    repeatedPatternsTableWidget_->verticalHeader()->setVisible(false);
    repeatedPatternsTableWidget_->horizontalHeader()->setSectionResizeMode(
        kRepeatedPatternRankColumn,
        QHeaderView::ResizeToContents
    );
    repeatedPatternsTableWidget_->horizontalHeader()->setSectionResizeMode(
        kRepeatedPatternValueColumn,
        QHeaderView::Stretch
    );
    repeatedPatternsTableWidget_->horizontalHeader()->setSectionResizeMode(
        kRepeatedPatternRepeatCountColumn,
        QHeaderView::ResizeToContents
    );
    repeatedPatternsTableWidget_->horizontalHeader()->setSectionResizeMode(
        kRepeatedPatternFirstStartColumn,
        QHeaderView::ResizeToContents
    );
    connect(
        repeatedPatternsTableWidget_,
        &QTableWidget::itemSelectionChanged,
        this,
        &PatternSearchDialog::handleDiscoveredPatternChanged
    );
    repeatedPatternsLayout->addWidget(repeatedPatternsTableWidget_, 1);
    contentSplitter->addWidget(repeatedPatternsGroup);

    contentSplitter->setStretchFactor(0, 3);
    contentSplitter->setStretchFactor(1, 2);
    contentSplitter->setSizes({360, 260});
    rootLayout->addWidget(contentSplitter, 1);

    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(8);
    frameButton_ = new QPushButton(QStringLiteral("Frame by Pattern"), this);
    frameButton_->setEnabled(false);
    connect(frameButton_, &QPushButton::clicked, this, &PatternSearchDialog::requestFrameByPattern);
    buttonLayout->addWidget(frameButton_);

    buttonLayout->addStretch();

    QPushButton* closeButton = new QPushButton(QStringLiteral("Close"), this);
    connect(closeButton, &QPushButton::clicked, this, &QDialog::close);
    buttonLayout->addWidget(closeButton);
    rootLayout->addLayout(buttonLayout);

    connect(patternLineEdit_, &QLineEdit::returnPressed, this, &PatternSearchDialog::runSearch);

    if (!initialPatternText.trimmed().isEmpty()) {
        runSearch();
    }
}

void PatternSearchDialog::closeEvent(QCloseEvent* closeEvent) {
    emit highlightRangesChanged({});
    QDialog::closeEvent(closeEvent);
}

void PatternSearchDialog::runSearch() {
    if (dataSource_ == nullptr || !dataSource_->hasData()) {
        clearResults(QStringLiteral("Load a file before searching for a pattern."));
        return;
    }

    QString errorMessage;
    const std::optional<features::frame_sync::PatternSearchResult> searchResult =
        features::frame_sync::FrameSyncSearch::findPatternMatches(
            *dataSource_,
            patternLineEdit_ != nullptr ? patternLineEdit_->text() : QString{},
            &errorMessage
        );
    if (!searchResult.has_value()) {
        clearResults(errorMessage);
        return;
    }

    patternBitCount_ = searchResult->patternBitCount;
    matchStartBits_ = searchResult->matchStartBits;
    refreshResultsTable();

    QVector<QPair<qsizetype, qsizetype>> highlightRanges;
    highlightRanges.reserve(matchStartBits_.size());
    for (qsizetype matchStartBit : matchStartBits_) {
        highlightRanges.append(qMakePair(matchStartBit, matchStartBit + patternBitCount_ - 1));
    }
    emit highlightRangesChanged(highlightRanges);

    summaryLabel_->setText(
        QStringLiteral("%1 match%2 found | pattern length %3 bit%4")
            .arg(matchStartBits_.size())
            .arg(matchStartBits_.size() == 1 ? QString() : QStringLiteral("es"))
            .arg(patternBitCount_)
            .arg(patternBitCount_ == 1 ? QString() : QStringLiteral("s"))
    );

    const bool hasMatches = !matchStartBits_.isEmpty();
    frameButton_->setEnabled(hasMatches);
    if (hasMatches) {
        resultsTableWidget_->selectRow(0);
        handleCurrentMatchChanged();
    }
}

void PatternSearchDialog::handleCurrentMatchChanged() {
    if (!hasCurrentMatch()) {
        return;
    }

    emit focusMatchRequested(currentMatchStartBit(), patternBitCount_);
}

void PatternSearchDialog::requestFrameByPattern() {
    const QString patternText = patternLineEdit_ != nullptr ? patternLineEdit_->text().trimmed() : QString{};
    if (patternText.isEmpty()) {
        return;
    }

    emit frameByPatternRequested(patternText);
}

void PatternSearchDialog::refreshResultsTable() {
    resultsTableWidget_->clearContents();
    resultsTableWidget_->setRowCount(matchStartBits_.size());

    for (int matchIndex = 0; matchIndex < matchStartBits_.size(); ++matchIndex) {
        const qsizetype matchStartBit = matchStartBits_.at(matchIndex);
        const qsizetype byteOffset = matchStartBit / 8;
        const int bitOffsetInByte = static_cast<int>(matchStartBit % 8);
        const QString distanceText = matchIndex <= 0
            ? QStringLiteral("-")
            : QString::number(matchStartBit - matchStartBits_.at(matchIndex - 1));

        auto makeItem = [](const QString& text) {
            QTableWidgetItem* item = new QTableWidgetItem(text);
            item->setTextAlignment(Qt::AlignCenter);
            return item;
        };

        resultsTableWidget_->setItem(matchIndex, kMatchIndexColumn, makeItem(QString::number(matchIndex + 1)));
        resultsTableWidget_->setItem(matchIndex, kByteOffsetColumn, makeItem(core::formatByteOffset(byteOffset)));
        resultsTableWidget_->setItem(matchIndex, kBitOffsetColumn, makeItem(QString::number(bitOffsetInByte)));
        resultsTableWidget_->setItem(matchIndex, kStartColumn, makeItem(matchStartLabel(matchStartBit)));
        resultsTableWidget_->setItem(matchIndex, kDistanceBitsColumn, makeItem(distanceText));
    }
}

void PatternSearchDialog::clearResults(const QString& summaryText) {
    patternBitCount_ = 0;
    matchStartBits_.clear();
    resultsTableWidget_->clearContents();
    resultsTableWidget_->setRowCount(0);
    summaryLabel_->setText(summaryText);
    frameButton_->setEnabled(false);
    emit highlightRangesChanged({});
}

void PatternSearchDialog::discoverRepeatedPatterns() {
    if (dataSource_ == nullptr || !dataSource_->hasData()) {
        clearDiscoveredPatterns(QStringLiteral("Load a file before finding repeated patterns."));
        return;
    }

    const int sizeValue = repeatedPatternSizeSpinBox_ != nullptr ? qMax(1, repeatedPatternSizeSpinBox_->value()) : 1;
    const bool byteUnit = repeatedPatternUnitComboBox_ != nullptr
        && repeatedPatternUnitComboBox_->currentText() == QStringLiteral("Bytes");
    const qsizetype patternBitCount = byteUnit
        ? static_cast<qsizetype>(sizeValue) * 8
        : static_cast<qsizetype>(sizeValue);
    const qsizetype stepBits = byteUnit ? 8 : 1;
    const qsizetype totalBitCount = dataSource_->bitCount();

    if (patternBitCount <= 0) {
        clearDiscoveredPatterns(QStringLiteral("Pattern size must be at least 1 bit."));
        return;
    }

    if (patternBitCount > totalBitCount) {
        clearDiscoveredPatterns(QStringLiteral("Pattern size is larger than the loaded data."));
        return;
    }

    const qsizetype windowCount = 1 + ((totalBitCount - patternBitCount) / stepBits);
    if (windowCount > kMaximumRepeatedPatternWindows) {
        clearDiscoveredPatterns(
            QStringLiteral("Too many candidate patterns (%1). Increase the size or switch to bytes.")
                .arg(windowCount)
        );
        return;
    }

    struct PatternStats {
        int repeatCount = 0;
        qsizetype firstStartBit = -1;
    };

    QHash<QByteArray, PatternStats> repeatedPatternStats;
    repeatedPatternStats.reserve(static_cast<int>(qMin(windowCount, static_cast<qsizetype>(200000))));

    for (qsizetype startBit = 0; startBit + patternBitCount <= totalBitCount; startBit += stepBits) {
        const QByteArray patternBits = dataSource_->bitRange(startBit, patternBitCount);
        if (patternBits.size() != patternBitCount) {
            continue;
        }

        PatternStats& patternStats = repeatedPatternStats[patternBits];
        if (patternStats.repeatCount == 0) {
            patternStats.firstStartBit = startBit;
        }
        ++patternStats.repeatCount;
    }

    QVector<RepeatedPatternEntry> repeatedPatterns;
    repeatedPatterns.reserve(repeatedPatternStats.size());
    for (auto patternIt = repeatedPatternStats.cbegin(); patternIt != repeatedPatternStats.cend(); ++patternIt) {
        if (patternIt.value().repeatCount <= 1) {
            continue;
        }

        repeatedPatterns.append({
            patternIt.key(),
            patternIt.value().repeatCount,
            patternIt.value().firstStartBit,
        });
    }

    std::sort(
        repeatedPatterns.begin(),
        repeatedPatterns.end(),
        [](const RepeatedPatternEntry& left, const RepeatedPatternEntry& right) {
            if (left.repeatCount != right.repeatCount) {
                return left.repeatCount > right.repeatCount;
            }
            if (left.firstStartBit != right.firstStartBit) {
                return left.firstStartBit < right.firstStartBit;
            }
            return left.bits < right.bits;
        }
    );

    repeatedPatternsTableWidget_->clearContents();
    repeatedPatternsTableWidget_->setRowCount(repeatedPatterns.size());

    for (int patternIndex = 0; patternIndex < repeatedPatterns.size(); ++patternIndex) {
        const RepeatedPatternEntry& repeatedPattern = repeatedPatterns.at(patternIndex);

        auto makeItem = [](const QString& text) {
            QTableWidgetItem* item = new QTableWidgetItem(text);
            item->setTextAlignment(Qt::AlignCenter);
            return item;
        };

        repeatedPatternsTableWidget_->setItem(
            patternIndex,
            kRepeatedPatternRankColumn,
            makeItem(QString::number(patternIndex + 1))
        );

        QTableWidgetItem* patternItem = new QTableWidgetItem(patternDisplayText(repeatedPattern.bits));
        patternItem->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        patternItem->setData(Qt::UserRole, searchTextForBits(repeatedPattern.bits));
        repeatedPatternsTableWidget_->setItem(patternIndex, kRepeatedPatternValueColumn, patternItem);

        repeatedPatternsTableWidget_->setItem(
            patternIndex,
            kRepeatedPatternRepeatCountColumn,
            makeItem(QString::number(repeatedPattern.repeatCount))
        );
        repeatedPatternsTableWidget_->setItem(
            patternIndex,
            kRepeatedPatternFirstStartColumn,
            makeItem(matchStartLabel(repeatedPattern.firstStartBit))
        );
    }

    if (repeatedPatterns.isEmpty()) {
        repeatedPatternSummaryLabel_->setText(
            QStringLiteral("No repeated patterns found | %1-bit size | %2 %3 scanned")
                .arg(patternBitCount)
                .arg(windowCount)
                .arg(byteUnit ? QStringLiteral("byte-aligned windows") : QStringLiteral("bit-aligned windows"))
        );
        return;
    }

    repeatedPatternSummaryLabel_->setText(
        QStringLiteral("%1 repeated pattern%2 | %3-bit size | %4 %5 scanned")
            .arg(repeatedPatterns.size())
            .arg(repeatedPatterns.size() == 1 ? QString() : QStringLiteral("s"))
            .arg(patternBitCount)
            .arg(windowCount)
            .arg(byteUnit ? QStringLiteral("byte-aligned windows") : QStringLiteral("bit-aligned windows"))
    );
    repeatedPatternsTableWidget_->selectRow(0);
}

void PatternSearchDialog::handleDiscoveredPatternChanged() {
    if (repeatedPatternsTableWidget_ == nullptr) {
        return;
    }

    const int currentRow = repeatedPatternsTableWidget_->currentRow();
    if (currentRow < 0 || currentRow >= repeatedPatternsTableWidget_->rowCount()) {
        return;
    }

    QTableWidgetItem* patternItem = repeatedPatternsTableWidget_->item(currentRow, kRepeatedPatternValueColumn);
    if (patternItem == nullptr || patternLineEdit_ == nullptr) {
        return;
    }

    const QString patternText = patternItem->data(Qt::UserRole).toString().trimmed();
    if (patternText.isEmpty()) {
        return;
    }

    patternLineEdit_->setText(patternText);
    runSearch();
}

void PatternSearchDialog::clearDiscoveredPatterns(const QString& summaryText) {
    if (repeatedPatternsTableWidget_ != nullptr) {
        repeatedPatternsTableWidget_->clearContents();
        repeatedPatternsTableWidget_->setRowCount(0);
    }
    if (repeatedPatternSummaryLabel_ != nullptr) {
        repeatedPatternSummaryLabel_->setText(summaryText);
    }
}

bool PatternSearchDialog::hasCurrentMatch() const {
    const int currentRow = resultsTableWidget_ != nullptr ? resultsTableWidget_->currentRow() : -1;
    return currentRow >= 0 && currentRow < matchStartBits_.size() && patternBitCount_ > 0;
}

qsizetype PatternSearchDialog::currentMatchStartBit() const {
    const int currentRow = resultsTableWidget_ != nullptr ? resultsTableWidget_->currentRow() : -1;
    if (currentRow < 0 || currentRow >= matchStartBits_.size()) {
        return -1;
    }
    return matchStartBits_.at(currentRow);
}

}  // namespace bitabyte::ui
