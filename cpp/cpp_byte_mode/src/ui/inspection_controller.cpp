#include "ui/inspection_controller.h"

#include "features/inspector/field_inspector_analysis.h"
#include "models/byte_table_model.h"
#include "ui/byte_table_view.h"
#include "ui/field_inspector_panel.h"
#include "ui/live_bit_viewer_widget.h"
#include "ui/main_window_internal.h"

#include <QtConcurrent>

#include <QAbstractButton>
#include <QButtonGroup>
#include <QFutureWatcher>
#include <QItemSelection>
#include <QItemSelectionRange>
#include <QLabel>
#include <QSpinBox>
#include <QTimer>

#include <algorithm>

namespace bitabyte::ui {

InspectionController::InspectionController(
    data::ByteDataSource& dataSource,
    features::framing::FrameLayout& frameLayout,
    QVector<features::columns::ByteColumnDefinition>& columnDefinitions,
    models::ByteTableModel& byteTableModel,
    ByteTableView& byteTableView,
    LiveBitViewerWidget& liveBitViewerWidget,
    FieldInspectorPanel& fieldInspectorPanel,
    QLabel& selectionInfoLabel,
    QButtonGroup& liveBitViewerModeGroup,
    QSpinBox& liveBitViewerSizeSpinBox,
    SelectionCallbacks selectionCallbacks,
    QObject* parent
)
    : QObject(parent),
      dataSource_(dataSource),
      frameLayout_(frameLayout),
      columnDefinitions_(columnDefinitions),
      byteTableModel_(byteTableModel),
      byteTableView_(byteTableView),
      liveBitViewerWidget_(liveBitViewerWidget),
      fieldInspectorPanel_(fieldInspectorPanel),
      selectionInfoLabel_(selectionInfoLabel),
      liveBitViewerModeGroup_(liveBitViewerModeGroup),
      liveBitViewerSizeSpinBox_(liveBitViewerSizeSpinBox),
      selectionCallbacks_(std::move(selectionCallbacks)),
      liveBitViewerRefreshTimer_(new QTimer(this)),
      fieldInspectorRefreshTimer_(new QTimer(this)),
      fieldInspectorWatcher_(new QFutureWatcher<features::inspector::FieldInspectorAnalysis>(this)) {
    liveBitViewerRefreshTimer_->setSingleShot(true);
    liveBitViewerRefreshTimer_->setInterval(0);
    connect(liveBitViewerRefreshTimer_, &QTimer::timeout, this, [this]() { refreshLiveBitViewer(); });

    fieldInspectorRefreshTimer_->setSingleShot(true);
    fieldInspectorRefreshTimer_->setInterval(0);
    connect(fieldInspectorRefreshTimer_, &QTimer::timeout, this, [this]() { refreshFieldInspector(); });

    connect(
        fieldInspectorWatcher_,
        &QFutureWatcher<features::inspector::FieldInspectorAnalysis>::finished,
        this,
        [this]() {
            if (activeFieldInspectorRequestId_ == fieldInspectorRequestId_) {
                fieldInspectorPanel_.setAnalysis(fieldInspectorWatcher_->result());
                return;
            }

            refreshFieldInspector();
        }
    );
}

void InspectionController::scheduleLiveBitViewerRefresh() {
    if (!liveBitViewerRefreshTimer_->isActive()) {
        liveBitViewerRefreshTimer_->start();
    }
}

void InspectionController::refreshLiveBitViewer() {
    liveBitViewerWidget_.setDisplayMode(currentLiveBitViewerMode());
    liveBitViewerWidget_.setCellSize(liveBitViewerSizeSpinBox_.value());

    const QSet<int> selectedColumnSet = selectionCallbacks_.selectedVisibleColumns != nullptr
        ? selectionCallbacks_.selectedVisibleColumns()
        : QSet<int>{};
    if (!dataSource_.hasData() || !frameLayout_.isFramed() || selectedColumnSet.isEmpty()) {
        liveBitViewerWidget_.setPreviewSource(nullptr, nullptr, {}, 0, 0);
        return;
    }

    QList<int> selectedColumns = selectedColumnSet.values();
    std::sort(selectedColumns.begin(), selectedColumns.end());

    constexpr int kMaxLivePreviewRows = 128;
    const int frameRowCount = static_cast<int>(frameLayout_.rowCount(dataSource_));
    const QModelIndex anchorIndex = selectionCallbacks_.topSelectedDataIndex != nullptr
        ? selectionCallbacks_.topSelectedDataIndex()
        : QModelIndex{};
    const int anchorRow = anchorIndex.isValid() ? anchorIndex.row() : 0;
    const int previewRowCount = qMin(frameRowCount, kMaxLivePreviewRows);
    const int firstPreviewRow = frameRowCount <= previewRowCount
        ? 0
        : qBound(0, anchorRow - (previewRowCount / 2), frameRowCount - previewRowCount);

    QVector<features::columns::VisibleByteColumn> previewColumns;
    previewColumns.reserve(selectedColumns.size());
    for (int visibleColumnIndex : selectedColumns) {
        previewColumns.append(byteTableModel_.visibleByteColumn(visibleColumnIndex));
    }

    liveBitViewerWidget_.setPreviewSource(
        &dataSource_,
        &frameLayout_,
        previewColumns,
        firstPreviewRow,
        previewRowCount
    );
}

void InspectionController::refreshFieldInspector() {
    if (!dataSource_.hasData()) {
        ++fieldInspectorRequestId_;
        fieldInspectorPanel_.clearAnalysis();
        return;
    }

    QSet<int> inspectedColumns = selectionCallbacks_.selectedVisibleColumns != nullptr
        ? selectionCallbacks_.selectedVisibleColumns()
        : QSet<int>{};
    if (inspectedColumns.size() == 1 && selectionCallbacks_.editableVisibleColumnsForSeed != nullptr) {
        inspectedColumns = selectionCallbacks_.editableVisibleColumnsForSeed(*inspectedColumns.begin());
    }

    QModelIndex anchorIndex = byteTableView_.currentIndex();
    int anchorVisibleColumnIndex = byteTableModel_.visibleColumnIndexForModelIndex(anchorIndex);
    if (inspectedColumns.isEmpty() && anchorVisibleColumnIndex >= 0
        && selectionCallbacks_.editableVisibleColumnsForSeed != nullptr) {
        inspectedColumns = selectionCallbacks_.editableVisibleColumnsForSeed(anchorVisibleColumnIndex);
    }

    if (inspectedColumns.isEmpty() || !byteTableModel_.visibleColumnsAreContiguous(inspectedColumns)) {
        ++fieldInspectorRequestId_;
        fieldInspectorPanel_.clearAnalysis();
        return;
    }

    QList<int> sortedVisibleColumns = inspectedColumns.values();
    std::sort(sortedVisibleColumns.begin(), sortedVisibleColumns.end());

    const features::columns::VisibleByteColumn firstVisibleColumn =
        byteTableModel_.visibleByteColumn(sortedVisibleColumns.first());
    const features::columns::VisibleByteColumn lastVisibleColumn =
        byteTableModel_.visibleByteColumn(sortedVisibleColumns.last());

    QString fieldLabel = detail::defaultFieldLabel(firstVisibleColumn, lastVisibleColumn);
    const int definitionIndex = selectionCallbacks_.definitionIndexForVisibleColumns != nullptr
        ? selectionCallbacks_.definitionIndexForVisibleColumns(inspectedColumns)
        : -1;
    if (definitionIndex >= 0 && definitionIndex < columnDefinitions_.size()) {
        const QString explicitLabel = columnDefinitions_.at(definitionIndex).label.trimmed();
        if (!explicitLabel.isEmpty()) {
            fieldLabel = explicitLabel;
        }
    } else {
        QString sharedSplitLabel;
        bool allVisibleColumnsShareSplit = true;
        for (int visibleColumnIndex : sortedVisibleColumns) {
            const features::columns::VisibleByteColumn visibleColumn =
                byteTableModel_.visibleByteColumn(visibleColumnIndex);
            const QString splitLabel = visibleColumn.splitLabel.trimmed();
            if (splitLabel.isEmpty()) {
                allVisibleColumnsShareSplit = false;
                break;
            }

            if (sharedSplitLabel.isEmpty()) {
                sharedSplitLabel = splitLabel;
                continue;
            }

            if (sharedSplitLabel != splitLabel) {
                allVisibleColumnsShareSplit = false;
                break;
            }
        }

        if (allVisibleColumnsShareSplit && !sharedSplitLabel.isEmpty()) {
            fieldLabel = sharedSplitLabel;
        }
    }

    if (anchorVisibleColumnIndex < 0 || byteTableModel_.isFrameLengthColumn(anchorIndex.column())) {
        anchorIndex = selectionCallbacks_.topSelectedDataIndex != nullptr
            ? selectionCallbacks_.topSelectedDataIndex()
            : QModelIndex{};
        anchorVisibleColumnIndex = byteTableModel_.visibleColumnIndexForModelIndex(anchorIndex);
    }

    const int anchorRow = anchorIndex.isValid() ? anchorIndex.row() : -1;
    features::inspector::FieldSelection fieldSelection;
    fieldSelection.label = fieldLabel;
    fieldSelection.startBit = firstVisibleColumn.absoluteStartBit;
    fieldSelection.endBit = lastVisibleColumn.absoluteEndBit;
    const int startByteIndex = fieldSelection.startBit / 8;
    const int endByteIndex = fieldSelection.endBit / 8;
    const QString byteRangeText = startByteIndex == endByteIndex
        ? QStringLiteral("byte %1").arg(startByteIndex)
        : QStringLiteral("bytes %1-%2").arg(startByteIndex).arg(endByteIndex);
    const QString positionText = QStringLiteral("bits %1-%2 | %3")
        .arg(fieldSelection.startBit)
        .arg(fieldSelection.endBit)
        .arg(byteRangeText);
    fieldInspectorPanel_.setPendingAnalysis(fieldLabel, positionText);
    ++fieldInspectorRequestId_;
    if (fieldInspectorWatcher_->isRunning()) {
        return;
    }

    startFieldInspectorAnalysis(fieldSelection, anchorRow, fieldInspectorRequestId_);
}

void InspectionController::updateSelectionStatus() {
    const QModelIndex currentIndex = byteTableView_.currentIndex();
    if (frameLayout_.isFramed() && byteTableModel_.isFrameLengthColumn(currentIndex.column())) {
        selectionInfoLabel_.setText(
            QStringLiteral("Selection: frame row %1 | length %2 bytes | %3 bits")
                .arg(currentIndex.row())
                .arg(frameLayout_.rowLengthBytes(dataSource_, currentIndex.row()))
                .arg(frameLayout_.rowLengthBits(dataSource_, currentIndex.row()))
        );
        scheduleFieldInspectorRefresh();
        return;
    }

    const qsizetype startBit = byteTableModel_.displayStartBitForIndex(currentIndex);
    if (!dataSource_.isBitOffsetValid(startBit)) {
        selectionInfoLabel_.setText(QStringLiteral("Selection: none"));
        scheduleFieldInspectorRefresh();
        return;
    }

    const QString rowLabel = frameLayout_.isFramed() ? QStringLiteral("frame row") : QStringLiteral("row");
    const int visibleColumnIndex = byteTableModel_.visibleColumnIndexForModelIndex(currentIndex);
    const features::columns::VisibleByteColumn visibleColumn = byteTableModel_.visibleByteColumn(visibleColumnIndex);
    const QString valueText = currentIndex.data(Qt::DisplayRole).toString();
    selectionInfoLabel_.setText(
        QStringLiteral("Selection: start bit %1 | %2 %3 col %4 | bits %5-%6 | value %7")
            .arg(startBit)
            .arg(rowLabel)
            .arg(currentIndex.row())
            .arg(visibleColumnIndex >= 0 ? visibleColumnIndex : currentIndex.column())
            .arg(visibleColumn.absoluteStartBit)
            .arg(visibleColumn.absoluteEndBit)
            .arg(valueText)
    );
    scheduleFieldInspectorRefresh();
}

void InspectionController::scheduleFieldInspectorRefresh() {
    if (!fieldInspectorRefreshTimer_->isActive()) {
        fieldInspectorRefreshTimer_->start();
    }
}

void InspectionController::startFieldInspectorAnalysis(
    const features::inspector::FieldSelection& fieldSelection,
    int currentRow,
    quint64 requestId
) {
    activeFieldInspectorRequestId_ = requestId;
    const data::ByteDataSource dataSourceSnapshot = dataSource_;
    const features::framing::FrameLayout frameLayoutSnapshot = frameLayout_;
    const features::inspector::FieldSelection fieldSelectionSnapshot = fieldSelection;
    fieldInspectorWatcher_->setFuture(QtConcurrent::run(
        [dataSourceSnapshot, frameLayoutSnapshot, fieldSelectionSnapshot, currentRow]() {
            return features::inspector::analyzeField(
                dataSourceSnapshot,
                frameLayoutSnapshot,
                fieldSelectionSnapshot,
                currentRow
            );
        }
    ));
}

QString InspectionController::currentLiveBitViewerMode() const {
    if (liveBitViewerModeGroup_.checkedButton() == nullptr) {
        return QStringLiteral("squares");
    }

    return liveBitViewerModeGroup_.checkedButton()->text().trimmed().toLower();
}

}  // namespace bitabyte::ui
