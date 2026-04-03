#include "ui/framing_controller.h"

#include "features/frame_sync/frame_sync_search.h"
#include "models/byte_table_model.h"
#include "ui/bitstream_sync_discovery_dialog.h"
#include "ui/byte_table_view.h"
#include "ui/frame_browser_controller.h"
#include "ui/inspection_controller.h"
#include "ui/main_window_internal.h"

#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QStatusBar>

namespace bitabyte::ui {

FramingController::FramingController(
    data::ByteDataSource& dataSource,
    features::framing::FrameLayout& frameLayout,
    QVector<features::columns::ByteColumnDefinition>& columnDefinitions,
    models::ByteTableModel& byteTableModel,
    ByteTableView& byteTableView,
    FrameBrowserController& frameBrowserController,
    InspectionController& inspectionController,
    QLineEdit& syncPatternLineEdit,
    QPushButton& frameChronologicalOrderButton,
    QPushButton& frameLengthOrderButton,
    QWidget& dialogParent,
    QStatusBar& statusBar,
    Callbacks callbacks
)
    : dataSource_(dataSource),
      frameLayout_(frameLayout),
      columnDefinitions_(columnDefinitions),
      byteTableModel_(byteTableModel),
      byteTableView_(byteTableView),
      frameBrowserController_(frameBrowserController),
      inspectionController_(inspectionController),
      syncPatternLineEdit_(syncPatternLineEdit),
      frameChronologicalOrderButton_(frameChronologicalOrderButton),
      frameLengthOrderButton_(frameLengthOrderButton),
      dialogParent_(dialogParent),
      statusBar_(statusBar),
      callbacks_(std::move(callbacks)) {}

void FramingController::resetState() {
    frameChronologicalDescending_ = false;
    frameLengthDescending_ = false;
    syncPatternLineEdit_.clear();
    updateFrameRowOrderButtons();
}

void FramingController::syncControlsFromState() {
    updateFrameRowOrderButtons();
}

void FramingController::frameSelection() {
    if (!dataSource_.hasData()) {
        QMessageBox::information(
            &dialogParent_,
            QStringLiteral("Frame Selection"),
            QStringLiteral("Load a file first.")
        );
        return;
    }

    const QSet<int> selectedColumns = callbacks_.selectedVisibleColumns != nullptr
        ? callbacks_.selectedVisibleColumns()
        : QSet<int>{};
    QString patternText;
    QString errorMessage;
    if (callbacks_.extractSelectionPattern == nullptr
        || !callbacks_.extractSelectionPattern(&patternText, &errorMessage)) {
        QMessageBox::warning(
            &dialogParent_,
            QStringLiteral("Frame Selection"),
            errorMessage.isEmpty() ? QStringLiteral("Select one or more byte cells first.") : errorMessage
        );
        return;
    }

    syncPatternLineEdit_.setText(patternText);
    if (!applySyncFramingPattern(patternText, &errorMessage)) {
        QMessageBox::warning(&dialogParent_, QStringLiteral("Frame Selection"), errorMessage);
        return;
    }

    upsertSyncDefinitionForSelection(selectedColumns);
}

void FramingController::applySyncFraming() {
    if (!dataSource_.hasData()) {
        QMessageBox::information(
            &dialogParent_,
            QStringLiteral("Apply Framing"),
            QStringLiteral("Load a file first.")
        );
        return;
    }

    QString errorMessage;
    if (!applySyncFramingPattern(syncPatternLineEdit_.text(), &errorMessage)) {
        QMessageBox::warning(&dialogParent_, QStringLiteral("Apply Framing"), errorMessage);
    }
}

void FramingController::openBitstreamSyncDiscovery() {
    if (!dataSource_.hasData()) {
        QMessageBox::information(
            &dialogParent_,
            QStringLiteral("Find Frames"),
            QStringLiteral("Load a file before running Find Frames.")
        );
        return;
    }

    BitstreamSyncDiscoveryDialog dialog(&dataSource_, &dialogParent_);
    if (dialog.exec() != QDialog::DialogCode::Accepted) {
        return;
    }

    const std::optional<features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate> selectedCandidate =
        dialog.selectedCandidate();
    if (!selectedCandidate.has_value()) {
        return;
    }

    syncPatternLineEdit_.setText(selectedCandidate->displayPattern);
    frameBrowserController_.setFrameSpans(selectedCandidate->frameSpans);
    frameLayout_.setFrames(frameBrowserController_.frameSpans());
    upsertSyncDefinition(0, selectedCandidate->refinedPattern.bitWidth, selectedCandidate->displayFormat);
    syncControlsFromState();
    byteTableModel_.reload();
    if (callbacks_.resizeTableColumns != nullptr) {
        callbacks_.resizeTableColumns();
    }
    byteTableView_.clearSelection();
    byteTableView_.setCurrentIndex(QModelIndex());
    if (callbacks_.updateLoadedFileState != nullptr) {
        callbacks_.updateLoadedFileState();
    }
    inspectionController_.updateSelectionStatus();
    inspectionController_.refreshLiveBitViewer();
    inspectionController_.scheduleFrameFieldHintsRefresh();
    statusBar_.showMessage(
        QStringLiteral("Applied frame result: %1 (%2 frames)")
            .arg(selectedCandidate->displayPattern)
            .arg(selectedCandidate->frameSpans.size()),
        5000
    );
}

void FramingController::clearFraming() {
    if (!frameLayout_.isFramed()) {
        return;
    }

    frameBrowserController_.clearState();
    frameLayout_.clearFrame();
    syncControlsFromState();
    byteTableModel_.reload();
    if (callbacks_.resizeTableColumns != nullptr) {
        callbacks_.resizeTableColumns();
    }
    byteTableView_.clearSelection();
    byteTableView_.setCurrentIndex(QModelIndex());
    if (callbacks_.updateLoadedFileState != nullptr) {
        callbacks_.updateLoadedFileState();
    }
    inspectionController_.updateSelectionStatus();
    inspectionController_.refreshLiveBitViewer();
    inspectionController_.scheduleFrameFieldHintsRefresh();
    statusBar_.showMessage(QStringLiteral("Cleared framing"), 3000);
}

void FramingController::cycleFrameChronologicalOrder() {
    const bool descending = frameLayout_.rowOrderMode() == features::framing::FrameLayout::RowOrderMode::Chronological
        ? !frameLayout_.rowOrderDescending()
        : frameChronologicalDescending_;
    frameChronologicalDescending_ = descending;
    applyFrameRowOrder(features::framing::FrameLayout::RowOrderMode::Chronological, descending);
}

void FramingController::cycleFrameLengthOrder() {
    const bool descending = frameLayout_.rowOrderMode() == features::framing::FrameLayout::RowOrderMode::Length
        ? !frameLayout_.rowOrderDescending()
        : frameLengthDescending_;
    frameLengthDescending_ = descending;
    applyFrameRowOrder(features::framing::FrameLayout::RowOrderMode::Length, descending);
}

void FramingController::applyFrameRowOrder(
    features::framing::FrameLayout::RowOrderMode rowOrderMode,
    bool descending
) {
    frameLayout_.setRowOrder(rowOrderMode, descending);
    updateFrameRowOrderButtons();

    byteTableModel_.reload();
    if (callbacks_.resizeTableColumns != nullptr) {
        callbacks_.resizeTableColumns();
    }
    byteTableView_.clearSelection();
    byteTableView_.setCurrentIndex(QModelIndex());
    inspectionController_.updateSelectionStatus();
    if (callbacks_.refreshColumnDefinitionsPanel != nullptr) {
        callbacks_.refreshColumnDefinitionsPanel();
    }
    inspectionController_.refreshLiveBitViewer();
    inspectionController_.refreshFieldInspector();
    inspectionController_.scheduleFrameFieldHintsRefresh();
    statusBar_.showMessage(
        QStringLiteral("Frame order: %1 %2")
            .arg(rowOrderMode == features::framing::FrameLayout::RowOrderMode::Length
                     ? QStringLiteral("frame size")
                     : QStringLiteral("chronological"))
            .arg(descending ? QStringLiteral("descending") : QStringLiteral("ascending")),
        3000
    );
}

void FramingController::updateFrameRowOrderButtons() {
    frameChronologicalOrderButton_.setText(
        detail::rowOrderButtonLabel(QStringLiteral("Chronological"), frameChronologicalDescending_)
    );
    frameChronologicalOrderButton_.setDefault(
        frameLayout_.rowOrderMode() == features::framing::FrameLayout::RowOrderMode::Chronological
    );

    frameLengthOrderButton_.setText(
        detail::rowOrderButtonLabel(QStringLiteral("Frame Size"), frameLengthDescending_)
    );
    frameLengthOrderButton_.setDefault(
        frameLayout_.rowOrderMode() == features::framing::FrameLayout::RowOrderMode::Length
    );
}

bool FramingController::applySyncFramingPattern(const QString& patternText, QString* errorMessage) {
    QString localErrorMessage;
    const std::optional<features::frame_sync::FrameSyncSearchResult> searchResult =
        features::frame_sync::FrameSyncSearch::findBitAccurateMatches(
            dataSource_,
            patternText,
            &localErrorMessage
        );
    if (!searchResult.has_value()) {
        if (errorMessage != nullptr) {
            *errorMessage = localErrorMessage;
        }
        return false;
    }

    frameBrowserController_.setFrameSpans(searchResult->frameSpans);
    frameLayout_.setFrames(frameBrowserController_.frameSpans());
    syncControlsFromState();
    byteTableModel_.reload();
    if (callbacks_.resizeTableColumns != nullptr) {
        callbacks_.resizeTableColumns();
    }
    byteTableView_.clearSelection();
    byteTableView_.setCurrentIndex(QModelIndex());
    if (callbacks_.updateLoadedFileState != nullptr) {
        callbacks_.updateLoadedFileState();
    }
    inspectionController_.updateSelectionStatus();
    inspectionController_.refreshLiveBitViewer();
    inspectionController_.scheduleFrameFieldHintsRefresh();
    statusBar_.showMessage(
        QStringLiteral("Framed into %1 rows from %2 sync match%3")
            .arg(searchResult->frameSpans.size())
            .arg(searchResult->matchCount)
            .arg(searchResult->matchCount == 1 ? QString() : QStringLiteral("es")),
        4000
    );
    return true;
}

void FramingController::upsertSyncDefinitionForSelection(const QSet<int>& selectedVisibleColumns) {
    if (selectedVisibleColumns.isEmpty()) {
        return;
    }

    features::columns::ByteColumnDefinition syncDefinition;
    QString errorMessage;
    if (callbacks_.buildDefinitionFromSelection == nullptr
        || !callbacks_.buildDefinitionFromSelection(selectedVisibleColumns, &syncDefinition, &errorMessage)) {
        return;
    }

    upsertSyncDefinition(
        syncDefinition.startAbsoluteBit(),
        syncDefinition.totalBits,
        syncDefinition.totalBits % 4 == 0 ? QStringLiteral("hex") : QStringLiteral("binary")
    );
}

void FramingController::upsertSyncDefinition(int startBit, int totalBits, const QString& displayFormat) {
    features::columns::ByteColumnDefinition syncDefinition;
    syncDefinition.label = QStringLiteral("Sync");
    syncDefinition.unit = QStringLiteral("bit");
    syncDefinition.startBit = qMax(0, startBit);
    syncDefinition.totalBits = qMax(1, totalBits);
    syncDefinition.startByte = syncDefinition.startBit / 8;
    syncDefinition.endByte = (syncDefinition.startBit + syncDefinition.totalBits - 1) / 8;
    syncDefinition.displayFormat =
        displayFormat.trimmed().isEmpty() ? QStringLiteral("hex") : displayFormat.trimmed().toLower();
    syncDefinition.colorName = QStringLiteral("Sky");

    int existingSyncIndex = -1;
    for (int definitionIndex = 0; definitionIndex < columnDefinitions_.size(); ++definitionIndex) {
        if (columnDefinitions_[definitionIndex].label == QStringLiteral("Sync")) {
            existingSyncIndex = definitionIndex;
            break;
        }
    }

    if (existingSyncIndex >= 0) {
        columnDefinitions_[existingSyncIndex] = syncDefinition;
    } else {
        columnDefinitions_.prepend(syncDefinition);
    }

    byteTableModel_.reload();
    if (callbacks_.refreshColumnDefinitionsPanel != nullptr) {
        callbacks_.refreshColumnDefinitionsPanel();
    }
    if (callbacks_.resizeTableColumns != nullptr) {
        callbacks_.resizeTableColumns();
    }
    inspectionController_.updateSelectionStatus();
    inspectionController_.refreshLiveBitViewer();
    inspectionController_.scheduleFrameFieldHintsRefresh();
}

}  // namespace bitabyte::ui
