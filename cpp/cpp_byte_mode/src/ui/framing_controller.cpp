#include "ui/framing_controller.h"

#include "features/frame_sync/frame_sync_search.h"
#include "models/byte_table_model.h"
#include "ui/bitstream_sync_discovery_dialog.h"
#include "ui/byte_table_view.h"
#include "ui/frame_browser_controller.h"
#include "ui/inspection_controller.h"
#include "ui/main_window_internal.h"

#include <QComboBox>
#include <QLineEdit>
#include <QMessageBox>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStatusBar>

namespace bitabyte::ui {
namespace {

enum FrameSortOptionIndex {
    kChronologicalAscending = 0,
    kChronologicalDescending = 1,
    kFrameSizeAscending = 2,
    kFrameSizeDescending = 3,
};

int frameSortOptionIndex(
    features::framing::FrameLayout::RowOrderMode rowOrderMode,
    bool descending
) {
    if (rowOrderMode == features::framing::FrameLayout::RowOrderMode::Length) {
        return descending ? kFrameSizeDescending : kFrameSizeAscending;
    }
    return descending ? kChronologicalDescending : kChronologicalAscending;
}

bool overlapsDefinitionRange(
    int startBit,
    int endBit,
    const QVector<features::columns::ByteColumnDefinition>& columnDefinitions
) {
    for (const features::columns::ByteColumnDefinition& definition : columnDefinitions) {
        if (!(endBit < definition.startAbsoluteBit() || startBit > definition.endAbsoluteBit())) {
            return true;
        }
    }
    return false;
}

QString fallbackDetectedFieldLabel(int startBit, int endBit) {
    if (startBit < 0 || endBit < startBit) {
        return QStringLiteral("Field");
    }

    if ((startBit % 8) == 0 && (endBit % 8) == 7) {
        const int startByte = startBit / 8;
        const int endByte = endBit / 8;
        return startByte == endByte
            ? QStringLiteral("Byte %1").arg(startByte)
            : QStringLiteral("Bytes %1-%2").arg(startByte).arg(endByte);
    }

    return startBit == endBit
        ? QStringLiteral("Bit %1").arg(startBit)
        : QStringLiteral("Bits %1-%2").arg(startBit).arg(endBit);
}

QString nextDetectedCounterLabel(
    const QVector<features::columns::ByteColumnDefinition>& columnDefinitions
) {
    int nextCounterIndex = 1;
    for (const features::columns::ByteColumnDefinition& definition : columnDefinitions) {
        const QString labelText = definition.label.trimmed();
        if (labelText == QStringLiteral("Counter")) {
            nextCounterIndex = qMax(nextCounterIndex, 2);
            continue;
        }
        if (!labelText.startsWith(QStringLiteral("Counter "))) {
            continue;
        }

        bool parsed = false;
        const int counterIndex = labelText.mid(QStringLiteral("Counter ").size()).toInt(&parsed);
        if (parsed) {
            nextCounterIndex = qMax(nextCounterIndex, counterIndex + 1);
        }
    }

    return QStringLiteral("Counter %1").arg(nextCounterIndex);
}

}  // namespace

FramingController::FramingController(
    data::ByteDataSource& dataSource,
    features::framing::FrameLayout& frameLayout,
    QVector<features::columns::ByteColumnDefinition>& columnDefinitions,
    models::ByteTableModel& byteTableModel,
    ByteTableView& byteTableView,
    FrameBrowserController& frameBrowserController,
    InspectionController& inspectionController,
    QLineEdit& syncPatternLineEdit,
    QSpinBox& fixedFrameWidthSpinBox,
    QSpinBox& fixedFrameBitOffsetSpinBox,
    QComboBox& frameSortComboBox,
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
      fixedFrameWidthSpinBox_(fixedFrameWidthSpinBox),
      fixedFrameBitOffsetSpinBox_(fixedFrameBitOffsetSpinBox),
      frameSortComboBox_(frameSortComboBox),
      dialogParent_(dialogParent),
      statusBar_(statusBar),
      callbacks_(std::move(callbacks)) {}

void FramingController::resetState() {
    framingSource_ = FramingSource::None;
    syncPatternLineEdit_.clear();
    {
        const QSignalBlocker widthBlocker(fixedFrameWidthSpinBox_);
        fixedFrameWidthSpinBox_.setValue(qMax(1, dataSource_.bytesPerRow()));
    }
    {
        const QSignalBlocker offsetBlocker(fixedFrameBitOffsetSpinBox_);
        fixedFrameBitOffsetSpinBox_.setValue(0);
    }
    updateFrameSortComboBox();
}

void FramingController::syncControlsFromState() {
    updateFrameSortComboBox();
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

void FramingController::applyFixedFraming() {
    if (!dataSource_.hasData()) {
        QMessageBox::information(
            &dialogParent_,
            QStringLiteral("Set Frame"),
            QStringLiteral("Load a file first.")
        );
        return;
    }

    QString errorMessage;
    if (!applyFixedFramingParameters(
            fixedFrameWidthSpinBox_.value(),
            fixedFrameBitOffsetSpinBox_.value(),
            &errorMessage
        )) {
        QMessageBox::warning(&dialogParent_, QStringLiteral("Set Frame"), errorMessage);
    }
}

bool FramingController::frameByPattern(const QString& patternText, QString* errorMessage) {
    syncPatternLineEdit_.setText(patternText);
    return applySyncFramingPattern(patternText, errorMessage);
}

bool FramingController::fixedFramingControlsEditable() const {
    return dataSource_.hasData()
        && (!frameLayout_.isFramed() || framingSource_ == FramingSource::FixedWidth);
}

void FramingController::openBitstreamSyncDiscovery() {
    if (!dataSource_.hasData()) {
        QMessageBox::information(
            &dialogParent_,
            QStringLiteral("Find Framing"),
            QStringLiteral("Load a file before running Find Framing.")
        );
        return;
    }

    BitstreamSyncDiscoveryDialog dialog(&dataSource_, &columnDefinitions_, &dialogParent_);
    if (dialog.exec() != QDialog::DialogCode::Accepted) {
        return;
    }

    const std::optional<features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate> selectedCandidate =
        dialog.selectedCandidate();
    if (!selectedCandidate.has_value()) {
        return;
    }

    const QVector<features::classification::FrameFieldHint> selectedDetectedHints = dialog.selectedColumnHints();
    syncPatternLineEdit_.setText(selectedCandidate->displayPattern);
    framingSource_ = FramingSource::Other;
    frameBrowserController_.setFrameSpans(selectedCandidate->frameSpans);
    frameLayout_.setPadFramedBitDisplayToByteBoundary(true);
    frameLayout_.setFrames(frameBrowserController_.frameSpans());
    upsertSyncDefinitionRecord(0, selectedCandidate->refinedPattern.bitWidth, selectedCandidate->displayFormat);
    const int addedDefinitionCount = appendDetectedFieldDefinitions(selectedDetectedHints);
    syncControlsFromState();
    byteTableModel_.reload();
    if (callbacks_.refreshColumnDefinitionsPanel != nullptr) {
        callbacks_.refreshColumnDefinitionsPanel();
    }
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
        QStringLiteral("Applied frame result: %1 (%2 frames%3)")
            .arg(selectedCandidate->displayPattern)
            .arg(selectedCandidate->frameSpans.size())
            .arg(
                addedDefinitionCount > 0
                    ? QStringLiteral(", %1 column%2 added")
                          .arg(addedDefinitionCount)
                          .arg(addedDefinitionCount == 1 ? QString() : QStringLiteral("s"))
                    : QString()
            ),
        5000
    );
}

void FramingController::clearFraming() {
    if (!frameLayout_.isFramed()) {
        return;
    }

    framingSource_ = FramingSource::None;
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

void FramingController::setFrameSortOption(int sortOptionIndex) {
    switch (sortOptionIndex) {
    case kChronologicalDescending:
        applyFrameRowOrder(features::framing::FrameLayout::RowOrderMode::Chronological, true);
        break;
    case kFrameSizeAscending:
        applyFrameRowOrder(features::framing::FrameLayout::RowOrderMode::Length, false);
        break;
    case kFrameSizeDescending:
        applyFrameRowOrder(features::framing::FrameLayout::RowOrderMode::Length, true);
        break;
    case kChronologicalAscending:
    default:
        applyFrameRowOrder(features::framing::FrameLayout::RowOrderMode::Chronological, false);
        break;
    }
}

void FramingController::applyFrameRowOrder(
    features::framing::FrameLayout::RowOrderMode rowOrderMode,
    bool descending
) {
    frameLayout_.setRowOrder(rowOrderMode, descending);
    updateFrameSortComboBox();

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

void FramingController::updateFrameSortComboBox() {
    const int currentSortOption = frameSortOptionIndex(
        frameLayout_.rowOrderMode(),
        frameLayout_.rowOrderDescending()
    );
    const QSignalBlocker blocker(frameSortComboBox_);
    frameSortComboBox_.setCurrentIndex(currentSortOption);
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

    framingSource_ = FramingSource::Other;
    frameBrowserController_.setFrameSpans(searchResult->frameSpans);
    frameLayout_.setPadFramedBitDisplayToByteBoundary(false);
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

bool FramingController::applyFixedFramingParameters(int frameWidthValue, int bitOffset, QString* errorMessage) {
    const bool bitModeEnabled = callbacks_.isBitModeEnabled != nullptr && callbacks_.isBitModeEnabled();
    QString localErrorMessage;
    if (frameWidthValue <= 0) {
        localErrorMessage = bitModeEnabled
            ? QStringLiteral("Frame width must be at least 1 bit.")
            : QStringLiteral("Frame width must be at least 1 byte.");
    } else if (bitOffset < 0 || bitOffset > 7) {
        localErrorMessage = QStringLiteral("Bit offset must be between 0 and 7.");
    } else if (!dataSource_.hasData()) {
        localErrorMessage = QStringLiteral("Load a file first.");
    }

    if (!localErrorMessage.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = localErrorMessage;
        }
        return false;
    }

    const qsizetype frameLengthBits = bitModeEnabled
        ? static_cast<qsizetype>(frameWidthValue)
        : static_cast<qsizetype>(frameWidthValue) * 8;
    const qsizetype firstFrameStartBit = bitOffset;
    const qsizetype totalBitCount = dataSource_.bitCount();

    QVector<features::framing::FrameSpan> frameSpans;
    if (firstFrameStartBit < totalBitCount) {
        const qsizetype addressableBits = totalBitCount - firstFrameStartBit;
        const qsizetype estimatedFrameCount = (addressableBits + frameLengthBits - 1) / frameLengthBits;
        frameSpans.reserve(static_cast<int>(estimatedFrameCount));

        for (qsizetype frameStartBit = firstFrameStartBit;
             frameStartBit < totalBitCount;
             frameStartBit += frameLengthBits) {
            frameSpans.append({
                frameStartBit,
                qMin(frameLengthBits, totalBitCount - frameStartBit),
            });
        }
    }

    if (frameSpans.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("No frames fit within the loaded data for that width and offset.");
        }
        return false;
    }

    framingSource_ = FramingSource::FixedWidth;
    frameBrowserController_.setFrameSpans(frameSpans);
    frameLayout_.setPadFramedBitDisplayToByteBoundary(false);
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
        bitModeEnabled
            ? (
                bitOffset == 0
                    ? QStringLiteral("Framed into %1 fixed rows of %2 bits")
                          .arg(frameSpans.size())
                          .arg(frameWidthValue)
                    : QStringLiteral("Framed into %1 fixed rows of %2 bits at bit offset %3")
                          .arg(frameSpans.size())
                          .arg(frameWidthValue)
                          .arg(bitOffset)
            )
            : (
                bitOffset == 0
                    ? QStringLiteral("Framed into %1 fixed rows of %2 bytes")
                          .arg(frameSpans.size())
                          .arg(frameWidthValue)
                    : QStringLiteral("Framed into %1 fixed rows of %2 bytes at bit offset %3")
                          .arg(frameSpans.size())
                          .arg(frameWidthValue)
                          .arg(bitOffset)
            ),
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

void FramingController::upsertSyncDefinitionRecord(int startBit, int totalBits, const QString& displayFormat) {
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
}

void FramingController::upsertSyncDefinition(int startBit, int totalBits, const QString& displayFormat) {
    upsertSyncDefinitionRecord(startBit, totalBits, displayFormat);

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

std::optional<features::columns::ByteColumnDefinition> FramingController::definitionForDetectedHint(
    const features::classification::FrameFieldHint& detectedHint
) const {
    if (detectedHint.absoluteStartBit < 0 || detectedHint.absoluteEndBit < detectedHint.absoluteStartBit) {
        return std::nullopt;
    }
    if (overlapsDefinitionRange(
            detectedHint.absoluteStartBit,
            detectedHint.absoluteEndBit,
            columnDefinitions_
        )) {
        return std::nullopt;
    }

    features::columns::ByteColumnDefinition definition;
    definition.unit = QStringLiteral("bit");
    definition.startBit = detectedHint.absoluteStartBit;
    definition.totalBits = detectedHint.absoluteEndBit - detectedHint.absoluteStartBit + 1;
    definition.startByte = definition.startBit / 8;
    definition.endByte = definition.endAbsoluteBit() / 8;
    const bool isConstantHint = !detectedHint.valueText.trimmed().isEmpty();
    definition.label = isConstantHint
        ? (detectedHint.label.trimmed().isEmpty()
               ? fallbackDetectedFieldLabel(detectedHint.absoluteStartBit, detectedHint.absoluteEndBit)
               : detectedHint.label.trimmed())
        : nextDetectedCounterLabel(columnDefinitions_);
    definition.displayFormat = !isConstantHint
        ? QStringLiteral("decimal")
        : (definition.totalBits % 4 == 0 ? QStringLiteral("hex") : QStringLiteral("binary"));
    definition.colorName = detail::nextAutoColumnColorName(columnDefinitions_);
    return definition;
}

int FramingController::appendDetectedFieldDefinitions(
    const QVector<features::classification::FrameFieldHint>& detectedHints
) {
    int addedDefinitionCount = 0;
    for (const features::classification::FrameFieldHint& detectedHint : detectedHints) {
        const std::optional<features::columns::ByteColumnDefinition> detectedDefinition =
            definitionForDetectedHint(detectedHint);
        if (!detectedDefinition.has_value()) {
            continue;
        }

        columnDefinitions_.append(detectedDefinition.value());
        ++addedDefinitionCount;
    }
    return addedDefinitionCount;
}

}  // namespace bitabyte::ui
