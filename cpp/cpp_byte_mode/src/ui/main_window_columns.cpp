#include "ui/main_window.h"

#include "models/byte_table_model.h"
#include "ui/byte_table_view.h"
#include "ui/column_definition_dialog.h"
#include "ui/column_definitions_panel.h"
#include "ui/frame_browser_controller.h"
#include "ui/framing_controller.h"
#include "ui/inspection_controller.h"
#include "ui/main_window_internal.h"

#include <QItemSelection>
#include <QItemSelectionModel>
#include <QMenu>
#include <QMessageBox>
#include <QStatusBar>

#include <algorithm>
#include <limits>

namespace bitabyte::ui {

void MainWindow::addColumnDefinition() {
    if (!dataSource_.hasData()) {
        QMessageBox::information(this, QStringLiteral("Add Column"), QStringLiteral("Load a file first."));
        return;
    }

    ColumnDefinitionDialog dialog(this, nextColumnColorName());
    dialog.setWindowTitle(QStringLiteral("Add Column Definition"));
    if (dialog.exec() != QDialog::DialogCode::Accepted) {
        return;
    }

    columnDefinitions_.append(dialog.definition());
    byteTableModel_->reload();
    refreshColumnDefinitionsPanel();
    resizeTableColumns();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
    statusBar()->showMessage(QStringLiteral("Added column definition"), 3000);
}

void MainWindow::defineColumnFromSelection() {
    if (!dataSource_.hasData()) {
        QMessageBox::information(this, QStringLiteral("Define Column"), QStringLiteral("Load a file first."));
        return;
    }

    QSet<int> selectedColumns = selectedVisibleColumns();
    if (selectedColumns.size() == 1) {
        selectedColumns = editableVisibleColumnsForSeed(*selectedColumns.begin());
    }

    if (selectedColumns.isEmpty()) {
        QMessageBox::warning(
            this,
            QStringLiteral("Define Column"),
            QStringLiteral("Select one or more columns first.")
        );
        return;
    }

    features::columns::ByteColumnDefinition definition;
    QString errorMessage;
    if (!buildDefinitionFromSelection(selectedColumns, &definition, &errorMessage)) {
        QMessageBox::warning(
            this,
            QStringLiteral("Define Column"),
            errorMessage.isEmpty() ? QStringLiteral("Unable to define the selected columns.") : errorMessage
        );
        return;
    }

    ColumnDefinitionDialog dialog(this, definition.colorName);
    dialog.setWindowTitle(QStringLiteral("Add Column Definition from Selection"));
    dialog.setDefinition(definition);
    if (dialog.exec() != QDialog::DialogCode::Accepted) {
        return;
    }

    const features::columns::ByteColumnDefinition acceptedDefinition = dialog.definition();
    const bool selectionCameFromSplit = !byteTableModel_->splitByteTargetsForVisibleColumns(selectedColumns).isEmpty();
    if (selectionCameFromSplit) {
        const bool removedSplitRange = byteTableModel_->removeSplitBitRange(
            acceptedDefinition.startAbsoluteBit(),
            acceptedDefinition.endAbsoluteBit()
        );
        Q_UNUSED(removedSplitRange);
    }

    columnDefinitions_.append(acceptedDefinition);
    byteTableModel_->reload();
    refreshColumnDefinitionsPanel();
    resizeTableColumns();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
    statusBar()->showMessage(
        !selectionCameFromSplit
            ? QStringLiteral("Added column definition from selection")
            : QStringLiteral("Promoted split to column definition"),
        3000
    );
}

void MainWindow::editColumnDefinition(int definitionIndex) {
    if (definitionIndex < 0 || definitionIndex >= columnDefinitions_.size()) {
        return;
    }

    ColumnDefinitionDialog dialog(this);
    dialog.setWindowTitle(QStringLiteral("Edit Column Definition"));
    dialog.setDefinition(columnDefinitions_[definitionIndex]);
    if (dialog.exec() != QDialog::DialogCode::Accepted) {
        return;
    }

    columnDefinitions_[definitionIndex] = dialog.definition();
    byteTableModel_->reload();
    refreshColumnDefinitionsPanel();
    resizeTableColumns();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
    statusBar()->showMessage(QStringLiteral("Updated column definition"), 3000);
}

void MainWindow::editVisibleColumns(const QSet<int>& visibleColumns) {
    if (visibleColumns.isEmpty()) {
        return;
    }

    selectVisibleColumns(visibleColumns);

    const int definitionIndex = definitionIndexForVisibleColumns(visibleColumns);
    if (definitionIndex >= 0) {
        editColumnDefinition(definitionIndex);
        return;
    }

    defineColumnFromSelection();
}

void MainWindow::removeColumnDefinition(int definitionIndex) {
    if (definitionIndex < 0 || definitionIndex >= columnDefinitions_.size()) {
        return;
    }

    const QString label = columnDefinitions_[definitionIndex].label;
    columnDefinitions_.removeAt(definitionIndex);
    byteTableModel_->reload();
    refreshColumnDefinitionsPanel();
    resizeTableColumns();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
    statusBar()->showMessage(
        QStringLiteral("Removed column %1").arg(label.isEmpty() ? QStringLiteral("(unnamed)") : label),
        3000
    );
}

void MainWindow::splitSelectionAsBinary() {
    const QSet<int> selectedColumns = selectedVisibleColumns();
    const QSet<int> plainByteTargets = byteTableModel_->plainByteTargetsForVisibleColumns(selectedColumns);
    bool changed = false;

    if (!plainByteTargets.isEmpty() && plainByteTargets.size() == selectedColumns.size()) {
        changed = byteTableModel_->applySplit(plainByteTargets, QStringLiteral("binary"));
    } else if (byteTableModel_->selectionCanBecomeBinary(selectedColumns)) {
        changed = byteTableModel_->convertSelectionToBinary(selectedColumns);
    }

    if (!changed) {
        QMessageBox::information(
            this,
            QStringLiteral("Split as Binary"),
            QStringLiteral("Select whole plain byte columns, or select one 4-bit nibble segment.")
        );
        return;
    }

    if (frameBrowserController_ != nullptr) {
        frameBrowserController_->saveCurrentSplitState();
    }
    resizeTableColumns();
    updateLoadedFileState();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
    statusBar()->showMessage(QStringLiteral("Applied binary split"), 3000);
}

void MainWindow::splitSelectionAsNibbles() {
    const QSet<int> selectedColumns = selectedVisibleColumns();
    const QSet<int> plainByteTargets = byteTableModel_->plainByteTargetsForVisibleColumns(selectedColumns);
    bool changed = false;

    if (!plainByteTargets.isEmpty() && plainByteTargets.size() == selectedColumns.size()) {
        changed = byteTableModel_->applySplit(plainByteTargets, QStringLiteral("nibble"));
    } else if (byteTableModel_->selectionCanBecomeNibble(selectedColumns)) {
        changed = byteTableModel_->convertSelectionToNibble(selectedColumns);
    }

    if (!changed) {
        QMessageBox::information(
            this,
            QStringLiteral("Split as Nibbles"),
            QStringLiteral("Select whole plain byte columns, or select one 4-bit binary range.")
        );
        return;
    }

    if (frameBrowserController_ != nullptr) {
        frameBrowserController_->saveCurrentSplitState();
    }
    resizeTableColumns();
    updateLoadedFileState();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
    statusBar()->showMessage(QStringLiteral("Applied nibble split"), 3000);
}

void MainWindow::clearSelectionSplits() {
    const QSet<int> splitByteTargets = byteTableModel_->splitByteTargetsForVisibleColumns(selectedVisibleColumns());
    if (splitByteTargets.isEmpty()) {
        QMessageBox::information(
            this,
            QStringLiteral("Clear Splits"),
            QStringLiteral("Select one or more split columns first.")
        );
        return;
    }

    if (!byteTableModel_->clearSplits(splitByteTargets)) {
        return;
    }

    if (frameBrowserController_ != nullptr) {
        frameBrowserController_->saveCurrentSplitState();
    }
    resizeTableColumns();
    updateLoadedFileState();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
    statusBar()->showMessage(QStringLiteral("Cleared selected splits"), 3000);
}

void MainWindow::combineSelection() {
    QString errorMessage;
    if (!combineSelectedVisibleColumns(&errorMessage)) {
        QMessageBox::warning(
            this,
            QStringLiteral("Combine Selection"),
            errorMessage.isEmpty() ? QStringLiteral("Unable to combine the selected columns.") : errorMessage
        );
        return;
    }

    byteTableModel_->reload();
    resizeTableColumns();
    updateLoadedFileState();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
    statusBar()->showMessage(QStringLiteral("Combined selection"), 3000);
}

void MainWindow::showTableContextMenu(const QPoint& pos) {
    if (byteTableView_ == nullptr || byteTableModel_ == nullptr) {
        return;
    }

    const QModelIndex clickedIndex = byteTableView_->indexAt(pos);
    if (clickedIndex.isValid() && byteTableView_->selectionModel() != nullptr
        && !byteTableView_->selectionModel()->isSelected(clickedIndex)) {
        byteTableView_->selectionModel()->setCurrentIndex(
            clickedIndex,
            QItemSelectionModel::SelectionFlag::ClearAndSelect | QItemSelectionModel::SelectionFlag::Current
        );
    }

    const QSet<int> selectedColumns = selectedVisibleColumns();
    const QSet<int> plainByteTargets = byteTableModel_->plainByteTargetsForVisibleColumns(selectedColumns);
    const int selectedDefinitionIndex = selectedDefinitionIndexFromCurrentSelection();
    const bool canSplitToBinary = byteTableModel_->selectionCanBecomeBinary(selectedColumns)
        || (!plainByteTargets.isEmpty() && plainByteTargets.size() == selectedColumns.size());
    const bool canSplitToNibbles = byteTableModel_->selectionCanBecomeNibble(selectedColumns)
        || (!plainByteTargets.isEmpty() && plainByteTargets.size() == selectedColumns.size());
    const bool canClearSplits = !byteTableModel_->splitByteTargetsForVisibleColumns(selectedColumns).isEmpty();
    QString combineErrorMessage;
    const bool canCombine = selectedColumns.size() >= 2
        && !combinedDisplayFormat(selectedColumns, &combineErrorMessage).isEmpty();

    QMenu menu(this);
    QAction* frameSelectionMenuAction = menu.addAction(QStringLiteral("Frame Selection"));
    frameSelectionMenuAction->setEnabled(!selectedColumns.isEmpty());

    QAction* defineSelectionMenuAction = menu.addAction(QStringLiteral("Define Column from Selection"));
    defineSelectionMenuAction->setEnabled(!selectedColumns.isEmpty());

    QAction* combineSelectionMenuAction = menu.addAction(QStringLiteral("Combine Selection"));
    combineSelectionMenuAction->setEnabled(canCombine);

    menu.addSeparator();
    QAction* splitBinaryMenuAction = menu.addAction(QStringLiteral("Split Selection as Binary"));
    splitBinaryMenuAction->setEnabled(canSplitToBinary);
    QAction* splitNibblesMenuAction = menu.addAction(QStringLiteral("Split Selection as Nibbles"));
    splitNibblesMenuAction->setEnabled(canSplitToNibbles);
    QAction* clearSplitsMenuAction = menu.addAction(QStringLiteral("Clear Selected Splits"));
    clearSplitsMenuAction->setEnabled(canClearSplits);
    QAction* groupSelectionMenuAction = nullptr;
    const bool canGroupSelection =
        frameLayout_.isFramed()
        && !selectedColumns.isEmpty()
        && byteTableModel_->visibleColumnsAreContiguous(selectedColumns);
    if (canGroupSelection) {
        menu.addSeparator();
        groupSelectionMenuAction = menu.addAction(QStringLiteral("Filter by Column"));
    }

    QAction* editColumnMenuAction = nullptr;
    QAction* removeColumnMenuAction = nullptr;
    if (!selectedColumns.isEmpty()) {
        menu.addSeparator();
        editColumnMenuAction = menu.addAction(QStringLiteral("Edit Column"));
    }
    if (selectedDefinitionIndex >= 0) {
        removeColumnMenuAction = menu.addAction(QStringLiteral("Remove Column"));
    }

    QAction* chosenAction = menu.exec(byteTableView_->viewport()->mapToGlobal(pos));
    if (chosenAction == frameSelectionMenuAction) {
        if (framingController_ != nullptr) {
            framingController_->frameSelection();
        }
    } else if (chosenAction == defineSelectionMenuAction) {
        defineColumnFromSelection();
    } else if (chosenAction == combineSelectionMenuAction) {
        combineSelection();
    } else if (chosenAction == splitBinaryMenuAction) {
        splitSelectionAsBinary();
    } else if (chosenAction == splitNibblesMenuAction) {
        splitSelectionAsNibbles();
    } else if (chosenAction == clearSplitsMenuAction) {
        clearSelectionSplits();
    } else if (chosenAction == groupSelectionMenuAction) {
        if (frameBrowserController_ != nullptr) {
            frameBrowserController_->applyGroupingKeysFromSelection(selectedColumns, true);
        }
    } else if (chosenAction == editColumnMenuAction) {
        editVisibleColumns(selectedColumns);
    } else if (chosenAction == removeColumnMenuAction) {
        removeColumnDefinition(selectedDefinitionIndex);
    }
}

void MainWindow::refreshColumnDefinitionsPanel() {
    const bool hasSelectableColumns = hasColumnSelection();
    const QSet<int> selectedColumns = selectedVisibleColumns();
    const QSet<int> plainByteTargets = byteTableModel_ != nullptr
        ? byteTableModel_->plainByteTargetsForVisibleColumns(selectedColumns)
        : QSet<int>{};
    if (frameSelectionAction_ != nullptr) {
        frameSelectionAction_->setEnabled(hasSelectableColumns);
    }
    if (defineSelectionColumnAction_ != nullptr) {
        defineSelectionColumnAction_->setEnabled(hasSelectableColumns);
    }
    if (combineSelectionAction_ != nullptr) {
        QString combineErrorMessage;
        combineSelectionAction_->setEnabled(
            hasSelectableColumns && !combinedDisplayFormat(selectedColumns, &combineErrorMessage).isEmpty()
        );
    }
    if (splitBinaryAction_ != nullptr) {
        const bool canSplitToBinary = byteTableModel_ != nullptr
            && (byteTableModel_->selectionCanBecomeBinary(selectedColumns)
                || (!plainByteTargets.isEmpty() && plainByteTargets.size() == selectedColumns.size()));
        splitBinaryAction_->setEnabled(canSplitToBinary);
    }
    if (splitNibblesAction_ != nullptr) {
        const bool canSplitToNibbles = byteTableModel_ != nullptr
            && (byteTableModel_->selectionCanBecomeNibble(selectedColumns)
                || (!plainByteTargets.isEmpty() && plainByteTargets.size() == selectedColumns.size()));
        splitNibblesAction_->setEnabled(canSplitToNibbles);
    }
    if (clearSelectionSplitsAction_ != nullptr) {
        clearSelectionSplitsAction_->setEnabled(
            byteTableModel_ != nullptr
            && !byteTableModel_->splitByteTargetsForVisibleColumns(selectedColumns).isEmpty()
        );
    }

    if (columnDefinitionsPanel_ == nullptr) {
        return;
    }

    columnDefinitionsPanel_->setEntries(columnDefinitions_, {});
    columnDefinitionsPanel_->setSelectionEnabled(hasSelectableColumns);
}

QString MainWindow::nextColumnColorName() const {
    static const QStringList autoColorSequence = {
        QStringLiteral("Sky"),
        QStringLiteral("Coral"),
        QStringLiteral("Mint"),
        QStringLiteral("Gold"),
        QStringLiteral("Lilac"),
        QStringLiteral("Sunshine"),
    };

    QHash<QString, int> usedCounts;
    for (const QString& colorName : autoColorSequence) {
        usedCounts.insert(colorName, 0);
    }

    for (const features::columns::ByteColumnDefinition& definition : columnDefinitions_) {
        if (usedCounts.contains(definition.colorName)) {
            usedCounts[definition.colorName] += 1;
        }
    }

    QString bestColor = autoColorSequence.first();
    int bestCount = std::numeric_limits<int>::max();
    for (const QString& colorName : autoColorSequence) {
        const int usedCount = usedCounts.value(colorName, 0);
        if (usedCount < bestCount) {
            bestCount = usedCount;
            bestColor = colorName;
        }
    }

    return bestColor;
}

bool MainWindow::buildDefinitionFromSelection(
    const QSet<int>& selectedVisibleColumns,
    features::columns::ByteColumnDefinition* definition,
    QString* errorMessage
) const {
    if (definition != nullptr) {
        *definition = {};
    }

    if (byteTableModel_ == nullptr || selectedVisibleColumns.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Select one or more columns to define.");
        }
        return false;
    }

    if (!byteTableModel_->visibleColumnsAreContiguous(selectedVisibleColumns)) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Selected columns must be adjacent to define a field.");
        }
        return false;
    }

    QList<int> sortedColumns = selectedVisibleColumns.values();
    std::sort(sortedColumns.begin(), sortedColumns.end());

    const features::columns::VisibleByteColumn firstVisibleColumn =
        byteTableModel_->visibleByteColumn(sortedColumns.first());
    const features::columns::VisibleByteColumn lastVisibleColumn =
        byteTableModel_->visibleByteColumn(sortedColumns.last());

    features::columns::ByteColumnDefinition selectionDefinition;
    selectionDefinition.colorName = nextColumnColorName();
    QString sharedSplitLabel;
    QString sharedSplitColorName;
    QString sharedSplitDisplayFormat;
    bool allSelectedColumnsShareSplit = true;

    bool allWholeBytes = true;
    for (int visibleColumnIndex : sortedColumns) {
        const features::columns::VisibleByteColumn visibleColumn = byteTableModel_->visibleByteColumn(visibleColumnIndex);
        if (visibleColumn.splitLabel.trimmed().isEmpty()) {
            allSelectedColumnsShareSplit = false;
        } else if (sharedSplitLabel.isEmpty()) {
            sharedSplitLabel = visibleColumn.splitLabel.trimmed();
            sharedSplitColorName = visibleColumn.splitColorName.trimmed();
            sharedSplitDisplayFormat = visibleColumn.splitDisplayFormat.trimmed();
        } else if (sharedSplitLabel != visibleColumn.splitLabel.trimmed()) {
            allSelectedColumnsShareSplit = false;
        }

        if (visibleColumn.definitionIndex >= 0
            || !visibleColumn.splitLabel.isEmpty()
            || visibleColumn.bitStart != 0
            || visibleColumn.bitEnd != 7) {
            allWholeBytes = false;
            break;
        }
    }

    if (allWholeBytes) {
        selectionDefinition.unit = QStringLiteral("byte");
        selectionDefinition.startByte = firstVisibleColumn.byteIndex;
        selectionDefinition.endByte = lastVisibleColumn.byteIndex;
        selectionDefinition.startBit = selectionDefinition.startByte * 8;
        selectionDefinition.totalBits = selectionDefinition.byteCount() * 8;
        selectionDefinition.displayFormat = QStringLiteral("hex");
        selectionDefinition.label = selectionDefinition.startByte == selectionDefinition.endByte
            ? QStringLiteral("Byte %1").arg(selectionDefinition.startByte)
            : QStringLiteral("Bytes %1-%2").arg(selectionDefinition.startByte).arg(selectionDefinition.endByte);
    } else {
        selectionDefinition.unit = QStringLiteral("bit");
        selectionDefinition.startBit = firstVisibleColumn.absoluteStartBit;
        selectionDefinition.totalBits = lastVisibleColumn.absoluteEndBit - firstVisibleColumn.absoluteStartBit + 1;
        selectionDefinition.startByte = selectionDefinition.startBit / 8;
        selectionDefinition.endByte = (selectionDefinition.startBit + selectionDefinition.totalBits - 1) / 8;
        selectionDefinition.displayFormat = (selectionDefinition.totalBits % 4 == 0)
            ? QStringLiteral("hex")
            : QStringLiteral("binary");
        selectionDefinition.label = allSelectedColumnsShareSplit ? sharedSplitLabel : QString();
        if (allSelectedColumnsShareSplit && !sharedSplitColorName.isEmpty()) {
            selectionDefinition.colorName = sharedSplitColorName;
        }
        const QString normalizedSplitDisplayFormat = sharedSplitDisplayFormat.trimmed().toLower();
        if (allSelectedColumnsShareSplit
            && !sharedSplitDisplayFormat.isEmpty()
            && !(normalizedSplitDisplayFormat == QStringLiteral("binary")
                && selectionDefinition.totalBits % 4 == 0)) {
            selectionDefinition.displayFormat = sharedSplitDisplayFormat;
        }
    }

    if (definition != nullptr) {
        *definition = selectionDefinition;
    }
    return true;
}

QString MainWindow::combinedDisplayFormat(const QSet<int>& selectedVisibleColumns, QString* errorMessage) const {
    if (byteTableModel_ == nullptr || selectedVisibleColumns.size() < 2) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Select at least two columns to combine.");
        }
        return {};
    }

    if (!byteTableModel_->visibleColumnsAreContiguous(selectedVisibleColumns)) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Selected columns must be adjacent to combine.");
        }
        return {};
    }

    QList<int> sortedColumns = selectedVisibleColumns.values();
    std::sort(sortedColumns.begin(), sortedColumns.end());

    bool hasDecimal = false;
    bool hasBinary = false;
    bool hasHex = false;
    int totalBits = 0;
    for (int visibleColumnIndex : sortedColumns) {
        const features::columns::VisibleByteColumn visibleColumn = byteTableModel_->visibleByteColumn(visibleColumnIndex);
        QString displayFormat = QStringLiteral("hex");
        if (const std::optional<int> definitionIndex = byteTableModel_->visibleDefinitionIndex(visibleColumnIndex);
            definitionIndex.has_value()) {
            displayFormat = columnDefinitions_.at(definitionIndex.value()).displayFormat;
        } else if (!visibleColumn.splitDisplayFormat.isEmpty()) {
            displayFormat = visibleColumn.splitDisplayFormat;
        } else if (visibleColumn.bitWidth() == 1) {
            displayFormat = QStringLiteral("binary");
        }

        const QString normalizedFormat = displayFormat.trimmed().toLower();
        if (normalizedFormat == QStringLiteral("decimal")) {
            hasDecimal = true;
        } else if (normalizedFormat == QStringLiteral("binary")) {
            hasBinary = true;
        } else {
            hasHex = true;
        }
        totalBits += visibleColumn.bitWidth();
    }

    if (hasDecimal) {
        return QStringLiteral("decimal");
    }
    if ((hasHex || hasBinary) && totalBits % 4 == 0) {
        return QStringLiteral("hex");
    }
    if (hasHex && totalBits % 4 != 0) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Hex columns can only be combined into widths that are a multiple of 4 bits.");
        }
        return {};
    }
    return QStringLiteral("binary");
}

QString MainWindow::nextCombinedLabel(const QSet<int>& selectedVisibleColumns) const {
    QString preservedLabel;
    for (int visibleColumnIndex : selectedVisibleColumns) {
        const std::optional<int> definitionIndex = byteTableModel_->visibleDefinitionIndex(visibleColumnIndex);
        if (!definitionIndex.has_value()) {
            continue;
        }

        const QString labelText = columnDefinitions_.at(definitionIndex.value()).label.trimmed();
        if (detail::isAutoGeneratedLabel(labelText)) {
            continue;
        }
        if (preservedLabel.isEmpty()) {
            preservedLabel = labelText;
            continue;
        }
        if (preservedLabel != labelText) {
            preservedLabel.clear();
            break;
        }
    }

    if (!preservedLabel.isEmpty()) {
        return preservedLabel;
    }

    int nextComboIndex = 0;
    for (const features::columns::ByteColumnDefinition& definition : columnDefinitions_) {
        const QString labelText = definition.label.trimmed();
        if (!labelText.startsWith(QStringLiteral("Combo "))) {
            continue;
        }
        bool parsed = false;
        const int comboIndex = labelText.mid(6).toInt(&parsed);
        if (parsed) {
            nextComboIndex = qMax(nextComboIndex, comboIndex + 1);
        }
    }
    return QStringLiteral("Combo %1").arg(nextComboIndex);
}

bool MainWindow::combineSelectedVisibleColumns(QString* errorMessage) {
    if (byteTableModel_ == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("No table model is available.");
        }
        return false;
    }

    const QSet<int> selectedColumns = selectedVisibleColumns();
    if (selectedColumns.size() < 2) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Select at least two columns to combine.");
        }
        return false;
    }

    QList<int> sortedColumns = selectedColumns.values();
    std::sort(sortedColumns.begin(), sortedColumns.end());

    QString displayFormat = combinedDisplayFormat(selectedColumns, errorMessage);
    if (displayFormat.isEmpty()) {
        return false;
    }

    const features::columns::VisibleByteColumn firstVisibleColumn = byteTableModel_->visibleByteColumn(sortedColumns.first());
    const features::columns::VisibleByteColumn lastVisibleColumn = byteTableModel_->visibleByteColumn(sortedColumns.last());

    features::columns::ByteColumnDefinition combinedDefinition;
    combinedDefinition.unit = QStringLiteral("bit");
    combinedDefinition.startBit = firstVisibleColumn.absoluteStartBit;
    combinedDefinition.totalBits = lastVisibleColumn.absoluteEndBit - firstVisibleColumn.absoluteStartBit + 1;
    combinedDefinition.startByte = combinedDefinition.startBit / 8;
    combinedDefinition.endByte = (combinedDefinition.startBit + combinedDefinition.totalBits - 1) / 8;
    combinedDefinition.displayFormat = displayFormat;
    combinedDefinition.colorName = nextColumnColorName();
    combinedDefinition.label = nextCombinedLabel(selectedColumns);

    const bool selectionCameFromSplit = !byteTableModel_->splitByteTargetsForVisibleColumns(selectedColumns).isEmpty();
    if (selectionCameFromSplit) {
        const bool removedSplitRange = byteTableModel_->removeSplitBitRange(
            combinedDefinition.startAbsoluteBit(),
            combinedDefinition.endAbsoluteBit()
        );
        Q_UNUSED(removedSplitRange);
    }

    QSet<int> definitionIndicesToRemove;
    for (int visibleColumnIndex : sortedColumns) {
        const std::optional<int> definitionIndex = byteTableModel_->visibleDefinitionIndex(visibleColumnIndex);
        if (definitionIndex.has_value()) {
            definitionIndicesToRemove.insert(definitionIndex.value());
        }
    }

    QList<int> sortedDefinitionIndices = definitionIndicesToRemove.values();
    std::sort(sortedDefinitionIndices.begin(), sortedDefinitionIndices.end(), std::greater<int>());
    const int insertIndex = sortedDefinitionIndices.isEmpty() ? columnDefinitions_.size() : sortedDefinitionIndices.last();
    for (int definitionIndex : sortedDefinitionIndices) {
        columnDefinitions_.removeAt(definitionIndex);
    }
    columnDefinitions_.insert(insertIndex, combinedDefinition);
    return true;
}

QSet<int> MainWindow::selectedVisibleColumns() const {
    QSet<int> visibleColumns;
    if (byteTableView_ == nullptr || byteTableView_->selectionModel() == nullptr || byteTableModel_ == nullptr) {
        return visibleColumns;
    }

    const QItemSelection selection = byteTableView_->selectionModel()->selection();
    for (const QItemSelectionRange& selectionRange : selection) {
        const int topRow = selectionRange.top();
        if (topRow < 0) {
            continue;
        }

        for (int modelColumn = selectionRange.left(); modelColumn <= selectionRange.right(); ++modelColumn) {
            const QModelIndex modelIndex = byteTableModel_->index(topRow, modelColumn);
            const int visibleColumnIndex = byteTableModel_->visibleColumnIndexForModelIndex(modelIndex);
            if (visibleColumnIndex >= 0) {
                visibleColumns.insert(visibleColumnIndex);
            }
        }
    }

    return visibleColumns;
}

QSet<int> MainWindow::visibleColumnsForAbsoluteBitRange(int startBit, int endBit) const {
    QSet<int> visibleColumns;
    if (byteTableModel_ == nullptr || startBit > endBit) {
        return visibleColumns;
    }

    for (int visibleColumnIndex = 0; visibleColumnIndex < byteTableModel_->visibleColumnCount(); ++visibleColumnIndex) {
        const features::columns::VisibleByteColumn visibleColumn = byteTableModel_->visibleByteColumn(visibleColumnIndex);
        if (visibleColumn.absoluteEndBit < startBit || visibleColumn.absoluteStartBit > endBit) {
            continue;
        }
        visibleColumns.insert(visibleColumnIndex);
    }

    return visibleColumns;
}

QSet<int> MainWindow::editableVisibleColumnsForSeed(int visibleColumnIndex) const {
    QSet<int> visibleColumns;
    if (byteTableModel_ == nullptr
        || visibleColumnIndex < 0
        || visibleColumnIndex >= byteTableModel_->visibleColumnCount()) {
        return visibleColumns;
    }

    if (const std::optional<int> definitionIndex = byteTableModel_->visibleDefinitionIndex(visibleColumnIndex);
        definitionIndex.has_value()) {
        for (int candidateColumnIndex = 0; candidateColumnIndex < byteTableModel_->visibleColumnCount(); ++candidateColumnIndex) {
            const std::optional<int> candidateDefinitionIndex = byteTableModel_->visibleDefinitionIndex(candidateColumnIndex);
            if (candidateDefinitionIndex.has_value() && candidateDefinitionIndex.value() == definitionIndex.value()) {
                visibleColumns.insert(candidateColumnIndex);
            }
        }
        return visibleColumns;
    }

    const features::columns::VisibleByteColumn seedVisibleColumn = byteTableModel_->visibleByteColumn(visibleColumnIndex);
    if (!seedVisibleColumn.splitLabel.trimmed().isEmpty()) {
        for (int candidateColumnIndex = 0; candidateColumnIndex < byteTableModel_->visibleColumnCount(); ++candidateColumnIndex) {
            const features::columns::VisibleByteColumn candidateVisibleColumn =
                byteTableModel_->visibleByteColumn(candidateColumnIndex);
            if (candidateVisibleColumn.byteIndex == seedVisibleColumn.byteIndex
                && candidateVisibleColumn.byteEndIndex == seedVisibleColumn.byteEndIndex
                && candidateVisibleColumn.splitLabel == seedVisibleColumn.splitLabel) {
                visibleColumns.insert(candidateColumnIndex);
            }
        }
    }

    if (visibleColumns.isEmpty()) {
        visibleColumns.insert(visibleColumnIndex);
    }

    return visibleColumns;
}

void MainWindow::selectVisibleColumns(const QSet<int>& visibleColumns) {
    if (byteTableView_ == nullptr
        || byteTableView_->selectionModel() == nullptr
        || byteTableModel_ == nullptr
        || visibleColumns.isEmpty()
        || byteTableModel_->rowCount() <= 0) {
        return;
    }

    QList<int> sortedColumns = visibleColumns.values();
    std::sort(sortedColumns.begin(), sortedColumns.end());

    int targetRow = byteTableView_->currentIndex().isValid() ? byteTableView_->currentIndex().row() : 0;
    targetRow = qBound(0, targetRow, byteTableModel_->rowCount() - 1);

    const int modelColumnOffset = byteTableModel_->hasFrameLengthColumn() ? 1 : 0;
    QItemSelection columnSelection;
    for (int visibleColumnIndex : sortedColumns) {
        const QModelIndex modelIndex = byteTableModel_->index(targetRow, visibleColumnIndex + modelColumnOffset);
        if (!modelIndex.isValid()) {
            continue;
        }
        columnSelection.select(modelIndex, modelIndex);
    }

    if (columnSelection.isEmpty()) {
        return;
    }

    byteTableView_->selectionModel()->select(
        columnSelection,
        QItemSelectionModel::SelectionFlag::ClearAndSelect
    );
    byteTableView_->selectionModel()->setCurrentIndex(
        byteTableModel_->index(targetRow, sortedColumns.first() + modelColumnOffset),
        QItemSelectionModel::SelectionFlag::ClearAndSelect | QItemSelectionModel::SelectionFlag::Current
    );
}

QModelIndex MainWindow::topSelectedDataIndex() const {
    if (byteTableView_ == nullptr || byteTableView_->selectionModel() == nullptr || byteTableModel_ == nullptr) {
        return {};
    }

    QModelIndex bestIndex;
    const QItemSelection selection = byteTableView_->selectionModel()->selection();
    for (const QItemSelectionRange& selectionRange : selection) {
        for (int modelColumn = selectionRange.left(); modelColumn <= selectionRange.right(); ++modelColumn) {
            const QModelIndex modelIndex = byteTableModel_->index(selectionRange.top(), modelColumn);
            if (byteTableModel_->visibleColumnIndexForModelIndex(modelIndex) < 0) {
                continue;
            }

            if (!bestIndex.isValid()
                || modelIndex.row() < bestIndex.row()
                || (modelIndex.row() == bestIndex.row() && modelIndex.column() < bestIndex.column())) {
                bestIndex = modelIndex;
            }
        }
    }

    return bestIndex;
}

bool MainWindow::extractSelectionPattern(QString* patternText, QString* errorMessage) const {
    if (patternText != nullptr) {
        patternText->clear();
    }

    const QSet<int> selectedColumnSet = selectedVisibleColumns();
    QList<int> selectedColumns = selectedColumnSet.values();
    if (selectedColumns.isEmpty()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Select one or more columns first.");
        }
        return false;
    }
    std::sort(selectedColumns.begin(), selectedColumns.end());

    if (!byteTableModel_->visibleColumnsAreContiguous(selectedColumnSet)) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Select adjacent columns to extract a frame pattern.");
        }
        return false;
    }

    const QModelIndex sourceRowIndex = topSelectedDataIndex();
    if (!sourceRowIndex.isValid()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Select one or more cells first.");
        }
        return false;
    }

    const int firstVisibleColumnIndex = selectedColumns.first();
    const int lastVisibleColumnIndex = selectedColumns.last();
    const features::columns::VisibleByteColumn firstVisibleColumn =
        byteTableModel_->visibleByteColumn(firstVisibleColumnIndex);
    const features::columns::VisibleByteColumn lastVisibleColumn =
        byteTableModel_->visibleByteColumn(lastVisibleColumnIndex);
    const qsizetype rowStartBit = frameLayout_.rowStartBit(dataSource_, sourceRowIndex.row());
    const qsizetype rowLengthBits = frameLayout_.rowLengthBits(dataSource_, sourceRowIndex.row());

    if (firstVisibleColumn.absoluteStartBit >= rowLengthBits) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Selection is out of range.");
        }
        return false;
    }

    QString extractedPattern;
    const int endBit = qMin<qsizetype>(lastVisibleColumn.absoluteEndBit, rowLengthBits - 1);
    const int selectedBitCount = endBit - firstVisibleColumn.absoluteStartBit + 1;
    if (selectedBitCount <= 0) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("Selection is out of range.");
        }
        return false;
    }

    if (selectedBitCount % 4 == 0) {
        static constexpr char kHexDigits[] = "0123456789ABCDEF";
        extractedPattern = QStringLiteral("0x");
        extractedPattern.reserve(2 + (selectedBitCount / 4));
        int nibbleValue = 0;
        int nibbleBitCount = 0;
        for (int relativeBit = firstVisibleColumn.absoluteStartBit; relativeBit <= endBit; ++relativeBit) {
            nibbleValue = (nibbleValue << 1) | (dataSource_.bitAt(rowStartBit + relativeBit) != 0 ? 1 : 0);
            ++nibbleBitCount;
            if (nibbleBitCount == 4) {
                extractedPattern.append(QLatin1Char(kHexDigits[nibbleValue & 0x0F]));
                nibbleValue = 0;
                nibbleBitCount = 0;
            }
        }
    } else {
        extractedPattern.reserve(selectedBitCount);
        for (int relativeBit = firstVisibleColumn.absoluteStartBit; relativeBit <= endBit; ++relativeBit) {
            extractedPattern.append(
                dataSource_.bitAt(rowStartBit + relativeBit) != 0 ? QLatin1Char('1') : QLatin1Char('0')
            );
        }
    }

    if (patternText != nullptr) {
        *patternText = extractedPattern;
    }
    return !extractedPattern.isEmpty();
}

bool MainWindow::hasColumnSelection() const {
    return !selectedVisibleColumns().isEmpty();
}

int MainWindow::definitionIndexForVisibleColumns(const QSet<int>& visibleColumns) const {
    QList<int> sortedColumns = visibleColumns.values();
    if (sortedColumns.isEmpty() || byteTableModel_ == nullptr) {
        return -1;
    }

    std::optional<int> matchingDefinitionIndex;
    for (int visibleColumnIndex : sortedColumns) {
        const std::optional<int> definitionIndex = byteTableModel_->visibleDefinitionIndex(visibleColumnIndex);
        if (!definitionIndex.has_value()) {
            return -1;
        }
        if (!matchingDefinitionIndex.has_value()) {
            matchingDefinitionIndex = definitionIndex;
            continue;
        }
        if (matchingDefinitionIndex.value() != definitionIndex.value()) {
            return -1;
        }
    }

    return matchingDefinitionIndex.value_or(-1);
}

int MainWindow::selectedDefinitionIndexFromCurrentSelection() const {
    return definitionIndexForVisibleColumns(selectedVisibleColumns());
}

}  // namespace bitabyte::ui
