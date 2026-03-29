#include "ui/frame_browser_controller.h"

#include "models/frame_group_tree_model.h"
#include "ui/byte_table_view.h"
#include "ui/frame_grouping_panel.h"
#include "ui/main_window_internal.h"

#include <QHeaderView>
#include <QMenu>
#include <QStringList>

#include <algorithm>

namespace bitabyte::ui {

FrameBrowserController::FrameBrowserController(
    data::ByteDataSource& dataSource,
    features::framing::FrameLayout& frameLayout,
    QVector<features::columns::ByteColumnDefinition>& columnDefinitions,
    models::ByteTableModel& byteTableModel,
    ByteTableView& byteTableView,
    models::FrameGroupTreeModel& frameGroupTreeModel,
    Callbacks callbacks
)
    : dataSource_(dataSource),
      frameLayout_(frameLayout),
      columnDefinitions_(columnDefinitions),
      byteTableModel_(byteTableModel),
      byteTableView_(byteTableView),
      frameGroupTreeModel_(frameGroupTreeModel),
      callbacks_(std::move(callbacks)) {}

void FrameBrowserController::setPanel(FrameGroupingPanel* frameGroupingPanel) {
    frameGroupingPanel_ = frameGroupingPanel;
    refreshPanel();
}

void FrameBrowserController::setFrameSpans(const QVector<features::framing::FrameSpan>& frameSpans) {
    frameSpans_ = frameSpans;
    activeScopedFrameStartBits_.clear();
    filterClauses_.clear();
    resetSplitScopeState();
    frameGroupTreeModel_.setActiveScopePath({});
}

const QVector<features::framing::FrameSpan>& FrameBrowserController::frameSpans() const noexcept {
    return frameSpans_;
}

void FrameBrowserController::refreshPanel() {
    if (frameGroupingPanel_ == nullptr) {
        return;
    }

    frameGroupingPanel_->setGroupingKeys(effectiveGroupingKeysForActiveScope());
    frameGroupingPanel_->setFramingActive(frameLayout_.isFramed() && dataSource_.hasData());
    frameGroupingPanel_->setStatusText(summaryText());
    frameGroupTreeModel_.setFrameSource(
        dataSource_.hasData() && !frameSpans_.isEmpty() ? &dataSource_ : nullptr,
        !frameSpans_.isEmpty() ? &frameSpans_ : nullptr
    );
    frameGroupTreeModel_.setGroupingKeys(groupingKeys_);
    frameGroupTreeModel_.setScopedGroupingKeys(scopedGroupingKeysByScopeKey_);
    frameGroupTreeModel_.setFilterClauses(filterClauses_);
    frameGroupingPanel_->setActiveScopePath(frameGroupTreeModel_.activeScopePath());
}

void FrameBrowserController::applyState() {
    if (frameSpans_.isEmpty()) {
        refreshPanel();
        if (callbacks_.updateLoadedFileState) {
            callbacks_.updateLoadedFileState();
        }
        return;
    }

    QVector<features::framing::FrameSpan> visibleFrameSpans;
    visibleFrameSpans.reserve(frameSpans_.size());
    for (const features::framing::FrameSpan& frameSpan : frameSpans_) {
        if (!activeScopedFrameStartBits_.isEmpty() && !activeScopedFrameStartBits_.contains(frameSpan.startBit)) {
            continue;
        }

        bool includeFrame = true;
        for (const features::frame_browser::FrameFilterClause& filterClause : filterClauses_) {
            if (!features::frame_browser::frameMatchesFilterClause(dataSource_, frameSpan, filterClause)) {
                includeFrame = false;
                break;
            }
        }
        if (includeFrame) {
            visibleFrameSpans.append(frameSpan);
        }
    }

    frameLayout_.setFrames(visibleFrameSpans);
    if (callbacks_.syncFramingControlsFromState) {
        callbacks_.syncFramingControlsFromState();
    }
    applySplitStateForActiveScope();
    if (callbacks_.resizeTableColumns) {
        callbacks_.resizeTableColumns();
    }
    byteTableView_.clearSelection();
    byteTableView_.setCurrentIndex(QModelIndex());
    refreshPanel();
    if (callbacks_.updateLoadedFileState) {
        callbacks_.updateLoadedFileState();
    }
    if (callbacks_.updateSelectionStatus) {
        callbacks_.updateSelectionStatus();
    }
    if (callbacks_.refreshLiveBitViewer) {
        callbacks_.refreshLiveBitViewer();
    }
}

void FrameBrowserController::clearState() {
    frameSpans_.clear();
    groupingKeys_.clear();
    scopedGroupingKeysByScopeKey_.clear();
    filterClauses_.clear();
    activeScopePath_.clear();
    activeScopeKey_.clear();
    activeScopedFrameStartBits_.clear();
    frameGroupTreeModel_.setActiveScopePath({});
    resetSplitScopeState();
    refreshPanel();
}

void FrameBrowserController::clearGrouping() {
    groupingKeys_.clear();
    applyState();
}

void FrameBrowserController::clearFilters() {
    if (filterClauses_.isEmpty()) {
        return;
    }
    filterClauses_.clear();
    applyState();
}

void FrameBrowserController::clearScope() {
    if (activeScopedFrameStartBits_.isEmpty()) {
        return;
    }

    if (activeScopePath_.size() <= 1) {
        saveCurrentSplitState();
        activeScopePath_.clear();
        activeScopeKey_.clear();
        activeScopedFrameStartBits_.clear();
        frameGroupTreeModel_.setActiveScopePath({});
        applyState();
        return;
    }

    QVector<features::frame_browser::FrameGroupValue> parentScopePath = activeScopePath_;
    parentScopePath.removeLast();
    const QModelIndex parentScopeIndex = frameGroupTreeModel_.indexForPath(parentScopePath);
    if (!parentScopeIndex.isValid()) {
        return;
    }

    applyScopeForTreeIndex(parentScopeIndex);
}

void FrameBrowserController::showTableHeaderContextMenu(const QPoint& pos) {
    if (!frameLayout_.isFramed()) {
        return;
    }

    const int logicalIndex = byteTableView_.horizontalHeader()->logicalIndexAt(pos);
    if (logicalIndex < 0) {
        return;
    }

    QMenu menu(&byteTableView_);
    QAction* groupByAction = nullptr;
    QAction* thenGroupByAction = nullptr;
    if (byteTableModel_.isFrameLengthColumn(logicalIndex)) {
        groupByAction = menu.addAction(QStringLiteral("Group by Frame Length"));
        thenGroupByAction = menu.addAction(QStringLiteral("Then Group by Frame Length"));
    } else {
        const int visibleColumnIndex = byteTableModel_.hasFrameLengthColumn() ? logicalIndex - 1 : logicalIndex;
        if (visibleColumnIndex < 0) {
            return;
        }
        groupByAction = menu.addAction(QStringLiteral("Group by This Field"));
        thenGroupByAction = menu.addAction(QStringLiteral("Then Group by This Field"));
        QAction* chosenAction = menu.exec(byteTableView_.horizontalHeader()->mapToGlobal(pos));
        if (chosenAction == groupByAction) {
            addGroupingKeyFromVisibleSeed(visibleColumnIndex, false);
        } else if (chosenAction == thenGroupByAction) {
            addGroupingKeyFromVisibleSeed(visibleColumnIndex, true);
        }
        return;
    }

    QAction* chosenAction = menu.exec(byteTableView_.horizontalHeader()->mapToGlobal(pos));
    if (chosenAction != groupByAction && chosenAction != thenGroupByAction) {
        return;
    }

    const features::frame_browser::FrameGroupingKey frameLengthKey{
        features::frame_browser::FrameGroupingKeyKind::FrameLength,
        QStringLiteral("Frame Length"),
        QStringLiteral("decimal"),
        0,
        0,
    };
    const QVector<features::frame_browser::FrameGroupValue> parentScopePath = activeScopePath_;
    if (activeScopeKey_.isEmpty()) {
        if (chosenAction == groupByAction) {
            groupingKeys_ = {frameLengthKey};
        } else if (!groupingKeys_.contains(frameLengthKey)) {
            groupingKeys_.append(frameLengthKey);
        }
    } else {
        QVector<features::frame_browser::FrameGroupingKey>& scopedGroupingKeys =
            scopedGroupingKeysByScopeKey_[activeScopeKey_];
        if (chosenAction == groupByAction) {
            scopedGroupingKeys = {frameLengthKey};
        } else if (!scopedGroupingKeys.contains(frameLengthKey)) {
            scopedGroupingKeys.append(frameLengthKey);
        }
    }
    applyState();
    scopeToFirstChildBranch(parentScopePath);
}

bool FrameBrowserController::buildGroupingKeyForVisibleSeed(
    int visibleColumnIndex,
    features::frame_browser::FrameGroupingKey* groupingKey
) const {
    if (visibleColumnIndex < 0) {
        return false;
    }
    return buildGroupingKeyForVisibleColumns(QSet<int>{visibleColumnIndex}, groupingKey);
}

bool FrameBrowserController::buildGroupingKeyForVisibleColumns(
    const QSet<int>& visibleColumns,
    features::frame_browser::FrameGroupingKey* groupingKey
) const {
    if (groupingKey != nullptr) {
        *groupingKey = {};
    }
    if (visibleColumns.isEmpty() || !byteTableModel_.visibleColumnsAreContiguous(visibleColumns)) {
        return false;
    }

    QList<int> sortedColumns = visibleColumns.values();
    std::sort(sortedColumns.begin(), sortedColumns.end());
    const features::columns::VisibleByteColumn firstVisibleColumn =
        byteTableModel_.visibleByteColumn(sortedColumns.first());
    const features::columns::VisibleByteColumn lastVisibleColumn =
        byteTableModel_.visibleByteColumn(sortedColumns.last());

    features::frame_browser::FrameGroupingKey key;
    key.kind = features::frame_browser::FrameGroupingKeyKind::FieldValue;
    key.startBit = firstVisibleColumn.absoluteStartBit;
    key.bitWidth = lastVisibleColumn.absoluteEndBit - firstVisibleColumn.absoluteStartBit + 1;
    key.displayFormat = key.bitWidth % 4 == 0 ? QStringLiteral("hex") : QStringLiteral("binary");
    key.label = detail::defaultFieldLabel(firstVisibleColumn, lastVisibleColumn);

    const int definitionIndex = definitionIndexForVisibleColumns(visibleColumns);
    if (definitionIndex >= 0 && definitionIndex < columnDefinitions_.size()) {
        const features::columns::ByteColumnDefinition& definition = columnDefinitions_.at(definitionIndex);
        if (!definition.label.trimmed().isEmpty()) {
            key.label = definition.label.trimmed();
        }
        key.displayFormat = definition.displayFormat.trimmed().isEmpty()
            ? key.displayFormat
            : definition.displayFormat.trimmed().toLower();
    } else {
        QString sharedSplitLabel;
        QString sharedSplitDisplayFormat;
        bool allColumnsShareSplit = true;
        for (int candidateVisibleColumnIndex : sortedColumns) {
            const features::columns::VisibleByteColumn candidateVisibleColumn =
                byteTableModel_.visibleByteColumn(candidateVisibleColumnIndex);
            const QString splitLabel = candidateVisibleColumn.splitLabel.trimmed();
            if (splitLabel.isEmpty()) {
                allColumnsShareSplit = false;
                break;
            }

            if (sharedSplitLabel.isEmpty()) {
                sharedSplitLabel = splitLabel;
                sharedSplitDisplayFormat = candidateVisibleColumn.splitDisplayFormat.trimmed().toLower();
            } else if (sharedSplitLabel != splitLabel) {
                allColumnsShareSplit = false;
                break;
            }
        }

        if (allColumnsShareSplit && !sharedSplitLabel.isEmpty() && sortedColumns.size() > 1) {
            key.label = sharedSplitLabel;
            if (!sharedSplitDisplayFormat.isEmpty()) {
                key.displayFormat = sharedSplitDisplayFormat;
            }
        }
    }

    if (groupingKey != nullptr) {
        *groupingKey = key;
    }
    return true;
}

QVector<features::frame_browser::FrameGroupingKey> FrameBrowserController::buildGroupingKeysForSelection(
    const QSet<int>& selectedVisibleColumns
) const {
    QVector<features::frame_browser::FrameGroupingKey> groupingKeys;
    if (selectedVisibleColumns.isEmpty()) {
        return groupingKeys;
    }

    features::frame_browser::FrameGroupingKey groupingKey;
    if (buildGroupingKeyForVisibleColumns(selectedVisibleColumns, &groupingKey)) {
        groupingKeys.append(groupingKey);
    }
    return groupingKeys;
}

void FrameBrowserController::addGroupingKeyFromVisibleSeed(int visibleColumnIndex, bool appendToGroupingStack) {
    features::frame_browser::FrameGroupingKey groupingKey;
    if (!buildGroupingKeyForVisibleSeed(visibleColumnIndex, &groupingKey)) {
        return;
    }

    const QVector<features::frame_browser::FrameGroupValue> parentScopePath = activeScopePath_;
    QVector<features::frame_browser::FrameGroupingKey>* targetGroupingKeys = &groupingKeys_;
    if (!activeScopeKey_.isEmpty()) {
        targetGroupingKeys = &scopedGroupingKeysByScopeKey_[activeScopeKey_];
    }

    if (!appendToGroupingStack) {
        targetGroupingKeys->clear();
    }
    if (!targetGroupingKeys->contains(groupingKey)) {
        targetGroupingKeys->append(groupingKey);
    }
    applyState();
    scopeToFirstChildBranch(parentScopePath);
}

void FrameBrowserController::applyGroupingKeysFromSelection(
    const QSet<int>& selectedVisibleColumns,
    bool appendToGroupingStack
) {
    const QVector<features::frame_browser::FrameGroupingKey> selectionKeys =
        buildGroupingKeysForSelection(selectedVisibleColumns);
    if (selectionKeys.isEmpty()) {
        return;
    }

    const QVector<features::frame_browser::FrameGroupValue> parentScopePath = activeScopePath_;
    QVector<features::frame_browser::FrameGroupingKey>* targetGroupingKeys = &groupingKeys_;
    if (!activeScopeKey_.isEmpty()) {
        targetGroupingKeys = &scopedGroupingKeysByScopeKey_[activeScopeKey_];
    }

    if (!appendToGroupingStack) {
        targetGroupingKeys->clear();
    }
    for (const features::frame_browser::FrameGroupingKey& selectionKey : selectionKeys) {
        if (!targetGroupingKeys->contains(selectionKey)) {
            targetGroupingKeys->append(selectionKey);
        }
    }
    applyState();
    scopeToFirstChildBranch(parentScopePath);
}

void FrameBrowserController::removeGroupingKey(int keyIndex) {
    const QVector<features::frame_browser::FrameGroupingKey> effectiveGroupingKeys =
        effectiveGroupingKeysForActiveScope();
    if (keyIndex < 0 || keyIndex >= effectiveGroupingKeys.size()) {
        return;
    }

    if (activeScopeKey_.isEmpty()) {
        groupingKeys_.removeAt(keyIndex);
        applyState();
        return;
    }

    const int rootKeyCount = groupingKeys_.size();
    if (keyIndex < rootKeyCount) {
        groupingKeys_.removeAt(keyIndex);
        applyState();
        return;
    }

    QVector<features::frame_browser::FrameGroupingKey>& scopedGroupingKeys =
        scopedGroupingKeysByScopeKey_[activeScopeKey_];
    const int scopedKeyIndex = keyIndex - rootKeyCount;
    if (scopedKeyIndex < 0 || scopedKeyIndex >= scopedGroupingKeys.size()) {
        return;
    }
    scopedGroupingKeys.removeAt(scopedKeyIndex);
    if (scopedGroupingKeys.isEmpty()) {
        scopedGroupingKeysByScopeKey_.remove(activeScopeKey_);
    }
    applyState();
}

void FrameBrowserController::reorderGroupingKeys(const QVector<int>& reorderedIndexes) {
    if (!activeScopeKey_.isEmpty()) {
        return;
    }

    if (reorderedIndexes.size() != groupingKeys_.size()) {
        return;
    }

    QVector<features::frame_browser::FrameGroupingKey> reorderedKeys;
    reorderedKeys.reserve(groupingKeys_.size());
    for (int sourceIndex : reorderedIndexes) {
        if (sourceIndex < 0 || sourceIndex >= groupingKeys_.size()) {
            return;
        }
        reorderedKeys.append(groupingKeys_.at(sourceIndex));
    }
    groupingKeys_ = reorderedKeys;
    applyState();
}

void FrameBrowserController::focusFrameByStartBit(qsizetype frameStartBit) {
    if (frameStartBit < 0) {
        return;
    }

    const int rowCount = static_cast<int>(frameLayout_.rowCount(dataSource_));
    for (int row = 0; row < rowCount; ++row) {
        if (frameLayout_.rowStartBit(dataSource_, row) != frameStartBit) {
            continue;
        }

        const int targetColumn = byteTableModel_.hasFrameLengthColumn() ? 1 : 0;
        byteTableView_.focusModelIndex(byteTableModel_.index(row, targetColumn));
        return;
    }
}

void FrameBrowserController::applyScopeForTreeIndex(const QModelIndex& treeIndex) {
    if (!treeIndex.isValid()) {
        return;
    }

    saveCurrentSplitState();
    activeScopePath_ = frameGroupTreeModel_.pathValuesForIndex(treeIndex);
    activeScopeKey_ = frameGroupTreeModel_.scopeKeyForIndex(treeIndex);
    const QVector<qsizetype> frameStartBits = frameGroupTreeModel_.frameStartBitsForIndex(treeIndex);
    frameGroupTreeModel_.setActiveScopePath(activeScopePath_);
    activeScopedFrameStartBits_.clear();
    for (qsizetype frameStartBit : frameStartBits) {
        activeScopedFrameStartBits_.insert(frameStartBit);
    }
    applyState();
}

bool FrameBrowserController::addFilterClauseForIndex(const QModelIndex& modelIndex) {
    if (!modelIndex.isValid() || !frameLayout_.isFramed()) {
        return false;
    }

    features::frame_browser::FrameFilterClause filterClause;
    if (byteTableModel_.isFrameLengthColumn(modelIndex.column())) {
        filterClause.key = {
            features::frame_browser::FrameGroupingKeyKind::FrameLength,
            QStringLiteral("Frame Length"),
            QStringLiteral("decimal"),
            0,
            0,
        };
    } else {
        const int visibleColumnIndex = byteTableModel_.visibleColumnIndexForModelIndex(modelIndex);
        if (!buildGroupingKeyForVisibleSeed(visibleColumnIndex, &filterClause.key)) {
            return false;
        }
    }

    const features::framing::FrameSpan currentFrameSpan{
        frameLayout_.rowStartBit(dataSource_, modelIndex.row()),
        frameLayout_.rowLengthBits(dataSource_, modelIndex.row()),
    };
    filterClause.value = features::frame_browser::evaluateFrameGroupValue(dataSource_, currentFrameSpan, filterClause.key);
    if (filterClause.value.missing) {
        return false;
    }

    filterClauses_.erase(
        std::remove_if(
            filterClauses_.begin(),
            filterClauses_.end(),
            [&](const features::frame_browser::FrameFilterClause& existingClause) {
                return existingClause.key == filterClause.key;
            }
        ),
        filterClauses_.end()
    );
    filterClauses_.append(filterClause);
    applyState();
    return true;
}

QString FrameBrowserController::summaryText() const {
    if (!dataSource_.hasData() || frameSpans_.isEmpty()) {
        return QStringLiteral("Frame grouping is available after framing is active.");
    }

    QStringList parts;
    parts.append(QStringLiteral("Frames: %1 of %2")
        .arg(frameLayout_.rowCount(dataSource_))
        .arg(frameSpans_.size()));
    const QVector<features::frame_browser::FrameGroupingKey> effectiveGroupingKeys =
        effectiveGroupingKeysForActiveScope();
    if (!effectiveGroupingKeys.isEmpty()) {
        QStringList groupingLabels;
        for (const features::frame_browser::FrameGroupingKey& groupingKey : effectiveGroupingKeys) {
            groupingLabels.append(groupingKey.label);
        }
        parts.append(QStringLiteral("Group: %1").arg(groupingLabels.join(QStringLiteral(" \u2192 "))));
    }
    if (!filterClauses_.isEmpty()) {
        QStringList filterLabels;
        for (const features::frame_browser::FrameFilterClause& filterClause : filterClauses_) {
            filterLabels.append(QStringLiteral("%1 = %2").arg(filterClause.key.label, filterClause.value.displayText));
        }
        parts.append(QStringLiteral("Filters: %1").arg(filterLabels.join(QStringLiteral(", "))));
    }
    if (!activeScopedFrameStartBits_.isEmpty()) {
        parts.append(QStringLiteral("Filtered"));
    }
    return parts.join(QStringLiteral(" | "));
}

QVector<features::frame_browser::FrameGroupingKey> FrameBrowserController::effectiveGroupingKeysForActiveScope() const {
    QVector<features::frame_browser::FrameGroupingKey> effectiveGroupingKeys = groupingKeys_;
    QString scopeKey = activeScopeKey_;
    QVector<QString> scopeChain;
    while (!scopeKey.isEmpty()) {
        scopeChain.prepend(scopeKey);
        scopeKey = features::frame_browser::parentScopeKey(scopeKey);
    }

    for (const QString& scopedKey : scopeChain) {
        effectiveGroupingKeys += scopedGroupingKeysByScopeKey_.value(scopedKey);
    }

    return effectiveGroupingKeys;
}

void FrameBrowserController::saveCurrentSplitState() {
    const QHash<int, models::SplitColumnState> currentSplitColumns = byteTableModel_.splitColumns();
    if (activeScopeKey_.isEmpty()) {
        rootSplitColumns_ = currentSplitColumns;
        return;
    }

    scopedSplitColumnsByScopeKey_.insert(activeScopeKey_, currentSplitColumns);
}

void FrameBrowserController::applySplitStateForActiveScope() {
    QHash<int, models::SplitColumnState> splitColumns = rootSplitColumns_;
    QString scopeKey = activeScopeKey_;
    while (!scopeKey.isEmpty()) {
        if (scopedSplitColumnsByScopeKey_.contains(scopeKey)) {
            splitColumns = scopedSplitColumnsByScopeKey_.value(scopeKey);
            break;
        }
        scopeKey = features::frame_browser::parentScopeKey(scopeKey);
    }

    byteTableModel_.setSplitColumns(splitColumns);
}

void FrameBrowserController::resetSplitScopeState() {
    rootSplitColumns_.clear();
    scopedSplitColumnsByScopeKey_.clear();
    activeScopePath_.clear();
    activeScopeKey_.clear();
    byteTableModel_.setSplitColumns({});
}

int FrameBrowserController::definitionIndexForVisibleColumns(const QSet<int>& visibleColumns) const {
    QList<int> sortedColumns = visibleColumns.values();
    if (sortedColumns.isEmpty()) {
        return -1;
    }

    std::optional<int> matchingDefinitionIndex;
    for (int visibleColumnIndex : sortedColumns) {
        const std::optional<int> definitionIndex = byteTableModel_.visibleDefinitionIndex(visibleColumnIndex);
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

void FrameBrowserController::scopeToFirstChildBranch(
    const QVector<features::frame_browser::FrameGroupValue>& parentPath
) {
    QModelIndex parentIndex;
    if (!parentPath.isEmpty()) {
        parentIndex = frameGroupTreeModel_.indexForPath(parentPath);
    }
    if (!parentIndex.isValid()) {
        parentIndex = frameGroupTreeModel_.index(0, 0);
    }
    if (!parentIndex.isValid()) {
        return;
    }

    const QModelIndex firstChildIndex = frameGroupTreeModel_.index(0, 0, parentIndex);
    if (firstChildIndex.isValid()) {
        applyScopeForTreeIndex(firstChildIndex);
    } else if (!parentPath.isEmpty()) {
        applyScopeForTreeIndex(parentIndex);
    }
}

}  // namespace bitabyte::ui
