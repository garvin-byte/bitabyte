#pragma once

#include <QHash>
#include <QModelIndex>
#include <QPoint>
#include <QSet>
#include <QString>
#include <QVector>

#include <functional>

#include "data/byte_data_source.h"
#include "features/columns/byte_column_definition.h"
#include "features/frame_browser/frame_grouping.h"
#include "features/framing/frame_layout.h"
#include "models/byte_table_model.h"

namespace bitabyte::models {
class FrameGroupTreeModel;
}

namespace bitabyte::ui {

class ByteTableView;
class FrameGroupingPanel;

class FrameBrowserController final {
public:
    struct Callbacks {
        std::function<void()> syncFramingControlsFromState;
        std::function<void()> resizeTableColumns;
        std::function<void()> updateLoadedFileState;
        std::function<void()> updateSelectionStatus;
        std::function<void()> refreshLiveBitViewer;
    };

    FrameBrowserController(
        data::ByteDataSource& dataSource,
        features::framing::FrameLayout& frameLayout,
        QVector<features::columns::ByteColumnDefinition>& columnDefinitions,
        models::ByteTableModel& byteTableModel,
        ByteTableView& byteTableView,
        models::FrameGroupTreeModel& frameGroupTreeModel,
        Callbacks callbacks
    );

    void setPanel(FrameGroupingPanel* frameGroupingPanel);

    void setFrameSpans(const QVector<features::framing::FrameSpan>& frameSpans);
    [[nodiscard]] const QVector<features::framing::FrameSpan>& frameSpans() const noexcept;

    void refreshPanel();
    void applyState();
    void clearState();
    void clearGrouping();
    void clearFilters();
    void clearScope();
    void showTableHeaderContextMenu(const QPoint& pos);
    bool buildGroupingKeyForVisibleSeed(
        int visibleColumnIndex,
        features::frame_browser::FrameGroupingKey* groupingKey
    ) const;
    bool buildGroupingKeyForVisibleColumns(
        const QSet<int>& visibleColumns,
        features::frame_browser::FrameGroupingKey* groupingKey
    ) const;
    [[nodiscard]] QVector<features::frame_browser::FrameGroupingKey> buildGroupingKeysForSelection(
        const QSet<int>& selectedVisibleColumns
    ) const;
    void addGroupingKeyFromVisibleSeed(int visibleColumnIndex, bool appendToGroupingStack);
    void applyGroupingKeysFromSelection(const QSet<int>& selectedVisibleColumns, bool appendToGroupingStack);
    void removeGroupingKey(int keyIndex);
    void reorderGroupingKeys(const QVector<int>& reorderedIndexes);
    void focusFrameByStartBit(qsizetype frameStartBit);
    void applyScopeForTreeIndex(const QModelIndex& treeIndex);
    bool addFilterClauseForIndex(const QModelIndex& modelIndex);
    [[nodiscard]] QString summaryText() const;
    [[nodiscard]] QVector<features::frame_browser::FrameGroupingKey> effectiveGroupingKeysForActiveScope() const;
    void saveCurrentSplitState();
    void applySplitStateForActiveScope();
    void resetSplitScopeState();

private:
    [[nodiscard]] int definitionIndexForVisibleColumns(const QSet<int>& visibleColumns) const;
    void scopeToFirstChildBranch(const QVector<features::frame_browser::FrameGroupValue>& parentPath);

    data::ByteDataSource& dataSource_;
    features::framing::FrameLayout& frameLayout_;
    QVector<features::columns::ByteColumnDefinition>& columnDefinitions_;
    models::ByteTableModel& byteTableModel_;
    ByteTableView& byteTableView_;
    models::FrameGroupTreeModel& frameGroupTreeModel_;
    FrameGroupingPanel* frameGroupingPanel_ = nullptr;
    Callbacks callbacks_;
    QVector<features::framing::FrameSpan> frameSpans_;
    QVector<features::frame_browser::FrameGroupingKey> groupingKeys_;
    QHash<QString, QVector<features::frame_browser::FrameGroupingKey>> scopedGroupingKeysByScopeKey_;
    QVector<features::frame_browser::FrameFilterClause> filterClauses_;
    QVector<features::frame_browser::FrameGroupValue> activeScopePath_;
    QString activeScopeKey_;
    QSet<qsizetype> activeScopedFrameStartBits_;
    QHash<int, models::SplitColumnState> rootSplitColumns_;
    QHash<QString, QHash<int, models::SplitColumnState>> scopedSplitColumnsByScopeKey_;
};

}  // namespace bitabyte::ui
