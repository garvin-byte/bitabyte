#pragma once

#include <QMainWindow>
#include <QHash>
#include <QSet>
#include <QVector>

#include "data/byte_data_source.h"
#include "features/columns/byte_column_definition.h"
#include "features/frame_browser/frame_grouping.h"
#include "features/framing/frame_layout.h"
#include "models/byte_table_model.h"

template <typename T>
class QFutureWatcher;

class QAction;
class QDockWidget;
class QLabel;
class QLineEdit;
class QModelIndex;
class QPoint;
class QPushButton;
class QSpinBox;
class QButtonGroup;
class QDragEnterEvent;
class QDropEvent;
class QTimer;

namespace bitabyte::models {
class ByteTableModel;
class FrameGroupTreeModel;
}

namespace bitabyte::features::inspector {
struct FieldSelection;
struct FieldInspectorAnalysis;
}

namespace bitabyte::ui {

class ByteTableView;
class ColumnDefinitionsPanel;
class FieldInspectorPanel;
class FrameGroupingPanel;
class LiveBitViewerWidget;

class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void openFile();
    void reloadFile();
    void exportCsv();
    void bytesPerRowChanged(int bytesPerRow);
    void applySyncFraming();
    void openBitstreamSyncDiscovery();
    void frameSelection();
    void clearFraming();
    void addColumnDefinition();
    void defineColumnFromSelection();
    void editColumnDefinition(int definitionIndex);
    void removeColumnDefinition(int definitionIndex);
    void splitSelectionAsBinary();
    void splitSelectionAsNibbles();
    void clearSelectionSplits();
    void combineSelection();
    void showTableContextMenu(const QPoint& pos);

protected:
    void dragEnterEvent(QDragEnterEvent* dragEnterEvent) override;
    void dropEvent(QDropEvent* dropEvent) override;

private:
    void buildMenus();
    void buildCentralWidget();
    void buildColumnDefinitionsDock();
    void buildLiveBitViewerDock();
    void applyInitialWindowLayout();
    void updateLoadedFileState();
    void updateSelectionStatus();
    void updateWindowTitle();
    void resizeTableColumns();
    void refreshFrameGroupingPanel();
    void applyFrameBrowserState();
    void clearFrameBrowserState();
    void clearFrameScope();
    void resetStateForFreshFile();
    void syncFramingControlsFromState();
    void refreshColumnDefinitionsPanel();
    void scheduleLiveBitViewerRefresh();
    void scheduleFieldInspectorRefresh();
    void refreshLiveBitViewer();
    void refreshFieldInspector();
    void startFieldInspectorAnalysis(
        const features::inspector::FieldSelection& fieldSelection,
        int currentRow,
        quint64 requestId
    );
    void validateFramingStateAfterLoad();
    void cycleFrameChronologicalOrder();
    void cycleFrameLengthOrder();
    void applyFrameRowOrder(features::framing::FrameLayout::RowOrderMode rowOrderMode, bool descending);
    void updateFrameRowOrderButtons();
    void showTableHeaderContextMenu(const QPoint& pos);
    bool applySyncFramingPattern(const QString& patternText, QString* errorMessage = nullptr);
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
    void scopeToFirstChildBranch(const QVector<features::frame_browser::FrameGroupValue>& parentPath);
    void removeGroupingKey(int keyIndex);
    void reorderGroupingKeys(const QVector<int>& reorderedIndexes);
    void focusFrameByStartBit(qsizetype frameStartBit);
    void applyScopeForTreeIndex(const QModelIndex& treeIndex);
    bool addFilterClauseForIndex(const QModelIndex& modelIndex);
    [[nodiscard]] QString frameBrowserSummaryText() const;
    [[nodiscard]] QVector<features::frame_browser::FrameGroupingKey> effectiveGroupingKeysForActiveScope() const;
    void saveCurrentSplitState();
    void applySplitStateForActiveScope();
    void resetSplitScopeState();
    void upsertSyncDefinitionForSelection(const QSet<int>& selectedVisibleColumns);
    void upsertSyncDefinition(int startBit, int totalBits, const QString& displayFormat);
    void editVisibleColumns(const QSet<int>& visibleColumns);
    void selectVisibleColumns(const QSet<int>& visibleColumns);
    bool buildDefinitionFromSelection(
        const QSet<int>& selectedVisibleColumns,
        features::columns::ByteColumnDefinition* definition,
        QString* errorMessage = nullptr
    ) const;
    bool combineSelectedVisibleColumns(QString* errorMessage = nullptr);
    [[nodiscard]] QString nextCombinedLabel(const QSet<int>& selectedVisibleColumns) const;
    [[nodiscard]] QString combinedDisplayFormat(const QSet<int>& selectedVisibleColumns, QString* errorMessage) const;
    [[nodiscard]] QString nextColumnColorName() const;
    [[nodiscard]] QSet<int> selectedVisibleColumns() const;
    [[nodiscard]] QSet<int> visibleColumnsForAbsoluteBitRange(int startBit, int endBit) const;
    [[nodiscard]] QSet<int> editableVisibleColumnsForSeed(int visibleColumnIndex) const;
    [[nodiscard]] QString currentLiveBitViewerMode() const;
    [[nodiscard]] QModelIndex topSelectedDataIndex() const;
    [[nodiscard]] bool extractSelectionPattern(QString* patternText, QString* errorMessage) const;
    [[nodiscard]] bool hasColumnSelection() const;
    [[nodiscard]] int definitionIndexForVisibleColumns(const QSet<int>& visibleColumns) const;
    [[nodiscard]] int selectedDefinitionIndexFromCurrentSelection() const;
    bool loadFilePath(const QString& filePath, bool resetStateForFreshFile = true);
    [[nodiscard]] QString fileSummaryText() const;

    data::ByteDataSource dataSource_;
    features::framing::FrameLayout frameLayout_;
    QVector<features::columns::ByteColumnDefinition> columnDefinitions_;
    QVector<features::framing::FrameSpan> allFrameSpans_;
    QVector<features::frame_browser::FrameGroupingKey> frameGroupingKeys_;
    QHash<QString, QVector<features::frame_browser::FrameGroupingKey>> scopedGroupingKeysByScopeKey_;
    QVector<features::frame_browser::FrameFilterClause> frameFilterClauses_;
    QVector<features::frame_browser::FrameGroupValue> activeScopePath_;
    QString activeScopeKey_;
    QSet<qsizetype> activeScopedFrameStartBits_;
    QHash<int, models::SplitColumnState> rootSplitColumns_;
    QHash<QString, QHash<int, models::SplitColumnState>> scopedSplitColumnsByScopeKey_;
    models::ByteTableModel* byteTableModel_ = nullptr;
    ByteTableView* byteTableView_ = nullptr;
    models::FrameGroupTreeModel* frameGroupTreeModel_ = nullptr;
    QDockWidget* columnDefinitionsDock_ = nullptr;
    ColumnDefinitionsPanel* columnDefinitionsPanel_ = nullptr;
    FrameGroupingPanel* frameGroupingPanel_ = nullptr;
    QDockWidget* liveBitViewerDock_ = nullptr;
    LiveBitViewerWidget* liveBitViewerWidget_ = nullptr;
    FieldInspectorPanel* fieldInspectorPanel_ = nullptr;
    QButtonGroup* liveBitViewerModeGroup_ = nullptr;
    QSpinBox* liveBitViewerSizeSpinBox_ = nullptr;
    QPushButton* frameChronologicalOrderButton_ = nullptr;
    QPushButton* frameLengthOrderButton_ = nullptr;
    QSpinBox* bytesPerRowSpinBox_ = nullptr;
    QLineEdit* syncPatternLineEdit_ = nullptr;
    QLabel* fileInfoLabel_ = nullptr;
    QLabel* frameBrowserInfoLabel_ = nullptr;
    QLabel* selectionInfoLabel_ = nullptr;
    QAction* openFileAction_ = nullptr;
    QAction* reloadFileAction_ = nullptr;
    QAction* exportCsvAction_ = nullptr;
    QAction* applyFramingAction_ = nullptr;
    QAction* bitstreamSyncDiscoveryAction_ = nullptr;
    QAction* frameSelectionAction_ = nullptr;
    QAction* clearFramingAction_ = nullptr;
    QAction* addColumnDefinitionAction_ = nullptr;
    QAction* defineSelectionColumnAction_ = nullptr;
    QAction* splitBinaryAction_ = nullptr;
    QAction* splitNibblesAction_ = nullptr;
    QAction* clearSelectionSplitsAction_ = nullptr;
    QAction* combineSelectionAction_ = nullptr;
    QAction* highlightConstantColumnsAction_ = nullptr;
    QTimer* liveBitViewerRefreshTimer_ = nullptr;
    QTimer* fieldInspectorRefreshTimer_ = nullptr;
    QFutureWatcher<features::inspector::FieldInspectorAnalysis>* fieldInspectorWatcher_ = nullptr;
    bool frameChronologicalDescending_ = false;
    bool frameLengthDescending_ = false;
    quint64 fieldInspectorRequestId_ = 0;
    quint64 activeFieldInspectorRequestId_ = 0;
};

}  // namespace bitabyte::ui
