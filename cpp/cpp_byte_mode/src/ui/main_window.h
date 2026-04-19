#pragma once

#include <QMainWindow>
#include <QSet>
#include <QVector>

#include <memory>

#include "data/byte_data_source.h"
#include "features/columns/byte_column_definition.h"
#include "features/framing/frame_layout.h"

class QAction;
class QComboBox;
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
class QResizeEvent;

namespace bitabyte::models {
class ByteTableModel;
class FrameGroupTreeModel;
}

namespace bitabyte::ui {

class ByteTableView;
class ColumnDefinitionsPanel;
class FieldCurrentValuePanel;
class FieldInspectorPanel;
class FrameFieldHintsPanel;
class FrameBrowserController;
class FrameGroupingPanel;
class FramingController;
class InspectionController;
class LiveBitViewerWidget;

class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void openFile();
    void reloadFile();
    void exportCsv();
    void bytesPerRowChanged(int bytesPerRow);
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
    void resizeEvent(QResizeEvent* resizeEvent) override;

private:
    void buildMenus();
    void buildCentralWidget();
    void buildColumnDefinitionsDock();
    void buildLiveBitViewerDock();
    void openPatternSearchDialog();
    void applyInitialWindowLayout();
    void updateLoadedFileState();
    void updateWindowTitle();
    void resizeTableColumns();
    void syncFrameWidthControls();
    void applyRawLayoutControls();
    void resetStateForFreshFile();
    void refreshColumnDefinitionsPanel();
    void validateFramingStateAfterLoad();
    void editVisibleColumns(const QSet<int>& visibleColumns);
    void focusPatternMatch(qsizetype startBit, qsizetype bitCount);
    void selectVisibleColumns(const QSet<int>& visibleColumns);
    void selectVisibleColumnsAtRow(const QSet<int>& visibleColumns, int row);
    void addColumnDefinitionFromHint(int startBit, int endBit, bool isConstant, const QString& label, const QString& valueText);
    void setBitModeEnabled(bool enabled);
    void syncTableBitDisplayMode();
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
    models::ByteTableModel* byteTableModel_ = nullptr;
    ByteTableView* byteTableView_ = nullptr;
    models::FrameGroupTreeModel* frameGroupTreeModel_ = nullptr;
    std::unique_ptr<FrameBrowserController> frameBrowserController_;
    std::unique_ptr<FramingController> framingController_;
    std::unique_ptr<InspectionController> inspectionController_;
    QDockWidget* columnDefinitionsDock_ = nullptr;
    ColumnDefinitionsPanel* columnDefinitionsPanel_ = nullptr;
    FrameGroupingPanel* frameGroupingPanel_ = nullptr;
    QDockWidget* liveBitViewerDock_ = nullptr;
    LiveBitViewerWidget* liveBitViewerWidget_ = nullptr;
    FrameFieldHintsPanel* frameFieldHintsPanel_ = nullptr;
    FieldInspectorPanel* fieldInspectorPanel_ = nullptr;
    FieldCurrentValuePanel* fieldCurrentValuePanel_ = nullptr;
    QButtonGroup* tableDisplayModeGroup_ = nullptr;
    QButtonGroup* liveBitViewerModeGroup_ = nullptr;
    QSpinBox* tableBitSizeSpinBox_ = nullptr;
    QSpinBox* liveBitViewerSizeSpinBox_ = nullptr;
    QComboBox* frameSortComboBox_ = nullptr;
    QPushButton* setFrameButton_ = nullptr;
    QSpinBox* frameWidthSpinBox_ = nullptr;
    QSpinBox* frameBitOffsetSpinBox_ = nullptr;
    QLabel* frameWidthLabel_ = nullptr;
    QLineEdit* syncPatternLineEdit_ = nullptr;
    QLabel* fileInfoLabel_ = nullptr;
    QLabel* frameBrowserInfoLabel_ = nullptr;
    QLabel* selectionInfoLabel_ = nullptr;
    QAction* openFileAction_ = nullptr;
    QAction* reloadFileAction_ = nullptr;
    QAction* exportCsvAction_ = nullptr;
    QAction* applyFramingAction_ = nullptr;
    QAction* applyFixedFramingAction_ = nullptr;
    QAction* bitstreamSyncDiscoveryAction_ = nullptr;
    QAction* patternSearchAction_ = nullptr;
    QAction* frameSelectionAction_ = nullptr;
    QAction* clearFramingAction_ = nullptr;
    QAction* addColumnDefinitionAction_ = nullptr;
    QAction* defineSelectionColumnAction_ = nullptr;
    QAction* splitBinaryAction_ = nullptr;
    QAction* splitNibblesAction_ = nullptr;
    QAction* clearSelectionSplitsAction_ = nullptr;
    QAction* combineSelectionAction_ = nullptr;
    QAction* highlightConstantColumnsAction_ = nullptr;
};

}  // namespace bitabyte::ui
