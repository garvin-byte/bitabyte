#pragma once

#include <QSet>
#include <QString>
#include <QVector>

#include <functional>

#include "data/byte_data_source.h"
#include "features/columns/byte_column_definition.h"
#include "features/framing/frame_layout.h"

class QLineEdit;
class QPushButton;
class QStatusBar;
class QWidget;

namespace bitabyte::models {
class ByteTableModel;
}

namespace bitabyte::ui {

class ByteTableView;
class FrameBrowserController;
class InspectionController;

class FramingController final {
public:
    struct Callbacks {
        std::function<QSet<int>()> selectedVisibleColumns;
        std::function<bool(QString*, QString*)> extractSelectionPattern;
        std::function<bool(const QSet<int>&, features::columns::ByteColumnDefinition*, QString*)>
            buildDefinitionFromSelection;
        std::function<void()> resizeTableColumns;
        std::function<void()> updateLoadedFileState;
        std::function<void()> refreshColumnDefinitionsPanel;
    };

    FramingController(
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
    );

    void resetState();
    void syncControlsFromState();
    void frameSelection();
    void applySyncFraming();
    void openBitstreamSyncDiscovery();
    void clearFraming();
    void cycleFrameChronologicalOrder();
    void cycleFrameLengthOrder();

private:
    void applyFrameRowOrder(features::framing::FrameLayout::RowOrderMode rowOrderMode, bool descending);
    void updateFrameRowOrderButtons();
    bool applySyncFramingPattern(const QString& patternText, QString* errorMessage = nullptr);
    void upsertSyncDefinitionForSelection(const QSet<int>& selectedVisibleColumns);
    void upsertSyncDefinition(int startBit, int totalBits, const QString& displayFormat);

    data::ByteDataSource& dataSource_;
    features::framing::FrameLayout& frameLayout_;
    QVector<features::columns::ByteColumnDefinition>& columnDefinitions_;
    models::ByteTableModel& byteTableModel_;
    ByteTableView& byteTableView_;
    FrameBrowserController& frameBrowserController_;
    InspectionController& inspectionController_;
    QLineEdit& syncPatternLineEdit_;
    QPushButton& frameChronologicalOrderButton_;
    QPushButton& frameLengthOrderButton_;
    QWidget& dialogParent_;
    QStatusBar& statusBar_;
    Callbacks callbacks_;
    bool frameChronologicalDescending_ = false;
    bool frameLengthDescending_ = false;
};

}  // namespace bitabyte::ui
