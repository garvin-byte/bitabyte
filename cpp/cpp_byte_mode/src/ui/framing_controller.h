#pragma once

#include <QComboBox>
#include <QSet>
#include <QString>
#include <QVector>

#include <functional>
#include <optional>

#include "data/byte_data_source.h"
#include "features/classification/frame_field_classification.h"
#include "features/columns/byte_column_definition.h"
#include "features/framing/frame_layout.h"

class QLineEdit;
class QStatusBar;
class QSpinBox;
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
        std::function<bool()> isBitModeEnabled;
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
        QSpinBox& fixedFrameWidthSpinBox,
        QSpinBox& fixedFrameBitOffsetSpinBox,
        QComboBox& frameSortComboBox,
        QWidget& dialogParent,
        QStatusBar& statusBar,
        Callbacks callbacks
    );

    void resetState();
    void syncControlsFromState();
    void frameSelection();
    void applySyncFraming();
    void applyFixedFraming();
    bool frameByPattern(const QString& patternText, QString* errorMessage = nullptr);
    [[nodiscard]] bool fixedFramingControlsEditable() const;
    void openBitstreamSyncDiscovery();
    void clearFraming();
    void setFrameSortOption(int sortOptionIndex);

private:
    void applyFrameRowOrder(features::framing::FrameLayout::RowOrderMode rowOrderMode, bool descending);
    void updateFrameSortComboBox();
    bool applySyncFramingPattern(const QString& patternText, QString* errorMessage = nullptr);
    bool applyFixedFramingParameters(int frameWidthValue, int bitOffset, QString* errorMessage = nullptr);
    void upsertSyncDefinitionForSelection(const QSet<int>& selectedVisibleColumns);
    void upsertSyncDefinitionRecord(int startBit, int totalBits, const QString& displayFormat);
    void upsertSyncDefinition(int startBit, int totalBits, const QString& displayFormat);
    [[nodiscard]] std::optional<features::columns::ByteColumnDefinition> definitionForDetectedHint(
        const features::classification::FrameFieldHint& detectedHint
    ) const;
    int appendDetectedFieldDefinitions(
        const QVector<features::classification::FrameFieldHint>& detectedHints
    );

    enum class FramingSource {
        None,
        FixedWidth,
        Other,
    };

    data::ByteDataSource& dataSource_;
    features::framing::FrameLayout& frameLayout_;
    QVector<features::columns::ByteColumnDefinition>& columnDefinitions_;
    models::ByteTableModel& byteTableModel_;
    ByteTableView& byteTableView_;
    FrameBrowserController& frameBrowserController_;
    InspectionController& inspectionController_;
    QLineEdit& syncPatternLineEdit_;
    QSpinBox& fixedFrameWidthSpinBox_;
    QSpinBox& fixedFrameBitOffsetSpinBox_;
    QComboBox& frameSortComboBox_;
    QWidget& dialogParent_;
    QStatusBar& statusBar_;
    Callbacks callbacks_;
    FramingSource framingSource_ = FramingSource::None;
};

}  // namespace bitabyte::ui
