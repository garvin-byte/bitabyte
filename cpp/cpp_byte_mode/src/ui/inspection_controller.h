#pragma once

#include <QModelIndex>
#include <QObject>
#include <QSet>
#include <QVector>

#include <functional>

#include "data/byte_data_source.h"
#include "features/columns/byte_column_definition.h"
#include "features/framing/frame_layout.h"

template <typename T>
class QFutureWatcher;

class QButtonGroup;
class QLabel;
class QSpinBox;
class QTimer;

namespace bitabyte::models {
class ByteTableModel;
}

namespace bitabyte::features::inspector {
struct FieldSelection;
struct FieldInspectorAnalysis;
}

namespace bitabyte::ui {

class ByteTableView;
class FieldInspectorPanel;
class LiveBitViewerWidget;

class InspectionController final : public QObject {
public:
    struct SelectionCallbacks {
        std::function<QSet<int>()> selectedVisibleColumns;
        std::function<QSet<int>(int)> editableVisibleColumnsForSeed;
        std::function<int(const QSet<int>&)> definitionIndexForVisibleColumns;
        std::function<QModelIndex()> topSelectedDataIndex;
    };

    InspectionController(
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
        QObject* parent = nullptr
    );

    void scheduleLiveBitViewerRefresh();
    void refreshLiveBitViewer();
    void refreshFieldInspector();
    void updateSelectionStatus();

private:
    void scheduleFieldInspectorRefresh();
    void startFieldInspectorAnalysis(
        const features::inspector::FieldSelection& fieldSelection,
        int currentRow,
        quint64 requestId
    );
    [[nodiscard]] QString currentLiveBitViewerMode() const;

    data::ByteDataSource& dataSource_;
    features::framing::FrameLayout& frameLayout_;
    QVector<features::columns::ByteColumnDefinition>& columnDefinitions_;
    models::ByteTableModel& byteTableModel_;
    ByteTableView& byteTableView_;
    LiveBitViewerWidget& liveBitViewerWidget_;
    FieldInspectorPanel& fieldInspectorPanel_;
    QLabel& selectionInfoLabel_;
    QButtonGroup& liveBitViewerModeGroup_;
    QSpinBox& liveBitViewerSizeSpinBox_;
    SelectionCallbacks selectionCallbacks_;
    QTimer* liveBitViewerRefreshTimer_ = nullptr;
    QTimer* fieldInspectorRefreshTimer_ = nullptr;
    QFutureWatcher<features::inspector::FieldInspectorAnalysis>* fieldInspectorWatcher_ = nullptr;
    quint64 fieldInspectorRequestId_ = 0;
    quint64 activeFieldInspectorRequestId_ = 0;
};

}  // namespace bitabyte::ui
