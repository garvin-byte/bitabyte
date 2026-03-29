#pragma once

#include <QWidget>

#include "features/frame_browser/frame_grouping.h"

class QLabel;
class QListWidget;
class QPoint;
class QPushButton;
class QModelIndex;
class QTreeView;

namespace bitabyte::models {
class FrameGroupTreeModel;
}

namespace bitabyte::ui {

class FrameGroupingPanel final : public QWidget {
    Q_OBJECT

public:
    explicit FrameGroupingPanel(QWidget* parent = nullptr);

    void setGroupingKeys(const QVector<features::frame_browser::FrameGroupingKey>& groupingKeys);
    void setTreeModel(models::FrameGroupTreeModel* frameGroupTreeModel);
    void setActiveScopePath(const QVector<features::frame_browser::FrameGroupValue>& activeScopePath);
    void setStatusText(const QString& statusText);
    void setFramingActive(bool framingActive);

signals:
    void groupingOrderChanged(const QVector<int>& reorderedIndexes);
    void removeGroupingKeyRequested(int keyIndex);
    void clearGroupingRequested();
    void clearFiltersRequested();
    void scopeRequested(const QModelIndex& treeIndex);
    void leafActivated(const QModelIndex& treeIndex);
    void clearScopeRequested();

private:
    void updateButtonState();
    void showTreeContextMenu(const QPoint& pos);

    QLabel* statusLabel_ = nullptr;
    QListWidget* groupingKeyListWidget_ = nullptr;
    QPushButton* removeKeyButton_ = nullptr;
    QPushButton* clearGroupingButton_ = nullptr;
    QPushButton* clearFiltersButton_ = nullptr;
    QPushButton* clearScopeButton_ = nullptr;
    QTreeView* treeView_ = nullptr;
};

}  // namespace bitabyte::ui
