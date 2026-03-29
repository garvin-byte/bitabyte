#include "ui/frame_grouping_panel.h"

#include "models/frame_group_tree_model.h"

#include <QAbstractItemView>
#include <QItemSelectionModel>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMenu>
#include <QPushButton>
#include <QTimer>
#include <QTreeView>
#include <QVBoxLayout>

namespace bitabyte::ui {

FrameGroupingPanel::FrameGroupingPanel(QWidget* parent)
    : QWidget(parent) {
    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(8, 8, 8, 8);
    rootLayout->setSpacing(8);

    statusLabel_ = new QLabel(QStringLiteral("No framing active"), this);
    statusLabel_->setWordWrap(true);
    rootLayout->addWidget(statusLabel_);

    QHBoxLayout* keyButtonLayout = new QHBoxLayout();
    keyButtonLayout->setSpacing(6);
    removeKeyButton_ = new QPushButton(QStringLiteral("Remove Key"), this);
    clearGroupingButton_ = new QPushButton(QStringLiteral("Clear Grouping"), this);
    clearFiltersButton_ = new QPushButton(QStringLiteral("Clear Filters"), this);
    clearScopeButton_ = new QPushButton(QStringLiteral("Remove Filter"), this);
    keyButtonLayout->addWidget(removeKeyButton_);
    keyButtonLayout->addWidget(clearGroupingButton_);
    keyButtonLayout->addWidget(clearFiltersButton_);
    keyButtonLayout->addWidget(clearScopeButton_);
    keyButtonLayout->addStretch();
    rootLayout->addLayout(keyButtonLayout);

    groupingKeyListWidget_ = new QListWidget(this);
    groupingKeyListWidget_->setSelectionMode(QAbstractItemView::SingleSelection);
    groupingKeyListWidget_->setDragDropMode(QAbstractItemView::InternalMove);
    groupingKeyListWidget_->setDefaultDropAction(Qt::MoveAction);
    groupingKeyListWidget_->setFlow(QListView::LeftToRight);
    groupingKeyListWidget_->setWrapping(false);
    groupingKeyListWidget_->setMaximumHeight(60);
    rootLayout->addWidget(groupingKeyListWidget_);

    treeView_ = new QTreeView(this);
    treeView_->setHeaderHidden(true);
    treeView_->setUniformRowHeights(true);
    treeView_->setContextMenuPolicy(Qt::CustomContextMenu);
    treeView_->setStyleSheet(QStringLiteral(
        "QTreeView::item:selected:active { background-color: rgb(255, 244, 179); color: black; }"
        "QTreeView::item:selected:!active { background-color: rgb(255, 244, 179); color: black; }"
    ));
    rootLayout->addWidget(treeView_, 1);

    connect(groupingKeyListWidget_->model(), &QAbstractItemModel::rowsMoved, this, [this]() {
        QVector<int> reorderedIndexes;
        reorderedIndexes.reserve(groupingKeyListWidget_->count());
        for (int itemIndex = 0; itemIndex < groupingKeyListWidget_->count(); ++itemIndex) {
            reorderedIndexes.append(groupingKeyListWidget_->item(itemIndex)->data(Qt::UserRole).toInt());
        }
        emit groupingOrderChanged(reorderedIndexes);
        updateButtonState();
    });
    connect(groupingKeyListWidget_, &QListWidget::itemSelectionChanged, this, &FrameGroupingPanel::updateButtonState);
    connect(removeKeyButton_, &QPushButton::clicked, this, [this]() {
        if (groupingKeyListWidget_->currentRow() < 0) {
            return;
        }
        emit removeGroupingKeyRequested(groupingKeyListWidget_->currentRow());
    });
    connect(clearGroupingButton_, &QPushButton::clicked, this, &FrameGroupingPanel::clearGroupingRequested);
    connect(clearFiltersButton_, &QPushButton::clicked, this, &FrameGroupingPanel::clearFiltersRequested);
    connect(clearScopeButton_, &QPushButton::clicked, this, &FrameGroupingPanel::clearScopeRequested);
    connect(treeView_, &QTreeView::clicked, this, [this](const QModelIndex& index) {
        if (!index.isValid()) {
            return;
        }

        const auto* frameGroupTreeModel = qobject_cast<const models::FrameGroupTreeModel*>(treeView_->model());
        if (frameGroupTreeModel != nullptr && frameGroupTreeModel->isLeaf(index)) {
            emit leafActivated(index);
            return;
        }
        emit scopeRequested(index);
    });
    connect(treeView_, &QTreeView::doubleClicked, this, [this](const QModelIndex& index) {
        if (!index.isValid()) {
            return;
        }
        emit scopeRequested(index);
    });
    connect(treeView_, &QWidget::customContextMenuRequested, this, &FrameGroupingPanel::showTreeContextMenu);

    updateButtonState();
}

void FrameGroupingPanel::setGroupingKeys(
    const QVector<features::frame_browser::FrameGroupingKey>& groupingKeys
) {
    groupingKeyListWidget_->clear();
    for (int keyIndex = 0; keyIndex < groupingKeys.size(); ++keyIndex) {
        QListWidgetItem* item = new QListWidgetItem(groupingKeys.at(keyIndex).label, groupingKeyListWidget_);
        item->setData(Qt::UserRole, keyIndex);
        item->setToolTip(groupingKeys.at(keyIndex).label);
    }
    updateButtonState();
}

void FrameGroupingPanel::setTreeModel(models::FrameGroupTreeModel* frameGroupTreeModel) {
    treeView_->setModel(frameGroupTreeModel);
    if (frameGroupTreeModel != nullptr) {
        treeView_->expandToDepth(1);
    }
    updateButtonState();
}

void FrameGroupingPanel::setActiveScopePath(
    const QVector<features::frame_browser::FrameGroupValue>& activeScopePath
) {
    const auto* frameGroupTreeModel = qobject_cast<const models::FrameGroupTreeModel*>(treeView_->model());
    if (frameGroupTreeModel == nullptr || treeView_->selectionModel() == nullptr) {
        return;
    }

    const auto applySelection = [this, frameGroupTreeModel, activeScopePath]() {
        if (treeView_->selectionModel() == nullptr) {
            return;
        }

        const auto defaultIndex = [frameGroupTreeModel]() {
            const QModelIndex allFramesIndex = frameGroupTreeModel->index(0, 0);
            if (!allFramesIndex.isValid()) {
                return QModelIndex{};
            }
            return allFramesIndex;
        };

        const auto selectIndex = [this](const QModelIndex& targetIndex) {
            if (!targetIndex.isValid()) {
                treeView_->viewport()->update();
                return;
            }

            treeView_->setCurrentIndex(targetIndex);
            treeView_->selectionModel()->select(
                targetIndex,
                QItemSelectionModel::SelectionFlag::ClearAndSelect
                    | QItemSelectionModel::SelectionFlag::Current
            );
            treeView_->scrollTo(targetIndex, QAbstractItemView::ScrollHint::EnsureVisible);
            treeView_->viewport()->update();
        };

        treeView_->selectionModel()->clearSelection();
        treeView_->setCurrentIndex(QModelIndex());

        if (activeScopePath.isEmpty()) {
            selectIndex(defaultIndex());
            return;
        }

        const QModelIndex activeIndex = frameGroupTreeModel->indexForPath(activeScopePath);
        if (!activeIndex.isValid()) {
            selectIndex(defaultIndex());
            return;
        }

        QModelIndex parentIndex = activeIndex.parent();
        while (parentIndex.isValid()) {
            treeView_->expand(parentIndex);
            parentIndex = parentIndex.parent();
        }

        selectIndex(activeIndex);
    };

    applySelection();
    QTimer::singleShot(0, this, applySelection);
}

void FrameGroupingPanel::setStatusText(const QString& statusText) {
    statusLabel_->setText(statusText);
}

void FrameGroupingPanel::setFramingActive(bool framingActive) {
    groupingKeyListWidget_->setEnabled(framingActive);
    treeView_->setEnabled(framingActive);
    clearGroupingButton_->setEnabled(framingActive && groupingKeyListWidget_->count() > 0);
    clearFiltersButton_->setEnabled(framingActive);
    clearScopeButton_->setEnabled(framingActive);
    removeKeyButton_->setEnabled(framingActive && groupingKeyListWidget_->currentRow() >= 0);
}

void FrameGroupingPanel::updateButtonState() {
    const bool hasKeys = groupingKeyListWidget_->count() > 0;
    removeKeyButton_->setEnabled(groupingKeyListWidget_->isEnabled() && groupingKeyListWidget_->currentRow() >= 0);
    clearGroupingButton_->setEnabled(groupingKeyListWidget_->isEnabled() && hasKeys);
}

void FrameGroupingPanel::showTreeContextMenu(const QPoint& pos) {
    if (treeView_->model() == nullptr) {
        return;
    }

    const QModelIndex index = treeView_->indexAt(pos);
    if (!index.isValid()) {
        return;
    }

    const auto* frameGroupTreeModel = qobject_cast<const models::FrameGroupTreeModel*>(treeView_->model());
    if (frameGroupTreeModel == nullptr) {
        return;
    }

    QMenu menu(this);
    QAction* scopeAction = menu.addAction(QStringLiteral("Show Only These Frames"));
    QAction* focusAction = nullptr;
    if (frameGroupTreeModel->isLeaf(index)) {
        focusAction = menu.addAction(QStringLiteral("Go To Frame"));
    }
    menu.addSeparator();
    QAction* clearScopeAction = menu.addAction(QStringLiteral("Remove Filter"));

    QAction* chosenAction = menu.exec(treeView_->viewport()->mapToGlobal(pos));
    if (chosenAction == scopeAction) {
        emit scopeRequested(index);
    } else if (chosenAction == focusAction) {
        emit leafActivated(index);
    } else if (chosenAction == clearScopeAction) {
        emit clearScopeRequested();
    }
}

}  // namespace bitabyte::ui
