#pragma once

#include <QAbstractItemModel>
#include <QBitArray>
#include <QHash>
#include <QVector>

#include "features/frame_browser/frame_grouping.h"
#include "features/framing/frame_layout.h"

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::models {

class FrameGroupTreeModel final : public QAbstractItemModel {
    Q_OBJECT

public:
    explicit FrameGroupTreeModel(QObject* parent = nullptr);

    void setFrameSource(
        const data::ByteDataSource* dataSource,
        const QVector<features::framing::FrameSpan>* frameSpans
    );
    void setGroupingKeys(const QVector<features::frame_browser::FrameGroupingKey>& groupingKeys);
    void setScopedGroupingKeys(const QHash<QString, QVector<features::frame_browser::FrameGroupingKey>>& scopedGroupingKeys);
    void setFilterClauses(const QVector<features::frame_browser::FrameFilterClause>& filterClauses);
    void setActiveScopePath(const QVector<features::frame_browser::FrameGroupValue>& activeScopePath);

    [[nodiscard]] QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
    [[nodiscard]] QModelIndex parent(const QModelIndex& child) const override;
    [[nodiscard]] int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    [[nodiscard]] int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    [[nodiscard]] QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    [[nodiscard]] Qt::ItemFlags flags(const QModelIndex& index) const override;
    [[nodiscard]] bool hasChildren(const QModelIndex& parent = QModelIndex()) const override;

    [[nodiscard]] bool isLeaf(const QModelIndex& index) const;
    [[nodiscard]] qsizetype leafFrameStartBit(const QModelIndex& index) const;
    [[nodiscard]] QVector<qsizetype> frameStartBitsForIndex(const QModelIndex& index) const;
    [[nodiscard]] const QVector<features::frame_browser::FrameGroupValue>& activeScopePath() const noexcept;
    [[nodiscard]] QVector<features::frame_browser::FrameGroupValue> pathValuesForIndex(const QModelIndex& index) const;
    [[nodiscard]] QModelIndex indexForPath(const QVector<features::frame_browser::FrameGroupValue>& pathValues) const;
    [[nodiscard]] QString scopeKeyForIndex(const QModelIndex& index) const;
    [[nodiscard]] QString summaryForIndex(const QModelIndex& index) const;
    [[nodiscard]] int frameCountForIndex(const QModelIndex& index) const;

private:
    struct Entry {
        int frameIndex = -1;
        qsizetype frameStartBit = -1;
    };

    struct Node {
        int parentIndex = -1;
        QVector<int> childNodeIndexes;
        int level = -1;
        int entryBegin = 0;
        int entryEnd = 0;
        int frameIndex = -1;
        features::frame_browser::FrameGroupValue groupValue;
        QVector<features::frame_browser::FrameGroupValue> pathValues;
        QString scopeKey;
        bool isLeaf = false;
    };

    struct GroupValueCache {
        features::frame_browser::FrameGroupingKey key;
        QVector<features::frame_browser::FrameGroupValue> values;
        QBitArray valid;
    };

    void rebuild();
    int appendNode(Node node);
    void buildBranchChildren(
        int parentNodeIndex,
        const QVector<features::frame_browser::FrameGroupingKey>& inheritedGroupingKeys,
        int entryBegin,
        int entryEnd
    );
    [[nodiscard]] int valueCacheIndexForKey(const features::frame_browser::FrameGroupingKey& key);
    [[nodiscard]] const features::frame_browser::FrameGroupValue& cachedGroupValue(int frameIndex, int cacheIndex);
    [[nodiscard]] const Node* nodeForIndex(const QModelIndex& index) const;
    [[nodiscard]] Node* nodeForIndex(const QModelIndex& index);
    [[nodiscard]] QVector<features::frame_browser::FrameGroupValue> pathValuesForNode(const Node& node) const;
    [[nodiscard]] QString labelForNode(const Node& node) const;

    const data::ByteDataSource* dataSource_ = nullptr;
    const QVector<features::framing::FrameSpan>* frameSpans_ = nullptr;
    QVector<features::frame_browser::FrameGroupingKey> groupingKeys_;
    QHash<QString, QVector<features::frame_browser::FrameGroupingKey>> scopedGroupingKeys_;
    QVector<features::frame_browser::FrameFilterClause> filterClauses_;
    QVector<features::frame_browser::FrameGroupValue> activeScopePath_;
    QVector<Entry> entries_;
    QVector<Node> nodes_;
    QVector<GroupValueCache> groupValueCaches_;
};

}  // namespace bitabyte::models
