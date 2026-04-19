#include "models/frame_group_tree_model.h"

#include "data/byte_data_source.h"

#include <QColor>
#include <QFont>

#include <algorithm>

namespace bitabyte::models {
namespace {

using bitabyte::features::frame_browser::appendScopeKey;
using bitabyte::features::frame_browser::compareFrameGroupValues;
using bitabyte::features::frame_browser::evaluateFrameGroupValue;

}  // namespace

FrameGroupTreeModel::FrameGroupTreeModel(QObject* parent)
    : QAbstractItemModel(parent) {
    nodes_.append(Node{});
}

void FrameGroupTreeModel::setFrameSource(
    const data::ByteDataSource* dataSource,
    const QVector<features::framing::FrameSpan>* frameSpans
) {
    dataSource_ = dataSource;
    frameSpans_ = frameSpans;
    rebuild();
}

void FrameGroupTreeModel::setGroupingKeys(
    const QVector<features::frame_browser::FrameGroupingKey>& groupingKeys
) {
    groupingKeys_ = groupingKeys;
    rebuild();
}

void FrameGroupTreeModel::setScopedGroupingKeys(
    const QHash<QString, QVector<features::frame_browser::FrameGroupingKey>>& scopedGroupingKeys
) {
    scopedGroupingKeys_ = scopedGroupingKeys;
    rebuild();
}

void FrameGroupTreeModel::setFilterClauses(
    const QVector<features::frame_browser::FrameFilterClause>& filterClauses
) {
    filterClauses_ = filterClauses;
    rebuild();
}

void FrameGroupTreeModel::setActiveScopePath(
    const QVector<features::frame_browser::FrameGroupValue>& activeScopePath
) {
    if (activeScopePath_ == activeScopePath) {
        return;
    }

    activeScopePath_ = activeScopePath;
    if (rowCount() <= 0) {
        return;
    }

    emit dataChanged(
        index(0, 0),
        index(rowCount() - 1, 0),
        {Qt::BackgroundRole, Qt::FontRole}
    );
}

QModelIndex FrameGroupTreeModel::index(int row, int column, const QModelIndex& parent) const {
    if (column != 0 || row < 0) {
        return {};
    }

    const Node* parentNode = nodeForIndex(parent);
    if (parentNode == nullptr || row >= parentNode->childNodeIndexes.size()) {
        return {};
    }

    const int nodeIndex = parentNode->childNodeIndexes.at(row);
    return createIndex(row, column, nodeIndex);
}

QModelIndex FrameGroupTreeModel::parent(const QModelIndex& child) const {
    const Node* childNode = nodeForIndex(child);
    if (childNode == nullptr || childNode->parentIndex <= 0) {
        return {};
    }

    const Node& parentNode = nodes_.at(childNode->parentIndex);
    if (parentNode.parentIndex < 0) {
        return createIndex(0, 0, childNode->parentIndex);
    }
    const int row = nodes_.at(parentNode.parentIndex).childNodeIndexes.indexOf(childNode->parentIndex);
    if (row < 0) {
        return {};
    }
    return createIndex(row, 0, childNode->parentIndex);
}

int FrameGroupTreeModel::rowCount(const QModelIndex& parent) const {
    const Node* parentNode = nodeForIndex(parent);
    return parentNode == nullptr ? 0 : parentNode->childNodeIndexes.size();
}

int FrameGroupTreeModel::columnCount(const QModelIndex& parent) const {
    Q_UNUSED(parent);
    return 1;
}

QVariant FrameGroupTreeModel::data(const QModelIndex& index, int role) const {
    const Node* node = nodeForIndex(index);
    if (node == nullptr) {
        return {};
    }

    if (role == Qt::DisplayRole) {
        return labelForNode(*node);
    }

    const bool isActiveScopeNode =
        !node->isLeaf
        && !activeScopePath_.isEmpty()
        && pathValuesForNode(*node) == activeScopePath_;
    const int allFramesNodeIndex =
        !nodes_.isEmpty() && !nodes_.first().childNodeIndexes.isEmpty()
            ? nodes_.first().childNodeIndexes.first()
            : -1;
    const int defaultSelectionNodeIndex = activeScopePath_.isEmpty() ? allFramesNodeIndex : -1;
    const bool isDefaultSelectionNode =
        !node->isLeaf
        && activeScopePath_.isEmpty()
        && defaultSelectionNodeIndex == index.internalId();

    if (role == Qt::BackgroundRole && (isActiveScopeNode || isDefaultSelectionNode)) {
        return QColor(255, 244, 179);
    }

    if (role == Qt::FontRole && (isActiveScopeNode || isDefaultSelectionNode)) {
        QFont boldFont;
        boldFont.setBold(true);
        return boldFont;
    }

    return {};
}

Qt::ItemFlags FrameGroupTreeModel::flags(const QModelIndex& index) const {
    if (!index.isValid()) {
        return Qt::NoItemFlags;
    }
    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

bool FrameGroupTreeModel::hasChildren(const QModelIndex& parent) const {
    const Node* parentNode = nodeForIndex(parent);
    return parentNode != nullptr && !parentNode->childNodeIndexes.isEmpty();
}

bool FrameGroupTreeModel::isLeaf(const QModelIndex& index) const {
    const Node* node = nodeForIndex(index);
    return node != nullptr && node->isLeaf;
}

qsizetype FrameGroupTreeModel::leafFrameStartBit(const QModelIndex& index) const {
    const Node* node = nodeForIndex(index);
    if (node == nullptr || !node->isLeaf || node->frameIndex < 0 || frameSpans_ == nullptr
        || node->frameIndex >= frameSpans_->size()) {
        return -1;
    }
    return frameSpans_->at(node->frameIndex).startBit;
}

QVector<qsizetype> FrameGroupTreeModel::frameStartBitsForIndex(const QModelIndex& index) const {
    QVector<qsizetype> frameStartBits;
    const Node* node = nodeForIndex(index);
    if (node == nullptr) {
        return frameStartBits;
    }

    frameStartBits.reserve(qMax(0, node->entryEnd - node->entryBegin));
    for (int entryIndex = node->entryBegin; entryIndex < node->entryEnd && entryIndex < entries_.size(); ++entryIndex) {
        frameStartBits.append(entries_.at(entryIndex).frameStartBit);
    }
    return frameStartBits;
}

const QVector<features::frame_browser::FrameGroupValue>& FrameGroupTreeModel::activeScopePath() const noexcept {
    return activeScopePath_;
}

QVector<features::frame_browser::FrameGroupValue> FrameGroupTreeModel::pathValuesForIndex(const QModelIndex& index) const {
    const Node* node = nodeForIndex(index);
    if (node == nullptr || node->isLeaf) {
        return {};
    }
    return pathValuesForNode(*node);
}

QModelIndex FrameGroupTreeModel::indexForPath(const QVector<features::frame_browser::FrameGroupValue>& pathValues) const {
    if (pathValues.isEmpty()) {
        return {};
    }

    for (int nodeIndex = 1; nodeIndex < nodes_.size(); ++nodeIndex) {
        const Node& node = nodes_.at(nodeIndex);
        if (node.isLeaf) {
            continue;
        }
        if (pathValuesForNode(node) != pathValues) {
            continue;
        }

        const int row = node.parentIndex >= 0
            ? nodes_.at(node.parentIndex).childNodeIndexes.indexOf(nodeIndex)
            : 0;
        if (row < 0) {
            return {};
        }
        return createIndex(row, 0, nodeIndex);
    }

    return {};
}

QString FrameGroupTreeModel::scopeKeyForIndex(const QModelIndex& index) const {
    const Node* node = nodeForIndex(index);
    if (node == nullptr || node->isLeaf) {
        return {};
    }
    return node->scopeKey;
}

QString FrameGroupTreeModel::summaryForIndex(const QModelIndex& index) const {
    const Node* node = nodeForIndex(index);
    if (node == nullptr) {
        return {};
    }

    if (node->isLeaf) {
        return QStringLiteral("Frame @ bit %1").arg(leafFrameStartBit(index));
    }

    return labelForNode(*node);
}

int FrameGroupTreeModel::frameCountForIndex(const QModelIndex& index) const {
    const Node* node = nodeForIndex(index);
    if (node == nullptr) {
        return 0;
    }
    return qMax(0, node->entryEnd - node->entryBegin);
}

void FrameGroupTreeModel::rebuild() {
    beginResetModel();

    entries_.clear();
    nodes_.clear();
    groupValueCaches_.clear();
    nodes_.append(Node{});

    if (dataSource_ == nullptr || frameSpans_ == nullptr || !dataSource_->hasData()) {
        endResetModel();
        return;
    }

    QVector<int> filterCacheIndices;
    filterCacheIndices.reserve(filterClauses_.size());
    for (const features::frame_browser::FrameFilterClause& filterClause : filterClauses_) {
        filterCacheIndices.append(valueCacheIndexForKey(filterClause.key));
    }

    entries_.reserve(frameSpans_->size());
    for (int frameIndex = 0; frameIndex < frameSpans_->size(); ++frameIndex) {
        bool includeFrame = true;
        for (int filterIndex = 0; filterIndex < filterClauses_.size(); ++filterIndex) {
            if (cachedGroupValue(frameIndex, filterCacheIndices.at(filterIndex))
                != filterClauses_.at(filterIndex).value) {
                includeFrame = false;
                break;
            }
        }
        if (!includeFrame) {
            continue;
        }

        Entry entry;
        entry.frameIndex = frameIndex;
        entry.frameStartBit = frameSpans_->at(frameIndex).startBit;
        entries_.append(entry);
    }

    std::stable_sort(entries_.begin(), entries_.end(), [](const Entry& leftEntry, const Entry& rightEntry) {
        return leftEntry.frameStartBit < rightEntry.frameStartBit;
    });

    if (!entries_.isEmpty()) {
        Node allFramesNode;
        allFramesNode.parentIndex = 0;
        allFramesNode.level = -1;
        allFramesNode.entryBegin = 0;
        allFramesNode.entryEnd = entries_.size();
        const int allFramesNodeIndex = appendNode(allFramesNode);
        nodes_[0].childNodeIndexes.append(allFramesNodeIndex);
        buildBranchChildren(allFramesNodeIndex, groupingKeys_, 0, entries_.size());
    }

    endResetModel();
}

int FrameGroupTreeModel::appendNode(Node node) {
    const int nodeIndex = nodes_.size();
    nodes_.append(std::move(node));
    return nodeIndex;
}

int FrameGroupTreeModel::valueCacheIndexForKey(const features::frame_browser::FrameGroupingKey& key) {
    for (int cacheIndex = 0; cacheIndex < groupValueCaches_.size(); ++cacheIndex) {
        if (groupValueCaches_.at(cacheIndex).key == key) {
            return cacheIndex;
        }
    }

    GroupValueCache cache;
    cache.key = key;
    if (frameSpans_ != nullptr) {
        cache.values.resize(frameSpans_->size());
        cache.valid = QBitArray(frameSpans_->size(), false);
    }
    groupValueCaches_.append(std::move(cache));
    return groupValueCaches_.size() - 1;
}

const features::frame_browser::FrameGroupValue& FrameGroupTreeModel::cachedGroupValue(int frameIndex, int cacheIndex) {
    GroupValueCache& cache = groupValueCaches_[cacheIndex];
    if (!cache.valid.testBit(frameIndex)) {
        cache.values[frameIndex] = evaluateFrameGroupValue(*dataSource_, frameSpans_->at(frameIndex), cache.key);
        cache.valid.setBit(frameIndex, true);
    }
    return cache.values.at(frameIndex);
}

void FrameGroupTreeModel::buildBranchChildren(
    int parentNodeIndex,
    const QVector<features::frame_browser::FrameGroupingKey>& inheritedGroupingKeys,
    int entryBegin,
    int entryEnd
) {
    QVector<features::frame_browser::FrameGroupingKey> availableGroupingKeys = inheritedGroupingKeys;
    availableGroupingKeys += scopedGroupingKeys_.value(nodes_.at(parentNodeIndex).scopeKey);

    if (availableGroupingKeys.isEmpty()) {
        return;
    }

    const features::frame_browser::FrameGroupingKey currentGroupingKey = availableGroupingKeys.first();
    const QVector<features::frame_browser::FrameGroupingKey> remainingGroupingKeys = availableGroupingKeys.mid(1);
    const int currentGroupingKeyCacheIndex = valueCacheIndexForKey(currentGroupingKey);

    QHash<int, features::frame_browser::FrameGroupValue> groupValuesByFrameIndex;
    groupValuesByFrameIndex.reserve(qMax(0, entryEnd - entryBegin));
    for (int entryIndex = entryBegin; entryIndex < entryEnd; ++entryIndex) {
        const Entry& entry = entries_.at(entryIndex);
        groupValuesByFrameIndex.insert(
            entry.frameIndex,
            cachedGroupValue(entry.frameIndex, currentGroupingKeyCacheIndex)
        );
    }

    std::stable_sort(
        entries_.begin() + entryBegin,
        entries_.begin() + entryEnd,
        [&](const Entry& leftEntry, const Entry& rightEntry) {
            const int valueCompare = compareFrameGroupValues(
                groupValuesByFrameIndex.value(leftEntry.frameIndex),
                groupValuesByFrameIndex.value(rightEntry.frameIndex)
            );
            if (valueCompare != 0) {
                return valueCompare < 0;
            }
            return leftEntry.frameStartBit < rightEntry.frameStartBit;
        }
    );

    int rangeBegin = entryBegin;
    while (rangeBegin < entryEnd) {
        int rangeEnd = rangeBegin + 1;
        while (rangeEnd < entryEnd
               && groupValuesByFrameIndex.value(entries_.at(rangeEnd).frameIndex)
                    == groupValuesByFrameIndex.value(entries_.at(rangeBegin).frameIndex)) {
            ++rangeEnd;
        }

        Node branchNode;
        branchNode.parentIndex = parentNodeIndex;
        branchNode.level = nodes_.at(parentNodeIndex).level + 1;
        branchNode.entryBegin = rangeBegin;
        branchNode.entryEnd = rangeEnd;
        branchNode.groupValue = groupValuesByFrameIndex.value(entries_.at(rangeBegin).frameIndex);
        branchNode.pathValues = nodes_.at(parentNodeIndex).pathValues;
        branchNode.pathValues.append(branchNode.groupValue);
        branchNode.scopeKey = appendScopeKey(nodes_.at(parentNodeIndex).scopeKey, currentGroupingKey, branchNode.groupValue);
        const int branchNodeIndex = appendNode(branchNode);
        nodes_[parentNodeIndex].childNodeIndexes.append(branchNodeIndex);
        buildBranchChildren(branchNodeIndex, remainingGroupingKeys, rangeBegin, rangeEnd);
        rangeBegin = rangeEnd;
    }
}

const FrameGroupTreeModel::Node* FrameGroupTreeModel::nodeForIndex(const QModelIndex& index) const {
    if (!index.isValid()) {
        return nodes_.isEmpty() ? nullptr : &nodes_.first();
    }

    const int nodeIndex = index.internalId();
    if (nodeIndex < 0 || nodeIndex >= nodes_.size()) {
        return nullptr;
    }
    return &nodes_.at(nodeIndex);
}

FrameGroupTreeModel::Node* FrameGroupTreeModel::nodeForIndex(const QModelIndex& index) {
    return const_cast<Node*>(std::as_const(*this).nodeForIndex(index));
}

QVector<features::frame_browser::FrameGroupValue> FrameGroupTreeModel::pathValuesForNode(const Node& node) const {
    return node.pathValues;
}

QString FrameGroupTreeModel::labelForNode(const Node& node) const {
    if (node.isLeaf) {
        const int rowNumber = node.frameIndex >= 0 ? node.frameIndex : 0;
        const features::framing::FrameSpan* frameSpan =
            frameSpans_ != nullptr && node.frameIndex >= 0 && node.frameIndex < frameSpans_->size()
            ? &frameSpans_->at(node.frameIndex)
            : nullptr;
        const int frameLengthBytes = frameSpan != nullptr ? static_cast<int>((frameSpan->lengthBits + 7) / 8) : 0;
        return QStringLiteral("Frame %1 (%2 bytes)")
            .arg(rowNumber)
            .arg(frameLengthBytes);
    }

    const int frameCount = qMax(0, node.entryEnd - node.entryBegin);
    if (node.parentIndex == 0 && node.level < 0) {
        return QStringLiteral("All Frames (%1)").arg(frameCount);
    }

    const QString valueLabel = node.groupValue.displayText.isEmpty()
        ? QStringLiteral("(missing)")
        : node.groupValue.displayText;
    return QStringLiteral("%1 (%2)")
        .arg(valueLabel)
        .arg(frameCount);
}

}  // namespace bitabyte::models
