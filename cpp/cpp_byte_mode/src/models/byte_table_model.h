#pragma once

#include <QAbstractTableModel>
#include <QFont>
#include <QHash>
#include <QPair>
#include <QSet>
#include <QVector>

#include <optional>

#include "features/columns/byte_column_definition.h"
#include "features/columns/visible_byte_column.h"

template <typename T>
class QFutureWatcher;

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::features::framing {
class FrameLayout;
}

namespace bitabyte::models {

struct SplitSegment {
    int startBit = 0;
    int endBit = 7;
    QString displayFormat = QStringLiteral("binary");
    QString label;
    QString colorName = QStringLiteral("None");

    [[nodiscard]] bool operator==(const SplitSegment& other) const noexcept = default;
};

struct SplitColumnState {
    QString type;
    QString label;
    QString colorName = QStringLiteral("None");
    QVector<SplitSegment> segments;
    int nextBinaryLabel = 0;
    int nextNibbleLabel = 0;

    [[nodiscard]] bool operator==(const SplitColumnState& other) const noexcept = default;
};

struct SplitPanelEntry {
    int payloadByteIndex = -1;
    int startBit = 0;
    int endBit = 0;
    QString label;
    QString displayFormat = QStringLiteral("binary");
    QString colorName = QStringLiteral("None");
};

class ByteTableModel final : public QAbstractTableModel {
    Q_OBJECT

public:
    explicit ByteTableModel(
        data::ByteDataSource* dataSource,
        features::framing::FrameLayout* frameLayout,
        const QVector<features::columns::ByteColumnDefinition>* columnDefinitions,
        QObject* parent = nullptr
    );

    [[nodiscard]] int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    [[nodiscard]] int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    [[nodiscard]] QVariant data(const QModelIndex& modelIndex, int role = Qt::DisplayRole) const override;
    [[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
    [[nodiscard]] Qt::ItemFlags flags(const QModelIndex& modelIndex) const override;

    [[nodiscard]] qsizetype displayStartBitForIndex(const QModelIndex& modelIndex) const;
    [[nodiscard]] bool hasFrameLengthColumn() const;
    [[nodiscard]] bool isFrameLengthColumn(int col) const;
    [[nodiscard]] int visibleColumnIndexForModelIndex(const QModelIndex& modelIndex) const;
    [[nodiscard]] int visibleColumnCount() const;
    [[nodiscard]] features::columns::VisibleByteColumn visibleByteColumn(int visibleColumn) const;
    [[nodiscard]] bool hasSplit(int byteIndex) const;
    [[nodiscard]] bool canSplitByte(int byteIndex) const;
    [[nodiscard]] bool applySplit(const QSet<int>& byteIndices, const QString& splitType);
    [[nodiscard]] bool clearSplits(const QSet<int>& byteIndices);
    [[nodiscard]] bool removeSplitBitRange(int startBit, int endBit);
    [[nodiscard]] std::optional<int> visibleDefinitionIndex(int visibleColumn) const;
    [[nodiscard]] QVector<SplitPanelEntry> visibleSplitEntries() const;
    [[nodiscard]] QString topHeaderLabelForVisibleColumn(int visibleColumn) const;
    [[nodiscard]] QString bottomHeaderLabelForVisibleColumn(int visibleColumn) const;
    [[nodiscard]] QSet<int> plainByteTargetsForVisibleColumns(const QSet<int>& visibleColumns) const;
    [[nodiscard]] QSet<int> splitByteTargetsForVisibleColumns(const QSet<int>& visibleColumns) const;
    [[nodiscard]] bool selectionCanBecomeNibble(const QSet<int>& visibleColumns) const;
    [[nodiscard]] bool selectionCanBecomeBinary(const QSet<int>& visibleColumns) const;
    [[nodiscard]] bool convertSelectionToNibble(const QSet<int>& visibleColumns);
    [[nodiscard]] bool convertSelectionToBinary(const QSet<int>& visibleColumns);
    [[nodiscard]] bool visibleColumnsAreContiguous(const QSet<int>& visibleColumns) const;
    [[nodiscard]] bool constantColumnHighlightEnabled() const;
    [[nodiscard]] QHash<int, SplitColumnState> splitColumns() const;

    void reload();
    void setConstantColumnHighlightEnabled(bool highlightEnabled);
    void setCounterHighlightedVisibleColumns(const QSet<int>& counterVisibleColumns);
    void setDetectedFieldDefinitions(const QVector<features::columns::ByteColumnDefinition>& detectedFieldDefinitions);
    void setSplitColumns(const QHash<int, SplitColumnState>& splitColumns);

private:
    struct SingleByteSelection {
        int byteIndex = -1;
        int startBit = 0;
        int endBit = 0;
        int bitWidth = 0;
    };

    [[nodiscard]] int visibleColumnIndexForModelColumn(int col) const;
    [[nodiscard]] const features::columns::ByteColumnDefinition* definitionAtIndex(int definitionIndex) const;
    [[nodiscard]] std::optional<int> definitionIndexForVisibleColumn(
        const features::columns::VisibleByteColumn& visibleColumn
    ) const;
    [[nodiscard]] QString formatVisibleColumnDisplayText(
        int row,
        const features::columns::VisibleByteColumn& visibleColumn
    ) const;
    [[nodiscard]] QString formatVisibleColumnToolTip(
        int row,
        qsizetype startBit,
        const features::columns::VisibleByteColumn& visibleColumn
    ) const;
    [[nodiscard]] QColor backgroundForVisibleColumn(
        int visibleColumnIndex,
        const features::columns::VisibleByteColumn& visibleColumn
    ) const;
    [[nodiscard]] bool hasDisplayValue(int row, const features::columns::VisibleByteColumn& visibleColumn) const;
    [[nodiscard]] QVector<SplitSegment> splitSegmentsForByte(int byteIndex) const;
    [[nodiscard]] int nextSplitLabelIndex(const QString& prefix) const;
    [[nodiscard]] QString nextAutoSplitColorName(
        const QString& preferredColor,
        const QSet<QString>& excludedColors = {}
    ) const;
    [[nodiscard]] QSet<int> plainByteSplitTargets(const QSet<int>& visibleColumns) const;
    [[nodiscard]] std::optional<QPair<int, int>> definitionBitSpan(
        const features::columns::ByteColumnDefinition& definition
    ) const;
    [[nodiscard]] features::columns::VisibleByteColumn buildVisibleDefinitionColumn(
        const features::columns::ByteColumnDefinition& definition,
        int definitionIndex
    ) const;
    [[nodiscard]] QString formatRangeLabel(int startValue, int endValue) const;
    void rebuildVisibleColumns();
    void rebuildByteColumnIndexMap();
    [[nodiscard]] qsizetype cellStartBit(int row, int col) const;
    [[nodiscard]] qsizetype rowStartBit(int row) const;
    [[nodiscard]] qsizetype rowLengthBits(int row) const;
    [[nodiscard]] QByteArray rowBitsSlice(int row, int startRelativeBit, int bitCount) const;
    [[nodiscard]] QByteArray rowBitsForVisibleColumn(
        int row,
        const features::columns::VisibleByteColumn& visibleColumn
    ) const;
    [[nodiscard]] QByteArray rowBytesSlice(int row, int startByte, int byteCount) const;
    [[nodiscard]] QString formatRowHeaderText(int row) const;
    [[nodiscard]] QString formatBytes(const QByteArray& rowBytes, const QString& displayFormat) const;
    [[nodiscard]] QString formatBits(const QByteArray& rowBits, const QString& displayFormat) const;
    [[nodiscard]] int displayWidthBytes() const;
    [[nodiscard]] std::optional<SingleByteSelection> singleByteSelection(
        const QSet<int>& visibleColumns
    ) const;
    [[nodiscard]] QString splitGroupLabel(const QVector<SplitSegment>& splitSegments) const;
    [[nodiscard]] QString splitGroupColorName(const QVector<SplitSegment>& splitSegments) const;
    [[nodiscard]] QVector<SplitSegment> currentSplitSegments(int byteIndex) const;
    [[nodiscard]] QVector<SplitSegment> replacementSegments(
        const QVector<SplitSegment>& sourceSegments,
        int targetStartBit,
        int targetEndBit,
        const QVector<SplitSegment>& replacementSegments
    ) const;
    void applyCustomSplitSegments(
        int byteIndex,
        const SplitColumnState& splitState,
        const QVector<SplitSegment>& segments,
        int nextBinaryLabel,
        int nextNibbleLabel
    );
    void invalidateConstantVisibleColumnCache(bool notifyDataChanged);
    void scheduleConstantVisibleColumnCacheRebuild();
    void startConstantVisibleColumnCacheRebuild();

    data::ByteDataSource* dataSource_ = nullptr;
    features::framing::FrameLayout* frameLayout_ = nullptr;
    const QVector<features::columns::ByteColumnDefinition>* columnDefinitions_ = nullptr;
    QVector<features::columns::ByteColumnDefinition> detectedFieldDefinitions_;
    QVector<features::columns::ByteColumnDefinition> effectiveColumnDefinitions_;
    QHash<int, SplitColumnState> splitColumns_;
    QVector<features::columns::VisibleByteColumn> visibleColumns_;
    QVector<int> byteColumnStarts_;
    QVector<int> byteColumnCounts_;
    QFont monospaceFont_;
    QFutureWatcher<QVector<int>>* constantHighlightWatcher_ = nullptr;
    QSet<int> constantVisibleColumnIndices_;
    QSet<int> counterVisibleColumnIndices_;
    bool constantColumnHighlightEnabled_ = true;
    quint64 constantHighlightRequestId_ = 0;
    quint64 activeConstantHighlightRequestId_ = 0;
};

}  // namespace bitabyte::models
