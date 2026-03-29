#pragma once

#include <QVector>
#include <QtTypes>

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::features::framing {

struct FrameSpan {
    qsizetype startBit = 0;
    qsizetype lengthBits = 0;
};

class FrameLayout {
public:
    enum class RowOrderMode {
        Chronological,
        Length,
    };

    [[nodiscard]] bool isFramed() const;
    [[nodiscard]] int frameMaxLengthBytes() const;
    [[nodiscard]] RowOrderMode rowOrderMode() const;
    [[nodiscard]] bool rowOrderDescending() const;

    void setFrames(const QVector<FrameSpan>& frameSpans);
    void clearFrame();
    void setRowOrder(RowOrderMode rowOrderMode, bool descending);

    [[nodiscard]] bool isValidForDataSource(const data::ByteDataSource& dataSource) const;
    [[nodiscard]] qsizetype rowCount(const data::ByteDataSource& dataSource) const;
    [[nodiscard]] int columnCount(const data::ByteDataSource& dataSource) const;
    [[nodiscard]] qsizetype rowStartBit(const data::ByteDataSource& dataSource, int row) const;
    [[nodiscard]] qsizetype rowLengthBits(const data::ByteDataSource& dataSource, int row) const;
    [[nodiscard]] int rowLengthBytes(const data::ByteDataSource& dataSource, int row) const;
    [[nodiscard]] qsizetype cellStartBit(const data::ByteDataSource& dataSource, int row, int col) const;
    [[nodiscard]] bool hasDisplayByte(const data::ByteDataSource& dataSource, int row, int col) const;

private:
    void rebuildFrameMaxLengthBytes();
    void rebuildDisplayOrder();
    [[nodiscard]] const FrameSpan* frameSpanForDisplayRow(int row) const;

    bool framedMode_ = false;
    QVector<FrameSpan> frameSpans_;
    QVector<int> displayRowOrder_;
    int frameMaxLengthBytes_ = 0;
    RowOrderMode rowOrderMode_ = RowOrderMode::Chronological;
    bool rowOrderDescending_ = false;
};

}  // namespace bitabyte::features::framing
