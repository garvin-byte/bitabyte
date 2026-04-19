#include "features/framing/frame_layout.h"

#include "data/byte_data_source.h"

#include <QtGlobal>

#include <algorithm>

namespace bitabyte::features::framing {

bool FrameLayout::isFramed() const {
    return framedMode_;
}

int FrameLayout::frameMaxLengthBytes() const {
    return frameMaxLengthBytes_;
}

qsizetype FrameLayout::frameMaxLengthBits() const {
    return frameMaxLengthBits_;
}

FrameLayout::RowOrderMode FrameLayout::rowOrderMode() const {
    return rowOrderMode_;
}

bool FrameLayout::rowOrderDescending() const {
    return rowOrderDescending_;
}

qsizetype FrameLayout::rawRowWidthBits() const {
    return rawRowWidthBits_;
}

qsizetype FrameLayout::rawStartBitOffset() const {
    return rawStartBitOffset_;
}

bool FrameLayout::padFramedBitDisplayToByteBoundary() const {
    return padFramedBitDisplayToByteBoundary_;
}

void FrameLayout::setFrames(const QVector<FrameSpan>& frameSpans) {
    framedMode_ = true;
    frameSpans_ = frameSpans;
    rebuildFrameMaxLengthBytes();
    rebuildDisplayOrder();
}

void FrameLayout::clearFrame() {
    framedMode_ = false;
    frameSpans_.clear();
    displayRowOrder_.clear();
    frameMaxLengthBytes_ = 0;
    frameMaxLengthBits_ = 0;
    padFramedBitDisplayToByteBoundary_ = false;
}

void FrameLayout::setRowOrder(RowOrderMode rowOrderMode, bool descending) {
    rowOrderMode_ = rowOrderMode;
    rowOrderDescending_ = descending;
    rebuildDisplayOrder();
}

void FrameLayout::setRawLayout(qsizetype rowWidthBits, qsizetype startBitOffset) {
    rawRowWidthBits_ = qMax<qsizetype>(1, rowWidthBits);
    rawStartBitOffset_ = qMax<qsizetype>(0, startBitOffset);
}

void FrameLayout::setPadFramedBitDisplayToByteBoundary(bool enabled) {
    padFramedBitDisplayToByteBoundary_ = enabled;
}

bool FrameLayout::isValidForDataSource(const data::ByteDataSource& dataSource) const {
    if (!isFramed()) {
        return rawRowWidthBits_ > 0
            && rawStartBitOffset_ >= 0
            && rawStartBitOffset_ <= dataSource.bitCount();
    }

    for (const FrameSpan& frameSpan : frameSpans_) {
        if (frameSpan.lengthBits <= 0) {
            return false;
        }
        if (!dataSource.isBitOffsetValid(frameSpan.startBit)) {
            return false;
        }
    }

    return true;
}

qsizetype FrameLayout::rowCount(const data::ByteDataSource& dataSource) const {
    if (!dataSource.hasData()) {
        return 0;
    }

    if (!framedMode_) {
        if (rawRowWidthBits_ <= 0 || rawStartBitOffset_ >= dataSource.bitCount()) {
            return 0;
        }

        const qsizetype addressableBits = dataSource.bitCount() - rawStartBitOffset_;
        return (addressableBits + rawRowWidthBits_ - 1) / rawRowWidthBits_;
    }

    return frameSpans_.size();
}

int FrameLayout::columnCount(const data::ByteDataSource& dataSource) const {
    if (framedMode_) {
        return frameMaxLengthBytes_;
    }

    return static_cast<int>((rawRowWidthBits_ + 7) / 8);
}

qsizetype FrameLayout::rowStartBit(const data::ByteDataSource& dataSource, int row) const {
    if (row < 0 || !dataSource.hasData()) {
        return -1;
    }

    if (framedMode_) {
        const FrameSpan* frameSpan = frameSpanForDisplayRow(row);
        if (frameSpan == nullptr) {
            return -1;
        }
        return frameSpan->startBit;
    }

    return rawStartBitOffset_ + (static_cast<qsizetype>(row) * rawRowWidthBits_);
}

qsizetype FrameLayout::rowLengthBits(const data::ByteDataSource& dataSource, int row) const {
    if (row < 0 || !dataSource.hasData()) {
        return 0;
    }

    if (framedMode_) {
        const FrameSpan* frameSpan = frameSpanForDisplayRow(row);
        if (frameSpan == nullptr) {
            return 0;
        }
        return frameSpan->lengthBits;
    }

    const qsizetype startBit = rowStartBit(dataSource, row);
    if (startBit < 0 || startBit >= dataSource.bitCount()) {
        return 0;
    }

    return qMin(rawRowWidthBits_, dataSource.bitCount() - startBit);
}

int FrameLayout::rowLengthBytes(const data::ByteDataSource& dataSource, int row) const {
    const qsizetype lengthBits = rowLengthBits(dataSource, row);
    if (lengthBits <= 0) {
        return 0;
    }

    return static_cast<int>((lengthBits + 7) / 8);
}

qsizetype FrameLayout::cellStartBit(const data::ByteDataSource& dataSource, int row, int col) const {
    if (row < 0 || col < 0) {
        return -1;
    }

    return rowStartBit(dataSource, row) + static_cast<qsizetype>(col) * 8;
}

bool FrameLayout::hasDisplayByte(const data::ByteDataSource& dataSource, int row, int col) const {
    if (row < 0 || col < 0) {
        return false;
    }

    const qsizetype lengthBits = rowLengthBits(dataSource, row);
    if (lengthBits <= 0) {
        return false;
    }

    const qsizetype byteStartBit = static_cast<qsizetype>(col) * 8;
    return byteStartBit < lengthBits;
}

void FrameLayout::rebuildFrameMaxLengthBytes() {
    frameMaxLengthBytes_ = 0;
    frameMaxLengthBits_ = 0;
    for (const FrameSpan& frameSpan : frameSpans_) {
        const int frameLengthBytes = static_cast<int>((frameSpan.lengthBits + 7) / 8);
        frameMaxLengthBytes_ = qMax(frameMaxLengthBytes_, frameLengthBytes);
        frameMaxLengthBits_ = qMax(frameMaxLengthBits_, frameSpan.lengthBits);
    }
}

void FrameLayout::rebuildDisplayOrder() {
    displayRowOrder_.clear();
    displayRowOrder_.reserve(frameSpans_.size());
    for (int frameIndex = 0; frameIndex < frameSpans_.size(); ++frameIndex) {
        displayRowOrder_.append(frameIndex);
    }

    if (!framedMode_) {
        return;
    }

    std::stable_sort(
        displayRowOrder_.begin(),
        displayRowOrder_.end(),
        [this](int leftIndex, int rightIndex) {
            if (leftIndex < 0 || leftIndex >= frameSpans_.size() || rightIndex < 0 || rightIndex >= frameSpans_.size()) {
                return leftIndex < rightIndex;
            }

            const FrameSpan& leftFrameSpan = frameSpans_.at(leftIndex);
            const FrameSpan& rightFrameSpan = frameSpans_.at(rightIndex);

            if (rowOrderMode_ == RowOrderMode::Length && leftFrameSpan.lengthBits != rightFrameSpan.lengthBits) {
                return rowOrderDescending_
                    ? leftFrameSpan.lengthBits > rightFrameSpan.lengthBits
                    : leftFrameSpan.lengthBits < rightFrameSpan.lengthBits;
            }

            return rowOrderDescending_ ? leftIndex > rightIndex : leftIndex < rightIndex;
        }
    );
}

const FrameSpan* FrameLayout::frameSpanForDisplayRow(int row) const {
    if (row < 0 || row >= displayRowOrder_.size()) {
        return nullptr;
    }

    const int frameIndex = displayRowOrder_.at(row);
    if (frameIndex < 0 || frameIndex >= frameSpans_.size()) {
        return nullptr;
    }

    return &frameSpans_.at(frameIndex);
}

}  // namespace bitabyte::features::framing
