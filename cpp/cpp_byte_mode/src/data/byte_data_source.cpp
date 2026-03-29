#include "data/byte_data_source.h"

#include <QFile>

#include <array>
#include <cstring>

namespace bitabyte::data {
namespace {

[[nodiscard]] const std::array<std::array<char, 8>, 256>& expandedBitsByByte() {
    static const auto table = [] {
        std::array<std::array<char, 8>, 256> table{};
        for (int byteValue = 0; byteValue < 256; ++byteValue) {
            for (int bitIndex = 0; bitIndex < 8; ++bitIndex) {
                table[byteValue][bitIndex] = static_cast<char>((byteValue >> (7 - bitIndex)) & 0x01);
            }
        }
        return table;
    }();
    return table;
}

void writeBitsFromByte(
    const std::array<std::array<char, 8>, 256>& bitTable,
    unsigned char byteValue,
    int startBitInByte,
    int bitCount,
    char* destination
) {
    if (bitCount <= 0 || destination == nullptr) {
        return;
    }

    const auto& expandedBits = bitTable.at(byteValue);
    std::memcpy(destination, expandedBits.data() + startBitInByte, static_cast<size_t>(bitCount));
}

}  // namespace

bool ByteDataSource::loadFile(const QString& filePath, QString* errorMessage) {
    QFile inputFile(filePath);
    if (!inputFile.open(QIODevice::ReadOnly)) {
        if (errorMessage != nullptr) {
            *errorMessage = inputFile.errorString();
        }
        return false;
    }

    byteBuffer_ = inputFile.readAll();
    sourceFilePath_ = filePath;
    return true;
}

void ByteDataSource::clear() {
    byteBuffer_.clear();
    sourceFilePath_.clear();
}

bool ByteDataSource::hasData() const {
    return !byteBuffer_.isEmpty();
}

qsizetype ByteDataSource::byteCount() const {
    return byteBuffer_.size();
}

qsizetype ByteDataSource::bitCount() const {
    return byteBuffer_.size() * 8;
}

int ByteDataSource::bytesPerRow() const {
    return bytesPerRow_;
}

qsizetype ByteDataSource::rowCount() const {
    if (byteBuffer_.isEmpty()) {
        return 0;
    }

    return (byteBuffer_.size() + bytesPerRow_ - 1) / bytesPerRow_;
}

QString ByteDataSource::sourceFilePath() const {
    return sourceFilePath_;
}

bool ByteDataSource::isByteOffsetValid(qsizetype byteOffset) const {
    return byteOffset >= 0 && byteOffset < byteBuffer_.size();
}

bool ByteDataSource::isBitOffsetValid(qsizetype bitOffset) const {
    return bitOffset >= 0 && bitOffset < bitCount();
}

const QByteArray& ByteDataSource::rawBytes() const {
    return byteBuffer_;
}

void ByteDataSource::setBytesPerRow(int bytesPerRow) {
    bytesPerRow_ = qMax(1, bytesPerRow);
}

unsigned char ByteDataSource::byteAt(qsizetype byteIndex) const {
    if (!isByteOffsetValid(byteIndex)) {
        return 0;
    }

    return static_cast<unsigned char>(byteBuffer_.at(byteIndex));
}

unsigned char ByteDataSource::bitAt(qsizetype bitIndex) const {
    if (!isBitOffsetValid(bitIndex)) {
        return 0;
    }

    const qsizetype byteOffset = bitIndex / 8;
    const int bitOffsetInByte = static_cast<int>(bitIndex % 8);
    const unsigned char byteValue = byteAt(byteOffset);
    return static_cast<unsigned char>((byteValue >> (7 - bitOffsetInByte)) & 0x01);
}

QByteArray ByteDataSource::bitRange(qsizetype startBit, qsizetype requestedBitCount) const {
    QByteArray bitValues;
    if (requestedBitCount <= 0 || !isBitOffsetValid(startBit)) {
        return bitValues;
    }

    const qsizetype availableBitCount = qMin(requestedBitCount, bitCount() - startBit);
    if (availableBitCount <= 0) {
        return bitValues;
    }

    bitValues.resize(static_cast<int>(availableBitCount));
    const auto& bitTable = expandedBitsByByte();
    char* destination = bitValues.data();
    qsizetype currentBit = startBit;
    qsizetype remainingBits = availableBitCount;

    const int leadingBitOffset = static_cast<int>(currentBit % 8);
    if (leadingBitOffset != 0) {
        const int leadingBitCount = qMin(static_cast<int>(remainingBits), 8 - leadingBitOffset);
        writeBitsFromByte(bitTable, byteAt(currentBit / 8), leadingBitOffset, leadingBitCount, destination);
        destination += leadingBitCount;
        currentBit += leadingBitCount;
        remainingBits -= leadingBitCount;
    }

    while (remainingBits >= 8) {
        const auto& expandedBits = bitTable.at(byteAt(currentBit / 8));
        std::memcpy(destination, expandedBits.data(), 8);
        destination += 8;
        currentBit += 8;
        remainingBits -= 8;
    }

    if (remainingBits > 0) {
        writeBitsFromByte(bitTable, byteAt(currentBit / 8), 0, static_cast<int>(remainingBits), destination);
    }

    return bitValues;
}

unsigned char ByteDataSource::byteValueAtBitOffset(qsizetype startBit) const {
    if (!isBitOffsetValid(startBit)) {
        return 0;
    }

    const qsizetype byteOffset = startBit / 8;
    const int bitOffsetInByte = static_cast<int>(startBit % 8);
    const unsigned char firstByte = byteAt(byteOffset);
    if (bitOffsetInByte == 0) {
        return firstByte;
    }

    const unsigned char secondByte = byteAt(byteOffset + 1);
    return static_cast<unsigned char>((firstByte << bitOffsetInByte) | (secondByte >> (8 - bitOffsetInByte)));
}

QByteArray ByteDataSource::byteRangeAtBitOffset(qsizetype startBit, int requestedByteCount) const {
    QByteArray byteValues;
    if (requestedByteCount <= 0 || !isBitOffsetValid(startBit)) {
        return byteValues;
    }

    const qsizetype availableByteCount = qMin<qsizetype>(
        requestedByteCount,
        (bitCount() - startBit) / 8
    );
    if (availableByteCount <= 0) {
        return byteValues;
    }

    byteValues.resize(static_cast<int>(availableByteCount));
    if ((startBit % 8) == 0) {
        std::memcpy(
            byteValues.data(),
            byteBuffer_.constData() + (startBit / 8),
            static_cast<size_t>(availableByteCount)
        );
        return byteValues;
    }

    for (qsizetype byteIndex = 0; byteIndex < availableByteCount; ++byteIndex) {
        byteValues[static_cast<int>(byteIndex)] = static_cast<char>(byteValueAtBitOffset(startBit + (byteIndex * 8)));
    }

    return byteValues;
}

}  // namespace bitabyte::data
