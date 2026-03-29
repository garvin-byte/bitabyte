#pragma once

#include <QByteArray>
#include <QString>

namespace bitabyte::data {

class ByteDataSource {
public:
    ByteDataSource() = default;

    bool loadFile(const QString& filePath, QString* errorMessage = nullptr);
    void clear();

    [[nodiscard]] bool hasData() const;
    [[nodiscard]] qsizetype byteCount() const;
    [[nodiscard]] qsizetype bitCount() const;
    [[nodiscard]] int bytesPerRow() const;
    [[nodiscard]] qsizetype rowCount() const;
    [[nodiscard]] QString sourceFilePath() const;
    [[nodiscard]] bool isByteOffsetValid(qsizetype byteOffset) const;
    [[nodiscard]] bool isBitOffsetValid(qsizetype bitOffset) const;
    [[nodiscard]] const QByteArray& rawBytes() const;

    void setBytesPerRow(int bytesPerRow);
    [[nodiscard]] unsigned char byteAt(qsizetype byteIndex) const;
    [[nodiscard]] unsigned char bitAt(qsizetype bitIndex) const;
    [[nodiscard]] QByteArray bitRange(qsizetype startBit, qsizetype bitCount) const;
    [[nodiscard]] unsigned char byteValueAtBitOffset(qsizetype startBit) const;
    [[nodiscard]] QByteArray byteRangeAtBitOffset(qsizetype startBit, int byteCount) const;

private:
    QByteArray byteBuffer_;
    QString sourceFilePath_;
    int bytesPerRow_ = 16;
};

}  // namespace bitabyte::data
