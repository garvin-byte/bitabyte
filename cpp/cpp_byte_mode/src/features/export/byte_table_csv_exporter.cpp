#include "features/export/byte_table_csv_exporter.h"

#include "core/byte_format_utils.h"
#include "data/byte_data_source.h"
#include "features/framing/frame_layout.h"

#include <QFile>
#include <QTextStream>

namespace bitabyte::features::exporting {

bool ByteTableCsvExporter::exportToFile(
    const data::ByteDataSource& dataSource,
    const framing::FrameLayout& frameLayout,
    const QString& filePath,
    QString* errorMessage
) {
    if (!dataSource.hasData()) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("No byte data is loaded.");
        }
        return false;
    }

    QFile outputFile(filePath);
    if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        if (errorMessage != nullptr) {
            *errorMessage = outputFile.errorString();
        }
        return false;
    }

    QTextStream outputStream(&outputFile);
    outputStream << "start_bit";
    if (frameLayout.isFramed()) {
        outputStream << ",frame_length_bytes";
    }
    for (int col = 0; col < frameLayout.columnCount(dataSource); ++col) {
        outputStream << ',' << core::formatByteHeaderLabel(col);
    }
    outputStream << '\n';

    for (qsizetype row = 0; row < frameLayout.rowCount(dataSource); ++row) {
        const qsizetype rowStartBit = frameLayout.rowStartBit(dataSource, static_cast<int>(row));
        outputStream << rowStartBit;
        if (frameLayout.isFramed()) {
            outputStream << ',' << frameLayout.rowLengthBytes(dataSource, static_cast<int>(row));
        }

        for (int col = 0; col < frameLayout.columnCount(dataSource); ++col) {
            const qsizetype startBit = frameLayout.cellStartBit(dataSource, static_cast<int>(row), col);
            outputStream << ',';

            if (frameLayout.hasDisplayByte(dataSource, static_cast<int>(row), col)) {
                outputStream << core::formatByteValue(dataSource.byteValueAtBitOffset(startBit));
            }
        }

        outputStream << '\n';
    }

    return true;
}

}  // namespace bitabyte::features::exporting
