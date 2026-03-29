#pragma once

#include <QString>

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::features::framing {
class FrameLayout;
}

namespace bitabyte::features::exporting {

class ByteTableCsvExporter {
public:
    static bool exportToFile(
        const data::ByteDataSource& dataSource,
        const framing::FrameLayout& frameLayout,
        const QString& filePath,
        QString* errorMessage = nullptr
    );
};

}  // namespace bitabyte::features::exporting
