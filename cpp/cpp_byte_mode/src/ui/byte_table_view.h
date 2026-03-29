#pragma once

#include <QTableView>

namespace bitabyte::ui {

class ByteTableView final : public QTableView {
    Q_OBJECT

public:
    explicit ByteTableView(QWidget* parent = nullptr);

    void applyVisibleColumnSizing(int columnCount, bool hasFrameLengthColumn);
    void focusModelIndex(const QModelIndex& modelIndex);
};

}  // namespace bitabyte::ui
