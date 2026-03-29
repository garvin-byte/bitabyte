#pragma once

#include <QColor>
#include <QString>
#include <QStringList>

class QComboBox;

namespace bitabyte::features::columns {

struct ColumnColorOption {
    QString name;
    QColor color;
    bool hasColor = false;
};

[[nodiscard]] const QList<ColumnColorOption>& columnColorOptions();
[[nodiscard]] QStringList columnColorNames();
[[nodiscard]] QColor colorForName(const QString& colorName);
void populateColorCombo(QComboBox* comboBox, const QString& currentName = QStringLiteral("None"));

}  // namespace bitabyte::features::columns
