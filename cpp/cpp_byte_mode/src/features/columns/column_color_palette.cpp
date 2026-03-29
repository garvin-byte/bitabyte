#include "features/columns/column_color_palette.h"

#include <QComboBox>
#include <QIcon>
#include <QPixmap>

namespace bitabyte::features::columns {
namespace {

const QList<ColumnColorOption> kColumnColorOptions = {
    {QStringLiteral("None"), QColor(), false},
    {QStringLiteral("Sunshine"), QColor(QStringLiteral("#FFF4B5")), true},
    {QStringLiteral("Mint"), QColor(QStringLiteral("#C8FACC")), true},
    {QStringLiteral("Sky"), QColor(QStringLiteral("#B9DEFF")), true},
    {QStringLiteral("Coral"), QColor(QStringLiteral("#FFB3AB")), true},
    {QStringLiteral("Lilac"), QColor(QStringLiteral("#E0C6FF")), true},
    {QStringLiteral("Seafoam"), QColor(QStringLiteral("#BFFFE3")), true},
    {QStringLiteral("Rose"), QColor(QStringLiteral("#FFC7DA")), true},
    {QStringLiteral("Lavender"), QColor(QStringLiteral("#D7C0FF")), true},
    {QStringLiteral("Orange"), QColor(QStringLiteral("#FFD299")), true},
    {QStringLiteral("Teal"), QColor(QStringLiteral("#9FE3D7")), true},
    {QStringLiteral("Gold"), QColor(QStringLiteral("#FFE08A")), true},
    {QStringLiteral("Aqua"), QColor(QStringLiteral("#9ADBF2")), true},
    {QStringLiteral("Moss"), QColor(QStringLiteral("#C4E5A2")), true},
    {QStringLiteral("Plum"), QColor(QStringLiteral("#E3B0FF")), true},
    {QStringLiteral("Salmon"), QColor(QStringLiteral("#FFCDC1")), true},
    {QStringLiteral("Slate"), QColor(QStringLiteral("#C7D2E5")), true},
    {QStringLiteral("Peach"), QColor(QStringLiteral("#FFD8B2")), true},
    {QStringLiteral("Lime"), QColor(QStringLiteral("#DBFF95")), true},
    {QStringLiteral("Berry"), QColor(QStringLiteral("#FFB2CF")), true},
    {QStringLiteral("Ocean"), QColor(QStringLiteral("#A4C8FF")), true},
};

}  // namespace

const QList<ColumnColorOption>& columnColorOptions() {
    return kColumnColorOptions;
}

QStringList columnColorNames() {
    QStringList names;
    for (const ColumnColorOption& option : kColumnColorOptions) {
        names.append(option.name);
    }
    return names;
}

QColor colorForName(const QString& colorName) {
    for (const ColumnColorOption& option : kColumnColorOptions) {
        if (option.name == colorName && option.hasColor) {
            return option.color;
        }
    }

    return {};
}

void populateColorCombo(QComboBox* comboBox, const QString& currentName) {
    if (comboBox == nullptr) {
        return;
    }

    comboBox->clear();
    for (const ColumnColorOption& option : kColumnColorOptions) {
        comboBox->addItem(option.name);
        QPixmap colorSwatch(16, 16);
        colorSwatch.fill(option.hasColor ? option.color : QColor(240, 240, 240));
        comboBox->setItemData(comboBox->count() - 1, QIcon(colorSwatch), Qt::DecorationRole);
    }

    const int selectedIndex = comboBox->findText(currentName);
    comboBox->setCurrentIndex(selectedIndex >= 0 ? selectedIndex : 0);
}

}  // namespace bitabyte::features::columns
