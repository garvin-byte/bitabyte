#include "ui/column_definition_dialog.h"

#include "features/columns/column_color_palette.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QVBoxLayout>

namespace bitabyte::ui {

ColumnDefinitionDialog::ColumnDefinitionDialog(QWidget* parent, const QString& defaultColor)
    : QDialog(parent) {
    setWindowTitle(QStringLiteral("Add Column Definition"));
    setMinimumWidth(360);

    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    QFormLayout* formLayout = new QFormLayout();

    labelLineEdit_ = new QLineEdit(this);
    formLayout->addRow(QStringLiteral("Label:"), labelLineEdit_);

    unitComboBox_ = new QComboBox(this);
    unitComboBox_->addItem(QStringLiteral("Byte Range"), QStringLiteral("byte"));
    unitComboBox_->addItem(QStringLiteral("Bit Range"), QStringLiteral("bit"));
    formLayout->addRow(QStringLiteral("Unit:"), unitComboBox_);

    startValueLabel_ = new QLabel(QStringLiteral("Start byte:"), this);
    startByteSpinBox_ = new QSpinBox(this);
    startByteSpinBox_->setRange(0, 10'000'000);
    formLayout->addRow(startValueLabel_, startByteSpinBox_);

    endByteLabel_ = new QLabel(QStringLiteral("End byte:"), this);
    endByteSpinBox_ = new QSpinBox(this);
    endByteSpinBox_->setRange(0, 10'000'000);
    formLayout->addRow(endByteLabel_, endByteSpinBox_);

    totalBitsLabel_ = new QLabel(QStringLiteral("Total bits:"), this);
    totalBitsSpinBox_ = new QSpinBox(this);
    totalBitsSpinBox_->setRange(1, 512);
    totalBitsSpinBox_->setValue(8);
    formLayout->addRow(totalBitsLabel_, totalBitsSpinBox_);

    formatComboBox_ = new QComboBox(this);
    formatComboBox_->addItem(QStringLiteral("Hex"), QStringLiteral("hex"));
    formatComboBox_->addItem(QStringLiteral("Binary"), QStringLiteral("binary"));
    formatComboBox_->addItem(QStringLiteral("Decimal"), QStringLiteral("decimal"));
    formatComboBox_->addItem(QStringLiteral("ASCII"), QStringLiteral("ascii"));
    formLayout->addRow(QStringLiteral("Display format:"), formatComboBox_);

    colorComboBox_ = new QComboBox(this);
    features::columns::populateColorCombo(colorComboBox_, defaultColor);
    formLayout->addRow(QStringLiteral("Color:"), colorComboBox_);

    rootLayout->addLayout(formLayout);

    QDialogButtonBox* buttonBox = new QDialogButtonBox(
        QDialogButtonBox::StandardButton::Ok | QDialogButtonBox::StandardButton::Cancel,
        this
    );
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    rootLayout->addWidget(buttonBox);

    connect(unitComboBox_, &QComboBox::currentIndexChanged, this, [this](int) { syncUnitFields(); });
    syncUnitFields();
}

void ColumnDefinitionDialog::setDefinition(const features::columns::ByteColumnDefinition& definition) {
    labelLineEdit_->setText(definition.label);
    const int unitIndex = unitComboBox_->findData(definition.unit);
    unitComboBox_->setCurrentIndex(unitIndex >= 0 ? unitIndex : 0);
    startByteSpinBox_->setValue(definition.unit == QStringLiteral("bit") ? definition.startBit : definition.startByte);
    endByteSpinBox_->setValue(definition.endByte);
    totalBitsSpinBox_->setValue(qMax(1, definition.totalBits));

    const int formatIndex = formatComboBox_->findData(definition.displayFormat);
    formatComboBox_->setCurrentIndex(formatIndex >= 0 ? formatIndex : 0);

    const int colorIndex = colorComboBox_->findText(definition.colorName);
    colorComboBox_->setCurrentIndex(colorIndex >= 0 ? colorIndex : 0);
    syncUnitFields();
}

features::columns::ByteColumnDefinition ColumnDefinitionDialog::definition() const {
    features::columns::ByteColumnDefinition definition;
    definition.label = labelLineEdit_->text().trimmed();
    definition.unit = unitComboBox_->currentData().toString();
    if (definition.unit.isEmpty()) {
        definition.unit = QStringLiteral("byte");
    }
    if (definition.unit == QStringLiteral("bit")) {
        definition.startBit = startByteSpinBox_->value();
        definition.totalBits = qMax(1, totalBitsSpinBox_->value());
        definition.startByte = definition.startBit / 8;
        definition.endByte = (definition.startBit + definition.totalBits - 1) / 8;
    } else {
        definition.startByte = startByteSpinBox_->value();
        definition.endByte = qMax(definition.startByte, endByteSpinBox_->value());
        definition.startBit = definition.startByte * 8;
        definition.totalBits = definition.byteCount() * 8;
    }
    definition.displayFormat = formatComboBox_->currentData().toString();
    if (definition.displayFormat.isEmpty()) {
        definition.displayFormat = QStringLiteral("hex");
    }
    definition.colorName = colorComboBox_->currentText();
    if (definition.colorName.isEmpty()) {
        definition.colorName = QStringLiteral("None");
    }
    return definition;
}

void ColumnDefinitionDialog::syncUnitFields() {
    const bool isBitRange = unitComboBox_->currentData().toString() == QStringLiteral("bit");
    startValueLabel_->setText(isBitRange ? QStringLiteral("Start bit:") : QStringLiteral("Start byte:"));
    endByteLabel_->setVisible(!isBitRange);
    endByteSpinBox_->setVisible(!isBitRange);
    totalBitsLabel_->setVisible(isBitRange);
    totalBitsSpinBox_->setVisible(isBitRange);
}

}  // namespace bitabyte::ui
