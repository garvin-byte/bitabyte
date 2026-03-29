#pragma once

#include <QDialog>

#include "features/columns/byte_column_definition.h"

class QComboBox;
class QLineEdit;
class QLabel;
class QSpinBox;

namespace bitabyte::ui {

class ColumnDefinitionDialog final : public QDialog {
    Q_OBJECT

public:
    explicit ColumnDefinitionDialog(QWidget* parent = nullptr, const QString& defaultColor = QStringLiteral("Sky"));

    void setDefinition(const features::columns::ByteColumnDefinition& definition);
    [[nodiscard]] features::columns::ByteColumnDefinition definition() const;

private:
    void syncUnitFields();

    QLineEdit* labelLineEdit_ = nullptr;
    QComboBox* unitComboBox_ = nullptr;
    QLabel* startValueLabel_ = nullptr;
    QSpinBox* startByteSpinBox_ = nullptr;
    QLabel* endByteLabel_ = nullptr;
    QSpinBox* endByteSpinBox_ = nullptr;
    QLabel* totalBitsLabel_ = nullptr;
    QSpinBox* totalBitsSpinBox_ = nullptr;
    QComboBox* formatComboBox_ = nullptr;
    QComboBox* colorComboBox_ = nullptr;
};

}  // namespace bitabyte::ui
