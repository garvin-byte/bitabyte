#include "ui/field_current_value_panel.h"

#include <QAbstractButton>
#include <QButtonGroup>
#include <QFont>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QRadioButton>
#include <QVBoxLayout>

namespace bitabyte::ui {
namespace {

QLabel* createValueLabel(QWidget* parent) {
    QLabel* label = new QLabel(parent);
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    label->setMinimumWidth(0);
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    QFont mono = label->font();
    mono.setFamilies({
        QStringLiteral("Consolas"),
        QStringLiteral("Courier New"),
        QStringLiteral("Monospace"),
    });
    mono.setStyleHint(QFont::StyleHint::Monospace);
    label->setFont(mono);
    return label;
}

}  // namespace

FieldCurrentValuePanel::FieldCurrentValuePanel(QWidget* parent)
    : QWidget(parent) {
    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(6);

    QHBoxLayout* radioLayout = new QHBoxLayout();
    radioLayout->setSpacing(12);
    byteOrderGroup_ = new QButtonGroup(this);
    QRadioButton* msbfButton = new QRadioButton(QStringLiteral("MSBF"), this);
    QRadioButton* lsbfButton = new QRadioButton(QStringLiteral("LSBF"), this);
    msbfButton->setChecked(true);
    byteOrderGroup_->addButton(msbfButton);
    byteOrderGroup_->addButton(lsbfButton);
    radioLayout->addStretch();
    radioLayout->addWidget(msbfButton);
    radioLayout->addWidget(lsbfButton);
    rootLayout->addLayout(radioLayout);

    QFormLayout* formLayout = new QFormLayout();
    formLayout->setContentsMargins(0, 0, 0, 0);
    formLayout->setSpacing(4);
    hexValue_      = createValueLabel(this);
    binaryValue_   = createValueLabel(this);
    asciiValue_    = createValueLabel(this);
    unsignedValue_ = createValueLabel(this);
    signedValue_   = createValueLabel(this);
    floatValue_    = createValueLabel(this);
    formLayout->addRow(QStringLiteral("Hex:"),      hexValue_);
    formLayout->addRow(QStringLiteral("Binary:"),   binaryValue_);
    formLayout->addRow(QStringLiteral("ASCII:"),    asciiValue_);
    formLayout->addRow(QStringLiteral("Unsigned:"), unsignedValue_);
    formLayout->addRow(QStringLiteral("Signed:"),   signedValue_);
    formLayout->addRow(QStringLiteral("Float:"),    floatValue_);
    rootLayout->addLayout(formLayout);
    rootLayout->addStretch();

    connect(
        byteOrderGroup_,
        QOverload<QAbstractButton*, bool>::of(&QButtonGroup::buttonToggled),
        this,
        [this](QAbstractButton*, bool) { refresh(); }
    );

    clearAnalysis();
}

void FieldCurrentValuePanel::clearAnalysis() {
    hasAnalysis_ = false;
    isPending_ = false;
    lastAnalysis_ = {};
    refresh();
}

void FieldCurrentValuePanel::setPendingAnalysis(const QString&, const QString&) {
    isPending_ = true;
    hasAnalysis_ = false;
    setLabelText(hexValue_,      QStringLiteral("..."));
    setLabelText(binaryValue_,   QStringLiteral("..."));
    setLabelText(asciiValue_,    QStringLiteral("..."));
    setLabelText(unsignedValue_, QStringLiteral("..."));
    setLabelText(signedValue_,   QStringLiteral("..."));
    setLabelText(floatValue_,    QStringLiteral("..."));
}

void FieldCurrentValuePanel::setAnalysis(const features::inspector::FieldInspectorAnalysis& analysis) {
    isPending_ = false;
    hasAnalysis_ = analysis.hasField;
    lastAnalysis_ = analysis;
    refresh();
}

void FieldCurrentValuePanel::refresh() {
    if (isPending_) {
        return;
    }

    if (!hasAnalysis_) {
        setLabelText(hexValue_,      QStringLiteral("-"));
        setLabelText(binaryValue_,   QStringLiteral("-"));
        setLabelText(asciiValue_,    QStringLiteral("-"));
        setLabelText(unsignedValue_, QStringLiteral("-"));
        setLabelText(signedValue_,   QStringLiteral("-"));
        setLabelText(floatValue_,    QStringLiteral("-"));
        return;
    }

    const bool isMsbf = byteOrderGroup_->checkedButton() != nullptr
        && byteOrderGroup_->checkedButton()->text() == QStringLiteral("MSBF");

    setLabelText(hexValue_,      lastAnalysis_.currentHexValue);
    setLabelText(binaryValue_,   lastAnalysis_.currentBinaryValue);
    setLabelText(asciiValue_,    lastAnalysis_.currentAsciiValue);
    setLabelText(unsignedValue_, isMsbf ? lastAnalysis_.currentUnsignedBigEndianValue
                                        : lastAnalysis_.currentUnsignedLittleEndianValue);
    setLabelText(signedValue_,   isMsbf ? lastAnalysis_.currentSignedBigEndianValue
                                        : lastAnalysis_.currentSignedLittleEndianValue);
    setLabelText(floatValue_,    isMsbf ? lastAnalysis_.currentFloatBigEndianValue
                                        : lastAnalysis_.currentFloatLittleEndianValue);
}

void FieldCurrentValuePanel::setLabelText(QLabel* label, const QString& text) {
    if (label == nullptr) {
        return;
    }
    label->setText(text.trimmed().isEmpty() ? QStringLiteral("-") : text);
}

}  // namespace bitabyte::ui
