#include "ui/field_inspector_panel.h"

#include <QFont>
#include <QFontMetrics>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPainter>
#include <QVBoxLayout>

namespace bitabyte::ui {
namespace {

QLabel* createValueLabel(QWidget* parent, bool monospace = false) {
    QLabel* valueLabel = new QLabel(parent);
    valueLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    valueLabel->setWordWrap(true);
    if (monospace) {
        QFont monospaceFont = valueLabel->font();
        monospaceFont.setFamilies({
            QStringLiteral("Consolas"),
            QStringLiteral("Courier New"),
            QStringLiteral("Monospace"),
        });
        monospaceFont.setStyleHint(QFont::StyleHint::Monospace);
        valueLabel->setFont(monospaceFont);
    }
    return valueLabel;
}

QString yesNoText(bool flagValue) {
    return flagValue ? QStringLiteral("Yes") : QStringLiteral("No");
}

}  // namespace

class FieldInspectorPanel::HistogramWidget final : public QWidget {
public:
    explicit HistogramWidget(QWidget* parent = nullptr)
        : QWidget(parent) {}

    void setBins(const QVector<features::inspector::HistogramBin>& histogramBins) {
        histogramBins_ = histogramBins;
        updateGeometry();
        update();
    }

    [[nodiscard]] QSize minimumSizeHint() const override {
        return QSize(280, histogramBins_.isEmpty() ? 72 : qMax(120, 16 + (histogramBins_.size() * 24)));
    }

    [[nodiscard]] QSize sizeHint() const override {
        return minimumSizeHint();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.fillRect(rect(), palette().base());

        if (histogramBins_.isEmpty()) {
            painter.setPen(palette().color(QPalette::Disabled, QPalette::Text));
            painter.drawText(rect().adjusted(8, 8, -8, -8), Qt::AlignCenter, QStringLiteral("No distribution"));
            return;
        }

        const int maximumCount = std::max_element(
            histogramBins_.cbegin(),
            histogramBins_.cend(),
            [](const auto& leftBin, const auto& rightBin) {
                return leftBin.count < rightBin.count;
            }
        )->count;

        const QFontMetrics fontMetrics(font());
        const int labelWidth = 112;
        const int countWidth = 44;
        const int barHeight = 14;
        const int rowHeight = 22;
        const int topMargin = 8;

        for (int index = 0; index < histogramBins_.size(); ++index) {
            const features::inspector::HistogramBin& histogramBin = histogramBins_.at(index);
            const int rowY = topMargin + (index * rowHeight);
            const QRect labelRect(8, rowY, labelWidth, barHeight);
            const QRect countRect(width() - countWidth - 8, rowY, countWidth, barHeight);
            const QRect barRect(labelRect.right() + 8, rowY, countRect.left() - labelRect.right() - 16, barHeight);

            painter.setPen(palette().color(QPalette::Text));
            painter.drawText(
                labelRect,
                Qt::AlignLeft | Qt::AlignVCenter,
                fontMetrics.elidedText(histogramBin.label, Qt::ElideRight, labelRect.width())
            );

            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(232, 236, 243));
            painter.drawRect(barRect);

            const int filledWidth = maximumCount > 0
                ? static_cast<int>(std::round((static_cast<double>(histogramBin.count) / static_cast<double>(maximumCount)) * barRect.width()))
                : 0;
            painter.setBrush(QColor(102, 160, 242));
            painter.drawRect(QRect(barRect.left(), barRect.top(), filledWidth, barRect.height()));

            painter.setPen(palette().color(QPalette::Text));
            painter.drawText(countRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(histogramBin.count));
        }
    }

private:
    QVector<features::inspector::HistogramBin> histogramBins_;
};

FieldInspectorPanel::FieldInspectorPanel(QWidget* parent)
    : QWidget(parent) {
    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(8);

    QGroupBox* selectionGroup = new QGroupBox(QStringLiteral("Selection"), this);
    QFormLayout* selectionLayout = new QFormLayout(selectionGroup);
    fieldLabelValue_ = createValueLabel(selectionGroup, true);
    positionValue_ = createValueLabel(selectionGroup, true);
    currentRowValue_ = createValueLabel(selectionGroup, true);
    analyzedFramesValue_ = createValueLabel(selectionGroup, true);
    missingFramesValue_ = createValueLabel(selectionGroup, true);
    uniqueValuesValue_ = createValueLabel(selectionGroup, true);
    selectionLayout->addRow(QStringLiteral("Field:"), fieldLabelValue_);
    selectionLayout->addRow(QStringLiteral("Position:"), positionValue_);
    selectionLayout->addRow(QStringLiteral("Current row:"), currentRowValue_);
    selectionLayout->addRow(QStringLiteral("Analyzed:"), analyzedFramesValue_);
    selectionLayout->addRow(QStringLiteral("Missing:"), missingFramesValue_);
    selectionLayout->addRow(QStringLiteral("Unique:"), uniqueValuesValue_);
    rootLayout->addWidget(selectionGroup);

    QGroupBox* currentValueGroup = new QGroupBox(QStringLiteral("Current Value"), this);
    QFormLayout* currentValueLayout = new QFormLayout(currentValueGroup);
    currentHexValue_ = createValueLabel(currentValueGroup, true);
    currentBinaryValue_ = createValueLabel(currentValueGroup, true);
    currentAsciiValue_ = createValueLabel(currentValueGroup, true);
    currentUnsignedBigEndianValue_ = createValueLabel(currentValueGroup, true);
    currentUnsignedLittleEndianValue_ = createValueLabel(currentValueGroup, true);
    currentSignedBigEndianValue_ = createValueLabel(currentValueGroup, true);
    currentSignedLittleEndianValue_ = createValueLabel(currentValueGroup, true);
    currentFloatBigEndianValue_ = createValueLabel(currentValueGroup, true);
    currentFloatLittleEndianValue_ = createValueLabel(currentValueGroup, true);
    currentValueLayout->addRow(QStringLiteral("Hex:"), currentHexValue_);
    currentValueLayout->addRow(QStringLiteral("Binary:"), currentBinaryValue_);
    currentValueLayout->addRow(QStringLiteral("ASCII:"), currentAsciiValue_);
    currentValueLayout->addRow(QStringLiteral("Unsigned BE:"), currentUnsignedBigEndianValue_);
    currentValueLayout->addRow(QStringLiteral("Unsigned LE:"), currentUnsignedLittleEndianValue_);
    currentValueLayout->addRow(QStringLiteral("Signed BE:"), currentSignedBigEndianValue_);
    currentValueLayout->addRow(QStringLiteral("Signed LE:"), currentSignedLittleEndianValue_);
    currentValueLayout->addRow(QStringLiteral("Float BE:"), currentFloatBigEndianValue_);
    currentValueLayout->addRow(QStringLiteral("Float LE:"), currentFloatLittleEndianValue_);
    rootLayout->addWidget(currentValueGroup);

    QGroupBox* statisticsGroup = new QGroupBox(QStringLiteral("Statistics"), this);
    QFormLayout* statisticsLayout = new QFormLayout(statisticsGroup);
    minimumValue_ = createValueLabel(statisticsGroup, true);
    maximumValue_ = createValueLabel(statisticsGroup, true);
    meanValue_ = createValueLabel(statisticsGroup, true);
    modeValue_ = createValueLabel(statisticsGroup, true);
    entropyValue_ = createValueLabel(statisticsGroup, true);
    constantFlagValue_ = createValueLabel(statisticsGroup, true);
    monotonicFlagValue_ = createValueLabel(statisticsGroup, true);
    highEntropyFlagValue_ = createValueLabel(statisticsGroup, true);
    statisticsLayout->addRow(QStringLiteral("Min (uBE):"), minimumValue_);
    statisticsLayout->addRow(QStringLiteral("Max (uBE):"), maximumValue_);
    statisticsLayout->addRow(QStringLiteral("Mean (uBE):"), meanValue_);
    statisticsLayout->addRow(QStringLiteral("Mode:"), modeValue_);
    statisticsLayout->addRow(QStringLiteral("Entropy:"), entropyValue_);
    statisticsLayout->addRow(QStringLiteral("Constant:"), constantFlagValue_);
    statisticsLayout->addRow(QStringLiteral("Monotonic:"), monotonicFlagValue_);
    statisticsLayout->addRow(QStringLiteral("High entropy:"), highEntropyFlagValue_);
    rootLayout->addWidget(statisticsGroup);

    QGroupBox* histogramGroup = new QGroupBox(QStringLiteral("Distribution"), this);
    QVBoxLayout* histogramLayout = new QVBoxLayout(histogramGroup);
    histogramWidget_ = new HistogramWidget(histogramGroup);
    histogramLayout->addWidget(histogramWidget_);
    rootLayout->addWidget(histogramGroup);
    rootLayout->addStretch();

    clearAnalysis();
}

void FieldInspectorPanel::clearAnalysis() {
    setLabelText(fieldLabelValue_, QStringLiteral("-"));
    setLabelText(positionValue_, QStringLiteral("-"));
    setLabelText(currentRowValue_, QStringLiteral("-"));
    setLabelText(analyzedFramesValue_, QStringLiteral("-"));
    setLabelText(missingFramesValue_, QStringLiteral("-"));
    setLabelText(uniqueValuesValue_, QStringLiteral("-"));
    setLabelText(currentHexValue_, QStringLiteral("-"));
    setLabelText(currentBinaryValue_, QStringLiteral("-"));
    setLabelText(currentAsciiValue_, QStringLiteral("-"));
    setLabelText(currentUnsignedBigEndianValue_, QStringLiteral("-"));
    setLabelText(currentUnsignedLittleEndianValue_, QStringLiteral("-"));
    setLabelText(currentSignedBigEndianValue_, QStringLiteral("-"));
    setLabelText(currentSignedLittleEndianValue_, QStringLiteral("-"));
    setLabelText(currentFloatBigEndianValue_, QStringLiteral("-"));
    setLabelText(currentFloatLittleEndianValue_, QStringLiteral("-"));
    setLabelText(minimumValue_, QStringLiteral("-"));
    setLabelText(maximumValue_, QStringLiteral("-"));
    setLabelText(meanValue_, QStringLiteral("-"));
    setLabelText(modeValue_, QStringLiteral("-"));
    setLabelText(entropyValue_, QStringLiteral("-"));
    setLabelText(constantFlagValue_, QStringLiteral("-"));
    setLabelText(monotonicFlagValue_, QStringLiteral("-"));
    setLabelText(highEntropyFlagValue_, QStringLiteral("-"));
    histogramWidget_->setBins({});
}

void FieldInspectorPanel::setPendingAnalysis(const QString& fieldLabel, const QString& positionText) {
    setLabelText(fieldLabelValue_, fieldLabel);
    setLabelText(positionValue_, positionText);
    setLabelText(currentRowValue_, QStringLiteral("Analyzing..."));
    setLabelText(analyzedFramesValue_, QStringLiteral("..."));
    setLabelText(missingFramesValue_, QStringLiteral("..."));
    setLabelText(uniqueValuesValue_, QStringLiteral("..."));
    setLabelText(currentHexValue_, QStringLiteral("..."));
    setLabelText(currentBinaryValue_, QStringLiteral("..."));
    setLabelText(currentAsciiValue_, QStringLiteral("..."));
    setLabelText(currentUnsignedBigEndianValue_, QStringLiteral("..."));
    setLabelText(currentUnsignedLittleEndianValue_, QStringLiteral("..."));
    setLabelText(currentSignedBigEndianValue_, QStringLiteral("..."));
    setLabelText(currentSignedLittleEndianValue_, QStringLiteral("..."));
    setLabelText(currentFloatBigEndianValue_, QStringLiteral("..."));
    setLabelText(currentFloatLittleEndianValue_, QStringLiteral("..."));
    setLabelText(minimumValue_, QStringLiteral("..."));
    setLabelText(maximumValue_, QStringLiteral("..."));
    setLabelText(meanValue_, QStringLiteral("..."));
    setLabelText(modeValue_, QStringLiteral("..."));
    setLabelText(entropyValue_, QStringLiteral("..."));
    setLabelText(constantFlagValue_, QStringLiteral("..."));
    setLabelText(monotonicFlagValue_, QStringLiteral("..."));
    setLabelText(highEntropyFlagValue_, QStringLiteral("..."));
    histogramWidget_->setBins({});
}

void FieldInspectorPanel::setAnalysis(const features::inspector::FieldInspectorAnalysis& analysis) {
    if (!analysis.hasField) {
        clearAnalysis();
        return;
    }

    setLabelText(fieldLabelValue_, analysis.fieldLabel);
    setLabelText(positionValue_, analysis.positionText);
    setLabelText(currentRowValue_, analysis.currentRowText);
    setLabelText(analyzedFramesValue_, QString::number(analysis.analyzedFrameCount));
    setLabelText(missingFramesValue_, QString::number(analysis.missingFrameCount));
    setLabelText(uniqueValuesValue_, QString::number(analysis.uniqueValueCount));
    setLabelText(currentHexValue_, analysis.currentHexValue);
    setLabelText(currentBinaryValue_, analysis.currentBinaryValue);
    setLabelText(currentAsciiValue_, analysis.currentAsciiValue);
    setLabelText(currentUnsignedBigEndianValue_, analysis.currentUnsignedBigEndianValue);
    setLabelText(currentUnsignedLittleEndianValue_, analysis.currentUnsignedLittleEndianValue);
    setLabelText(currentSignedBigEndianValue_, analysis.currentSignedBigEndianValue);
    setLabelText(currentSignedLittleEndianValue_, analysis.currentSignedLittleEndianValue);
    setLabelText(currentFloatBigEndianValue_, analysis.currentFloatBigEndianValue);
    setLabelText(currentFloatLittleEndianValue_, analysis.currentFloatLittleEndianValue);
    setLabelText(minimumValue_, analysis.minValueText);
    setLabelText(maximumValue_, analysis.maxValueText);
    setLabelText(meanValue_, analysis.meanValueText);
    setLabelText(modeValue_, analysis.modeValueText);
    setLabelText(entropyValue_, analysis.entropyText);
    setLabelText(constantFlagValue_, yesNoText(analysis.isConstant));
    setLabelText(monotonicFlagValue_, yesNoText(analysis.isMonotonicIncreasing));
    setLabelText(highEntropyFlagValue_, yesNoText(analysis.isHighEntropy));
    histogramWidget_->setBins(analysis.histogramBins);
}

void FieldInspectorPanel::setLabelText(QLabel* label, const QString& text) {
    if (label == nullptr) {
        return;
    }

    label->setText(text.trimmed().isEmpty() ? QStringLiteral("-") : text);
}

}  // namespace bitabyte::ui
