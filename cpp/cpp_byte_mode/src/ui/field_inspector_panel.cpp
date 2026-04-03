#include "ui/field_inspector_panel.h"

#include <QFontMetrics>
#include <QPainter>
#include <QVBoxLayout>

namespace bitabyte::ui {

class FieldInspectorPanel::HistogramWidget final : public QWidget {
public:
    explicit HistogramWidget(QWidget* parent = nullptr)
        : QWidget(parent) {}

    void setBins(const QVector<features::inspector::HistogramBin>& histogramBins) {
        histogramBins_ = histogramBins;
        update();
    }

    void setEmptyMessage(const QString& emptyMessage) {
        if (emptyMessage_ == emptyMessage) {
            return;
        }

        emptyMessage_ = emptyMessage;
        update();
    }

    [[nodiscard]] QSize minimumSizeHint() const override {
        return QSize(280, 120);
    }

    [[nodiscard]] QSize sizeHint() const override {
        return QSize(280, 120);
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.fillRect(rect(), palette().base());

        if (histogramBins_.isEmpty()) {
            painter.setPen(palette().color(QPalette::Disabled, QPalette::Text));
            painter.drawText(rect().adjusted(8, 8, -8, -8), Qt::AlignCenter, emptyMessage_);
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
    QString emptyMessage_ = QStringLiteral("No distribution");
};

FieldInspectorPanel::FieldInspectorPanel(QWidget* parent)
    : QWidget(parent) {
    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);
    histogramWidget_ = new HistogramWidget(this);
    rootLayout->addWidget(histogramWidget_);
    clearAnalysis();
}

void FieldInspectorPanel::clearAnalysis() {
    histogramWidget_->setEmptyMessage(QStringLiteral("No distribution"));
    histogramWidget_->setBins({});
}

void FieldInspectorPanel::showUnavailable() {
    histogramWidget_->setEmptyMessage(QStringLiteral("Distribution is available after framing is active."));
    histogramWidget_->setBins({});
}

void FieldInspectorPanel::setPendingAnalysis(const QString&, const QString&) {
    // Keep existing histogram visible while analysis is running.
}

void FieldInspectorPanel::setAnalysis(const features::inspector::FieldInspectorAnalysis& analysis) {
    if (!analysis.hasField) {
        clearAnalysis();
        return;
    }
    histogramWidget_->setBins(analysis.histogramBins);
}

}  // namespace bitabyte::ui
