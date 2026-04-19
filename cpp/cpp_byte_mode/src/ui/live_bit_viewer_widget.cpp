#include "ui/live_bit_viewer_widget.h"

#include "data/byte_data_source.h"
#include "features/framing/frame_layout.h"

#include <QColor>
#include <QEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPalette>
#include <QShowEvent>
#include <QSizePolicy>

namespace bitabyte::ui {
namespace {

constexpr int kRowLabelWidth = 55;
constexpr int kCanvasPadding = 6;
constexpr int kDefaultEmptyWidth = 420;
constexpr int kDefaultEmptyHeight = 280;
constexpr int kMaxCanvasHeight = 32'000;
constexpr int kAutoFitVisibleColumnLimit = 10;
constexpr int kMinimumVisibleCellSize = 2;
constexpr int kMaximumAutoFitCellSize = 24;

QFont bitDigitFont(const QFont& baseFont, int cellSize) {
    QFont digitFont(baseFont);
    digitFont.setFamilies({
        QStringLiteral("Cascadia Mono"),
        QStringLiteral("Consolas"),
        QStringLiteral("Courier New"),
        QStringLiteral("Monospace"),
    });
    digitFont.setStyleHint(QFont::StyleHint::Monospace);
    digitFont.setBold(true);
    digitFont.setHintingPreference(QFont::HintingPreference::PreferFullHinting);
    digitFont.setPixelSize(qMax(7, cellSize - 1));
    return digitFont;
}

}  // namespace

LiveBitViewerWidget::LiveBitViewerWidget(QWidget* parent)
    : QWidget(parent) {
    setAutoFillBackground(true);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    QPalette widgetPalette = palette();
    widgetPalette.setColor(QPalette::ColorRole::Window, QColor(255, 255, 255));
    setPalette(widgetPalette);
    attachViewportResizeTracking();
    updateCanvasSize();
}

void LiveBitViewerWidget::setPreviewSource(
    const data::ByteDataSource* dataSource,
    const features::framing::FrameLayout* frameLayout,
    const QVector<features::columns::VisibleByteColumn>& visibleColumns,
    int firstFrameRowIndex,
    int previewRowCount
) {
    dataSource_ = dataSource;
    frameLayout_ = frameLayout;
    firstFrameRowIndex_ = qMax(0, firstFrameRowIndex);
    previewRowCount_ = qMax(0, previewRowCount);
    previewColumns_.clear();
    maxFrameBits_ = 0;
    autoFitBitCount_ = 0;

    int previewBitOffset = 0;
    for (int visibleColumnIndex = 0; visibleColumnIndex < visibleColumns.size(); ++visibleColumnIndex) {
        const features::columns::VisibleByteColumn& visibleColumn = visibleColumns.at(visibleColumnIndex);
        PreviewColumn previewColumn;
        previewColumn.visibleColumn = visibleColumn;
        previewColumn.previewBitStart = previewBitOffset;
        previewColumn.bitWidth = visibleColumn.bitWidth();
        previewColumns_.append(previewColumn);
        previewBitOffset += previewColumn.bitWidth;
        if (visibleColumnIndex < kAutoFitVisibleColumnLimit) {
            autoFitBitCount_ += previewColumn.bitWidth;
        }
    }
    maxFrameBits_ = previewBitOffset;
    if (autoFitBitCount_ <= 0) {
        autoFitBitCount_ = maxFrameBits_;
    }
    updateCanvasSize();
    update();
}

void LiveBitViewerWidget::setPreviewColumnHighlights(const QHash<int, QColor>& previewColumnHighlights) {
    if (previewColumnHighlights_ == previewColumnHighlights) {
        return;
    }

    previewColumnHighlights_ = previewColumnHighlights;
    update();
}

void LiveBitViewerWidget::setPreviewBitHighlights(const QVector<PreviewBitHighlight>& previewBitHighlights) {
    if (previewBitHighlights_.size() == previewBitHighlights.size()) {
        bool allEqual = true;
        for (int highlightIndex = 0; highlightIndex < previewBitHighlights.size(); ++highlightIndex) {
            const PreviewBitHighlight& leftHighlight = previewBitHighlights_.at(highlightIndex);
            const PreviewBitHighlight& rightHighlight = previewBitHighlights.at(highlightIndex);
            if (leftHighlight.absoluteStartBit != rightHighlight.absoluteStartBit
                || leftHighlight.absoluteEndBit != rightHighlight.absoluteEndBit
                || leftHighlight.color != rightHighlight.color) {
                allEqual = false;
                break;
            }
        }
        if (allEqual) {
            return;
        }
    }

    previewBitHighlights_ = previewBitHighlights;
    update();
}

void LiveBitViewerWidget::setDisplayMode(const QString& displayMode) {
    if (displayMode_ == displayMode) {
        return;
    }

    displayMode_ = displayMode;
    updateCanvasSize();
    update();
}

void LiveBitViewerWidget::setAutoFitEnabled(bool enabled) {
    if (autoFitEnabled_ == enabled) {
        return;
    }

    autoFitEnabled_ = enabled;
    attachViewportResizeTracking();
    updateCanvasSize();
    update();
}

void LiveBitViewerWidget::setCellSize(int cellSize) {
    const int normalizedCellSize = qMax(4, cellSize);
    if (requestedCellSize_ == normalizedCellSize) {
        return;
    }

    requestedCellSize_ = normalizedCellSize;
    updateCanvasSize();
    update();
}

QSize LiveBitViewerWidget::minimumSizeHint() const {
    return sizeHint();
}

QSize LiveBitViewerWidget::sizeHint() const {
    if (previewColumns_.isEmpty() || previewRowCount_ <= 0 || maxFrameBits_ <= 0) {
        return QSize(kDefaultEmptyWidth, kDefaultEmptyHeight);
    }

    const int cellSize = effectiveCellSize();
    const int width = kRowLabelWidth + (maxFrameBits_ * cellSize) + (kCanvasPadding * 2);
    const int height = qMin(kMaxCanvasHeight, (previewRowCount_ * cellSize) + (kCanvasPadding * 2));
    return QSize(qMax(kDefaultEmptyWidth, width), qMax(kDefaultEmptyHeight, height));
}

bool LiveBitViewerWidget::eventFilter(QObject* watched, QEvent* event) {
    if (autoFitEnabled_
        && watched == trackedViewport_
        && event != nullptr
        && event->type() == QEvent::Type::Resize) {
        updateCanvasSize();
        update();
    }
    return QWidget::eventFilter(watched, event);
}

void LiveBitViewerWidget::paintEvent(QPaintEvent* paintEvent) {
    QPainter painter(this);
    painter.fillRect(rect(), QColor(255, 255, 255));

    if (previewColumns_.isEmpty()
        || previewRowCount_ <= 0
        || maxFrameBits_ <= 0
        || dataSource_ == nullptr
        || frameLayout_ == nullptr) {
        paintEmptyState(&painter, paintEvent->rect());
        return;
    }

    paintFrames(&painter, paintEvent->rect());
}

void LiveBitViewerWidget::showEvent(QShowEvent* showEvent) {
    QWidget::showEvent(showEvent);
    attachViewportResizeTracking();
    updateCanvasSize();
}

int LiveBitViewerWidget::effectiveCellSize() const {
    const int requestedCellSize = qMax(4, requestedCellSize_);
    if (!autoFitEnabled_
        || previewColumns_.isEmpty()
        || previewRowCount_ <= 0
        || maxFrameBits_ <= 0
        || displayMode_ == QStringLiteral("binary")
        || displayMode_ == QStringLiteral("hex")
        || displayMode_ == QStringLiteral("digits")) {
        return requestedCellSize;
    }

    const int availableWidth = qMax(1, availableViewportWidth() - kRowLabelWidth - (kCanvasPadding * 2));
    const int widthFittedCellSize = autoFitBitCount_ > 0
        ? qMax(kMinimumVisibleCellSize, availableWidth / autoFitBitCount_)
        : requestedCellSize;
    return qBound(kMinimumVisibleCellSize, widthFittedCellSize, kMaximumAutoFitCellSize);
}

int LiveBitViewerWidget::availableViewportWidth() const {
    QWidget* viewport = parentWidget();
    if (viewport == nullptr) {
        return qMax(width(), kDefaultEmptyWidth);
    }
    return qMax(viewport->width(), kDefaultEmptyWidth);
}

int LiveBitViewerWidget::availableViewportHeight() const {
    QWidget* viewport = parentWidget();
    if (viewport == nullptr) {
        return qMax(height(), kDefaultEmptyHeight);
    }
    return qMax(viewport->height(), kDefaultEmptyHeight);
}

void LiveBitViewerWidget::attachViewportResizeTracking() {
    if (!autoFitEnabled_) {
        if (trackedViewport_ != nullptr) {
            trackedViewport_->removeEventFilter(this);
            trackedViewport_ = nullptr;
        }
        return;
    }

    QWidget* viewport = parentWidget();
    if (trackedViewport_ == viewport) {
        return;
    }

    if (trackedViewport_ != nullptr) {
        trackedViewport_->removeEventFilter(this);
    }

    trackedViewport_ = viewport;
    if (trackedViewport_ != nullptr) {
        trackedViewport_->installEventFilter(this);
    }
}

void LiveBitViewerWidget::updateCanvasSize() {
    const QSize canvasSize = sizeHint();
    const bool minimumSizeChanged = minimumSize() != canvasSize;
    const bool widgetSizeChanged = size() != canvasSize;
    if (!minimumSizeChanged && !widgetSizeChanged) {
        return;
    }

    if (minimumSizeChanged) {
        setMinimumSize(canvasSize);
    }
    if (widgetSizeChanged) {
        resize(canvasSize);
    }
    updateGeometry();
}

void LiveBitViewerWidget::paintEmptyState(QPainter* painter, const QRect& clipRect) {
    painter->save();
    painter->setPen(QColor(130, 130, 130));
    painter->drawText(
        clipRect,
        Qt::AlignCenter,
        QStringLiteral("Highlight byte columns to preview bits")
    );
    painter->restore();
}

void LiveBitViewerWidget::paintFrames(QPainter* painter, const QRect& clipRect) {
    painter->save();
    const int cellSize = effectiveCellSize();
    if (cellSize <= 0) {
        painter->restore();
        return;
    }
    QFont rowLabelFont = painter->font();
    rowLabelFont.setFamilies({
        QStringLiteral("Consolas"),
        QStringLiteral("Courier New"),
        QStringLiteral("Monospace"),
    });
    rowLabelFont.setStyleHint(QFont::StyleHint::Monospace);
    painter->setFont(rowLabelFont);

    const int firstVisibleRow = qMax(0, (clipRect.top() - kCanvasPadding) / cellSize);
    const int lastVisibleRow = qMin(previewRowCount_ - 1, (clipRect.bottom() - kCanvasPadding) / cellSize);
    if (firstVisibleRow > lastVisibleRow) {
        painter->restore();
        return;
    }

    const bool showRowLabels = cellSize >= 8;
    const bool useDigitGlyphs = displayMode_ == QStringLiteral("digits");
    const bool useCircleGlyphs = !useDigitGlyphs && displayMode_ == QStringLiteral("circles") && cellSize >= 4;
    const int firstVisiblePreviewBit = qMax(0, (clipRect.left() - kRowLabelWidth - kCanvasPadding) / cellSize);
    const int lastVisiblePreviewBit = qMax(
        firstVisiblePreviewBit,
        (clipRect.right() - kRowLabelWidth - kCanvasPadding) / cellSize
    );
    const int drawRight = kRowLabelWidth + kCanvasPadding + ((lastVisiblePreviewBit + 1) * cellSize);
    if (drawRight < clipRect.left()) {
        painter->restore();
        return;
    }

    for (int row = firstVisibleRow; row <= lastVisibleRow; ++row) {
        const int frameRowIndex = firstFrameRowIndex_ + row;
        const int rowY = kCanvasPadding + (row * cellSize);
        const qsizetype rowStartBit = frameLayout_->rowStartBit(*dataSource_, frameRowIndex);
        const qsizetype rowLengthBits = frameLayout_->rowLengthBits(*dataSource_, frameRowIndex);
        if (rowStartBit < 0 || rowLengthBits <= 0) {
            continue;
        }

        if (showRowLabels) {
            painter->setPen(QColor(80, 80, 80));
            painter->drawText(
                QRect(0, rowY, kRowLabelWidth - 4, cellSize),
                Qt::AlignRight | Qt::AlignVCenter,
                QString::number(frameRowIndex)
            );
        }

        int highlightIndex = 0;
        for (int previewColumnIndex = 0; previewColumnIndex < previewColumns_.size(); ++previewColumnIndex) {
            const PreviewColumn& previewColumn = previewColumns_.at(previewColumnIndex);
            const int previewColumnEndBit = previewColumn.previewBitStart + previewColumn.bitWidth - 1;
            if (previewColumnEndBit < firstVisiblePreviewBit || previewColumn.previewBitStart > lastVisiblePreviewBit) {
                continue;
            }

            const QColor previewColumnHighlightColor = previewColumnHighlights_.value(previewColumnIndex, QColor());

            for (int bitOffset = 0; bitOffset < previewColumn.bitWidth; ++bitOffset) {
                const int relativeBit = previewColumn.visibleColumn.absoluteStartBit + bitOffset;
                if (relativeBit >= rowLengthBits) {
                    break;
                }

                const int previewBitIndex = previewColumn.previewBitStart + bitOffset;
                if (previewBitIndex < firstVisiblePreviewBit || previewBitIndex > lastVisiblePreviewBit) {
                    continue;
                }
                const int cellX = kRowLabelWidth + kCanvasPadding + (previewBitIndex * cellSize);
                const QRect cellRect(cellX, rowY, cellSize, cellSize);
                const bool bitIsOne = dataSource_->bitAt(rowStartBit + relativeBit) != 0;
                const QRect paintedCellRect = cellRect.adjusted(0, 0, -1, -1);
                while (highlightIndex < previewBitHighlights_.size()
                    && previewBitHighlights_.at(highlightIndex).absoluteEndBit < relativeBit) {
                    ++highlightIndex;
                }
                QColor bitHighlightColor;
                if (highlightIndex < previewBitHighlights_.size()) {
                    const PreviewBitHighlight& previewBitHighlight = previewBitHighlights_.at(highlightIndex);
                    if (relativeBit >= previewBitHighlight.absoluteStartBit
                        && relativeBit <= previewBitHighlight.absoluteEndBit) {
                        bitHighlightColor = previewBitHighlight.color;
                    }
                }
                const QColor effectiveHighlightColor =
                    bitHighlightColor.isValid() ? bitHighlightColor : previewColumnHighlightColor;
                const bool hasHighlight = effectiveHighlightColor.isValid();

                if (hasHighlight) {
                    painter->fillRect(paintedCellRect, effectiveHighlightColor);
                }

                if (useDigitGlyphs) {
                    painter->setFont(bitDigitFont(font(), cellSize));
                    painter->setRenderHint(QPainter::TextAntialiasing, true);
                    painter->setPen(QColor(0, 0, 0));
                    painter->drawText(paintedCellRect, Qt::AlignCenter, bitIsOne ? QStringLiteral("1") : QStringLiteral("0"));
                    continue;
                }

                if (useCircleGlyphs) {
                    painter->setPen(QPen(QColor(100, 100, 100), 1));
                    painter->setBrush(bitIsOne ? QBrush(QColor(0, 0, 0)) : QBrush(QColor(255, 255, 255)));
                    painter->drawEllipse(hasHighlight ? paintedCellRect.adjusted(1, 1, -1, -1) : paintedCellRect);
                    continue;
                }

                if (hasHighlight) {
                    painter->fillRect(
                        paintedCellRect.adjusted(1, 1, -1, -1),
                        bitIsOne ? QColor(0, 0, 0) : QColor(255, 255, 255)
                    );
                } else {
                    painter->fillRect(paintedCellRect, bitIsOne ? QColor(0, 0, 0) : QColor(255, 255, 255));
                }
                if (cellSize >= 3) {
                    painter->setPen(QPen(QColor(180, 180, 180), 1));
                    painter->drawRect(paintedCellRect);
                }
            }
        }
    }

    painter->restore();
}

}  // namespace bitabyte::ui
