#pragma once

#include <QColor>
#include <QHash>
#include <QVector>
#include <QWidget>

#include "features/columns/visible_byte_column.h"

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::features::framing {
class FrameLayout;
}

namespace bitabyte::ui {

class LiveBitViewerWidget final : public QWidget {
    Q_OBJECT

public:
    struct PreviewBitHighlight {
        int absoluteStartBit = 0;
        int absoluteEndBit = -1;
        QColor color;
    };

    explicit LiveBitViewerWidget(QWidget* parent = nullptr);

    void setPreviewSource(
        const data::ByteDataSource* dataSource,
        const features::framing::FrameLayout* frameLayout,
        const QVector<features::columns::VisibleByteColumn>& visibleColumns,
        int firstFrameRowIndex,
        int previewRowCount
    );
    void setPreviewColumnHighlights(const QHash<int, QColor>& previewColumnHighlights);
    void setPreviewBitHighlights(const QVector<PreviewBitHighlight>& previewBitHighlights);
    void setDisplayMode(const QString& displayMode);
    void setAutoFitEnabled(bool enabled);
    void setCellSize(int cellSize);

    [[nodiscard]] QSize minimumSizeHint() const override;
    [[nodiscard]] QSize sizeHint() const override;

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void paintEvent(QPaintEvent* paintEvent) override;
    void showEvent(QShowEvent* showEvent) override;

private:
    struct PreviewColumn {
        features::columns::VisibleByteColumn visibleColumn;
        int previewBitStart = 0;
        int bitWidth = 0;
    };

    [[nodiscard]] int effectiveCellSize() const;
    [[nodiscard]] int availableViewportWidth() const;
    [[nodiscard]] int availableViewportHeight() const;
    void attachViewportResizeTracking();
    void updateCanvasSize();
    void paintEmptyState(QPainter* painter, const QRect& clipRect);
    void paintFrames(QPainter* painter, const QRect& clipRect);

    const data::ByteDataSource* dataSource_ = nullptr;
    const features::framing::FrameLayout* frameLayout_ = nullptr;
    QVector<PreviewColumn> previewColumns_;
    QString displayMode_ = QStringLiteral("squares");
    int requestedCellSize_ = 10;
    bool autoFitEnabled_ = true;
    int maxFrameBits_ = 0;
    int autoFitBitCount_ = 0;
    int firstFrameRowIndex_ = 0;
    int previewRowCount_ = 0;
    QHash<int, QColor> previewColumnHighlights_;
    QVector<PreviewBitHighlight> previewBitHighlights_;
    QWidget* trackedViewport_ = nullptr;
};

}  // namespace bitabyte::ui
