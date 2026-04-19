#include "ui/byte_table_view.h"

#include "models/byte_table_model.h"

#include <QApplication>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QPaintEvent>
#include <QPainter>
#include <QPalette>
#include <QStyledItemDelegate>

namespace bitabyte::ui {
namespace {

QRect bitGlyphRect(const QRect& paintedCellRect, bool hasHighlight) {
    if (!paintedCellRect.isValid()) {
        return paintedCellRect;
    }

    const int minDimension = qMin(paintedCellRect.width(), paintedCellRect.height());
    const int inset = (hasHighlight || minDimension >= 7) ? 1 : 0;
    if (inset <= 0 || paintedCellRect.width() <= (inset * 2) || paintedCellRect.height() <= (inset * 2)) {
        return paintedCellRect;
    }

    return paintedCellRect.adjusted(inset, inset, -inset, -inset);
}

QFont bitDigitFont(const QFont& baseFont, const QRect& paintedCellRect) {
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
    digitFont.setPixelSize(qMax(7, qMin(paintedCellRect.width(), paintedCellRect.height()) - 1));
    return digitFont;
}

class ByteTableHeaderView final : public QHeaderView {
public:
    explicit ByteTableHeaderView(Qt::Orientation orientation, QWidget* parent = nullptr)
        : QHeaderView(orientation, parent) {
        setDefaultAlignment(Qt::AlignCenter);
        setFixedHeight(38);
    }

protected:
    void paintEvent(QPaintEvent* paintEvent) override {
        QPainter painter(viewport());
        painter.fillRect(paintEvent->rect(), palette().color(QPalette::Button));

        const auto* byteTableModel = qobject_cast<const bitabyte::models::ByteTableModel*>(model());
        if (byteTableModel == nullptr || orientation() != Qt::Horizontal) {
            QHeaderView::paintEvent(paintEvent);
            return;
        }

        constexpr int topBandHeight = 18;
        const int bottomBandHeight = height() - topBandHeight;
        const bool hasFrameLengthColumn = byteTableModel->hasFrameLengthColumn();
        const int modelColumnCount = byteTableModel->columnCount();
        const int visibleColumnCount = byteTableModel->visibleColumnCount();

        auto sectionRectForLogicalIndex = [this](int logicalIndex) {
            return QRect(sectionViewportPosition(logicalIndex), 0, sectionSize(logicalIndex), height());
        };

        auto spanRect = [&, this](int firstLogicalIndex, int lastLogicalIndex, int top, int bandHeight) {
            const int left = sectionViewportPosition(firstLogicalIndex);
            const int right = sectionViewportPosition(lastLogicalIndex) + sectionSize(lastLogicalIndex);
            return QRect(left, top, right - left, bandHeight);
        };

        auto drawCell = [&](const QRect& cellRect, const QString& text, bool drawText) {
            painter.fillRect(cellRect, palette().color(QPalette::Button));
            painter.setPen(QColor(190, 190, 190));
            painter.drawRect(cellRect.adjusted(0, 0, -1, -1));
            if (!drawText) {
                return;
            }
            painter.setPen(palette().color(QPalette::ButtonText));
            const QFontMetrics metrics(font());
            painter.drawText(
                cellRect.adjusted(2, 0, -2, 0),
                Qt::AlignCenter,
                metrics.elidedText(text, Qt::ElideRight, cellRect.width() - 4)
            );
        };

        if (hasFrameLengthColumn && modelColumnCount > 0) {
            const QRect lenRect = sectionRectForLogicalIndex(0);
            drawCell(lenRect, QStringLiteral("Len"), true);
        }

        QVector<QString> topLabels;
        QVector<QString> bottomLabels;
        topLabels.reserve(visibleColumnCount);
        bottomLabels.reserve(visibleColumnCount);
        for (int visibleColumnIndex = 0; visibleColumnIndex < visibleColumnCount; ++visibleColumnIndex) {
            topLabels.append(byteTableModel->topHeaderLabelForVisibleColumn(visibleColumnIndex));
            bottomLabels.append(byteTableModel->bottomHeaderLabelForVisibleColumn(visibleColumnIndex));
        }

        auto drawBandSpans = [&](const QVector<QString>& labels, int top, int bandHeight, bool mergeEmptyLabels) {
            int visibleColumnIndex = 0;
            while (visibleColumnIndex < labels.size()) {
                const QString labelText = labels.at(visibleColumnIndex);
                int spanEndIndex = visibleColumnIndex;
                if (mergeEmptyLabels || !labelText.isEmpty()) {
                    while (spanEndIndex + 1 < labels.size() && labels.at(spanEndIndex + 1) == labelText) {
                        ++spanEndIndex;
                    }
                }

                if (labelText.isEmpty() && !mergeEmptyLabels) {
                    spanEndIndex = visibleColumnIndex;
                }

                const int logicalOffset = hasFrameLengthColumn ? 1 : 0;
                const QRect bandRect = spanRect(
                    visibleColumnIndex + logicalOffset,
                    spanEndIndex + logicalOffset,
                    top,
                    bandHeight
                );
                drawCell(bandRect, labelText, !labelText.isEmpty());
                visibleColumnIndex = spanEndIndex + 1;
            }
        };

        drawBandSpans(topLabels, 0, topBandHeight, false);
        drawBandSpans(bottomLabels, topBandHeight, bottomBandHeight, true);
    }
};

class ByteTableItemDelegate final : public QStyledItemDelegate {
public:
    explicit ByteTableItemDelegate(QObject* parent = nullptr)
        : QStyledItemDelegate(parent) {}

    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override {
        const auto* byteTableModel = qobject_cast<const bitabyte::models::ByteTableModel*>(index.model());
        if (byteTableModel == nullptr
            || !byteTableModel->isBitDisplayMode()
            || byteTableModel->isFrameLengthColumn(index.column())) {
            QStyledItemDelegate::paint(painter, option, index);
            return;
        }

        QStyleOptionViewItem viewOption(option);
        initStyleOption(&viewOption, index);

        painter->save();

        const QBrush backgroundBrush = index.data(Qt::BackgroundRole).value<QBrush>();
        const bool hasModelBackground =
            backgroundBrush.style() != Qt::NoBrush && backgroundBrush.color().isValid();
        const QColor backgroundColor = hasModelBackground ? backgroundBrush.color() : QColor(255, 255, 255);
        painter->fillRect(viewOption.rect, backgroundColor);

        const QRect paintedCellRect = viewOption.rect.adjusted(0, 0, -1, -1);
        if (viewOption.text.isEmpty()) {
            if (viewOption.state.testFlag(QStyle::State_Selected)) {
                painter->setPen(QPen(viewOption.palette.highlight().color(), 1));
                painter->setBrush(Qt::NoBrush);
                painter->drawRect(paintedCellRect);
            }
            painter->restore();
            return;
        }

        const bool bitIsOne = viewOption.text == QStringLiteral("1");
        const bool hasHighlight = hasModelBackground;
        const auto bitCellDisplayMode = byteTableModel->bitCellDisplayMode();
        const QRect glyphRect = bitGlyphRect(paintedCellRect, hasHighlight);

        if (bitCellDisplayMode == bitabyte::models::ByteTableModel::BitCellDisplayMode::Digits) {
            painter->setFont(bitDigitFont(viewOption.font, paintedCellRect));
            painter->setRenderHint(QPainter::TextAntialiasing, true);
            painter->setPen(QColor(0, 0, 0));
            painter->drawText(paintedCellRect, Qt::AlignCenter, viewOption.text);
        } else if (bitCellDisplayMode == bitabyte::models::ByteTableModel::BitCellDisplayMode::Circles) {
            painter->setPen(QPen(QColor(190, 190, 190), 1));
            painter->setBrush(bitIsOne ? QBrush(QColor(0, 0, 0)) : QBrush(QColor(255, 255, 255)));
            painter->drawEllipse(glyphRect);
        } else {
            painter->fillRect(glyphRect, bitIsOne ? QColor(0, 0, 0) : QColor(255, 255, 255));
        }

        if (viewOption.state.testFlag(QStyle::State_Selected)) {
            painter->setPen(QPen(viewOption.palette.highlight().color(), 1));
            painter->setBrush(Qt::NoBrush);
            painter->drawRect(paintedCellRect);
        }

        painter->restore();
    }
};

}  // namespace

ByteTableView::ByteTableView(QWidget* parent)
    : QTableView(parent) {
    setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectItems);
    setSelectionMode(QAbstractItemView::SelectionMode::ExtendedSelection);
    setAlternatingRowColors(false);
    setShowGrid(true);
    setGridStyle(Qt::SolidLine);
    setWordWrap(false);
    setTextElideMode(Qt::TextElideMode::ElideNone);
    setHorizontalScrollMode(QAbstractItemView::ScrollMode::ScrollPerPixel);
    setVerticalScrollMode(QAbstractItemView::ScrollMode::ScrollPerPixel);
    QPalette tablePalette = palette();
    tablePalette.setColor(QPalette::Mid, QColor(236, 236, 236));
    setPalette(tablePalette);
    setHorizontalHeader(new ByteTableHeaderView(Qt::Horizontal, this));
    setItemDelegate(new ByteTableItemDelegate(this));
    horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeMode::Fixed);
    horizontalHeader()->setMinimumSectionSize(2);
    verticalHeader()->setSectionResizeMode(QHeaderView::ResizeMode::Fixed);
    verticalHeader()->setDefaultAlignment(Qt::AlignmentFlag::AlignCenter);
    verticalHeader()->setMinimumSectionSize(4);
    verticalHeader()->setDefaultSectionSize(22);
    verticalHeader()->setMinimumWidth(80);
}

void ByteTableView::applyVisibleColumnSizing(int columnCount, bool hasFrameLengthColumn) {
    for (int col = 0; col < columnCount; ++col) {
        if (hasFrameLengthColumn && col == 0) {
            setColumnWidth(col, 52);
            continue;
        }

        setColumnWidth(col, 36);
    }
}

void ByteTableView::focusModelIndex(const QModelIndex& modelIndex) {
    if (!modelIndex.isValid() || selectionModel() == nullptr) {
        return;
    }

    selectionModel()->setCurrentIndex(
        modelIndex,
        QItemSelectionModel::SelectionFlag::ClearAndSelect
            | QItemSelectionModel::SelectionFlag::Current
    );
    scrollTo(modelIndex, QAbstractItemView::ScrollHint::PositionAtCenter);
    setFocus(Qt::FocusReason::OtherFocusReason);
}

}  // namespace bitabyte::ui
