#include "ui/main_window.h"

#include "features/export/byte_table_csv_exporter.h"
#include "models/byte_table_model.h"
#include "ui/byte_table_view.h"
#include "ui/frame_browser_controller.h"
#include "ui/framing_controller.h"
#include "ui/inspection_controller.h"

#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QMimeData>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStatusBar>
#include <QUrl>

namespace bitabyte::ui {

void MainWindow::openFile() {
    const QString filePath = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("Open Binary File"),
        QString(),
        QStringLiteral("Binary files (*.bin);;All Files (*.*)")
    );
    loadFilePath(filePath);
}

void MainWindow::reloadFile() {
    if (!dataSource_.hasData()) {
        return;
    }

    loadFilePath(dataSource_.sourceFilePath(), false);
}

void MainWindow::exportCsv() {
    if (!dataSource_.hasData()) {
        QMessageBox::information(this, QStringLiteral("Export CSV"), QStringLiteral("Load a file before exporting."));
        return;
    }

    const QString filePath = QFileDialog::getSaveFileName(
        this,
        QStringLiteral("Export Byte Table as CSV"),
        QString(),
        QStringLiteral("CSV files (*.csv);;All Files (*.*)")
    );
    if (filePath.isEmpty()) {
        return;
    }

    QString errorMessage;
    if (!features::exporting::ByteTableCsvExporter::exportToFile(
            dataSource_,
            frameLayout_,
            filePath,
            &errorMessage
        )) {
        QMessageBox::critical(
            this,
            QStringLiteral("Export CSV"),
            QStringLiteral("Failed to write file:\n%1").arg(errorMessage)
        );
        return;
    }

    statusBar()->showMessage(QStringLiteral("Exported CSV"), 3000);
}

void MainWindow::bytesPerRowChanged(int bytesPerRow) {
    dataSource_.setBytesPerRow(bytesPerRow);
    frameLayout_.setRawLayout(
        static_cast<qsizetype>(qMax(1, bytesPerRow)) * 8,
        frameBitOffsetSpinBox_ != nullptr ? frameBitOffsetSpinBox_->value() : 0
    );
    byteTableModel_->reload();
    resizeTableColumns();
    updateLoadedFileState();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
}

void MainWindow::dragEnterEvent(QDragEnterEvent* dragEnterEvent) {
    if (dragEnterEvent == nullptr || dragEnterEvent->mimeData() == nullptr) {
        return;
    }

    const QList<QUrl> droppedUrls = dragEnterEvent->mimeData()->urls();
    if (droppedUrls.isEmpty() || !droppedUrls.first().isLocalFile()) {
        return;
    }

    dragEnterEvent->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent* dropEvent) {
    if (dropEvent == nullptr || dropEvent->mimeData() == nullptr) {
        return;
    }

    const QList<QUrl> droppedUrls = dropEvent->mimeData()->urls();
    if (droppedUrls.isEmpty() || !droppedUrls.first().isLocalFile()) {
        return;
    }

    if (loadFilePath(droppedUrls.first().toLocalFile())) {
        dropEvent->acceptProposedAction();
    }
}

void MainWindow::resetStateForFreshFile() {
    if (frameBrowserController_ != nullptr) {
        frameBrowserController_->clearState();
    }
    frameLayout_ = features::framing::FrameLayout{};
    columnDefinitions_.clear();
    dataSource_.setBytesPerRow(16);

    if (framingController_ != nullptr) {
        framingController_->resetState();
    }
}

void MainWindow::validateFramingStateAfterLoad() {
    if (!frameLayout_.isValidForDataSource(dataSource_)) {
        if (frameBrowserController_ != nullptr) {
            frameBrowserController_->clearState();
        }
        frameLayout_.clearFrame();
    }
}

bool MainWindow::loadFilePath(const QString& filePath, bool resetStateForFreshFile) {
    if (filePath.isEmpty()) {
        return false;
    }

    QString errorMessage;
    if (!dataSource_.loadFile(filePath, &errorMessage)) {
        QMessageBox::critical(
            this,
            QStringLiteral("Open File"),
            QStringLiteral("Failed to load file:\n%1").arg(errorMessage)
        );
        return false;
    }

    if (resetStateForFreshFile) {
        this->resetStateForFreshFile();
    } else {
        if (frameBrowserController_ != nullptr) {
            frameBrowserController_->clearState();
        }
    }
    validateFramingStateAfterLoad();
    byteTableModel_->reload();
    resizeTableColumns();
    byteTableView_->clearSelection();
    byteTableView_->setCurrentIndex(QModelIndex());
    updateLoadedFileState();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
    statusBar()->showMessage(QStringLiteral("Loaded %1").arg(QFileInfo(filePath).fileName()), 3000);
    return true;
}

QString MainWindow::fileSummaryText() const {
    if (!dataSource_.hasData()) {
        return QStringLiteral("No file loaded");
    }

    const QFileInfo fileInfo(dataSource_.sourceFilePath());
    if (frameLayout_.isFramed()) {
        return QStringLiteral("%1\n%2 bytes | %3 framed rows | longest frame %4 bytes")
            .arg(fileInfo.fileName())
            .arg(dataSource_.byteCount())
            .arg(frameLayout_.rowCount(dataSource_))
            .arg(frameLayout_.frameMaxLengthBytes());
    }

    if (byteTableModel_ != nullptr && byteTableModel_->isBitDisplayMode()) {
        return frameLayout_.rawStartBitOffset() == 0
            ? QStringLiteral("%1\n%2 bytes | %3 raw rows | %4 bits/row")
                  .arg(fileInfo.fileName())
                  .arg(dataSource_.byteCount())
                  .arg(frameLayout_.rowCount(dataSource_))
                  .arg(frameLayout_.rawRowWidthBits())
            : QStringLiteral("%1\n%2 bytes | %3 raw rows | %4 bits/row @ bit offset %5")
                  .arg(fileInfo.fileName())
                  .arg(dataSource_.byteCount())
                  .arg(frameLayout_.rowCount(dataSource_))
                  .arg(frameLayout_.rawRowWidthBits())
                  .arg(frameLayout_.rawStartBitOffset());
    }

    return frameLayout_.rawStartBitOffset() == 0
        ? QStringLiteral("%1\n%2 bytes | %3 raw rows | %4 bytes/row")
              .arg(fileInfo.fileName())
              .arg(dataSource_.byteCount())
              .arg(frameLayout_.rowCount(dataSource_))
              .arg(dataSource_.bytesPerRow())
        : QStringLiteral("%1\n%2 bytes | %3 raw rows | %4 bytes/row @ bit offset %5")
              .arg(fileInfo.fileName())
              .arg(dataSource_.byteCount())
              .arg(frameLayout_.rowCount(dataSource_))
              .arg(dataSource_.bytesPerRow())
              .arg(frameLayout_.rawStartBitOffset());
}

}  // namespace bitabyte::ui
