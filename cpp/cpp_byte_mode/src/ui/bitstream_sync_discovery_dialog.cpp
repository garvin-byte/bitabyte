#include "ui/bitstream_sync_discovery_dialog.h"

#include "data/byte_data_source.h"
#include "features/bitstream_sync_discovery/bitstream_sync_discovery_worker.h"
#include "features/columns/visible_byte_column.h"
#include "features/framing/frame_layout.h"
#include "ui/live_bit_viewer_widget.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>

namespace bitabyte::ui {
namespace {

constexpr int kRankColumn = 0;
constexpr int kPatternColumn = 1;
constexpr int kBitsColumn = 2;
constexpr int kMatchesColumn = 3;
constexpr int kFramesColumn = 4;
constexpr int kMedianBytesColumn = 5;
constexpr int kGroupsColumn = 6;
constexpr int kCliffsColumn = 7;
constexpr int kScoreColumn = 8;

QString bitLengthLabel(qsizetype lengthBits) {
    return QStringLiteral("%1 bits (%2 bytes)")
        .arg(lengthBits)
        .arg((lengthBits + 7) / 8);
}

qsizetype representativeFrameLengthBits(
    const features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidate
) {
    return qMax<qsizetype>(
        candidate.refinedPattern.bitWidth,
        qMax<qsizetype>(1, candidate.frameLengthSummary.medianLengthBits)
    );
}

qsizetype estimatedFrameCountFromMedianLength(
    const features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidate,
    qsizetype totalBitCount
) {
    if (candidate.frameSpans.isEmpty()) {
        return 0;
    }

    const qsizetype candidateStartBit = candidate.frameSpans.first().startBit;
    if (candidateStartBit >= totalBitCount) {
        return 0;
    }

    const qsizetype frameLengthBits = representativeFrameLengthBits(candidate);
    const qsizetype coveredBitCount = totalBitCount - candidateStartBit;
    return qMax<qsizetype>(1, (coveredBitCount + frameLengthBits - 1) / frameLengthBits);
}

QString rowOrderButtonLabel(const QString& labelText, bool descending) {
    return QStringLiteral("%1 %2")
        .arg(labelText)
        .arg(descending ? QStringLiteral("↓") : QStringLiteral("↑"));
}

}  // namespace

BitstreamSyncDiscoveryDialog::BitstreamSyncDiscoveryDialog(data::ByteDataSource* dataSource, QWidget* parent)
    : QDialog(parent),
      dataSource_(dataSource),
      previewFrameLayout_(new features::framing::FrameLayout()) {
    qRegisterMetaType<features::bitstream_sync_discovery::BitstreamSyncDiscoveryProgressUpdate>();
    qRegisterMetaType<features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate>();
    qRegisterMetaType<features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidateList>();

    setModal(true);
    setWindowTitle(QStringLiteral("Bitstream Sync Discovery"));
    resize(1280, 760);
    buildLayout();
    setScanningState(false);
}

BitstreamSyncDiscoveryDialog::~BitstreamSyncDiscoveryDialog() {
    stopWorkerThread();
    delete previewFrameLayout_;
}

std::optional<features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate>
BitstreamSyncDiscoveryDialog::selectedCandidate() const {
    const int currentRow = resultsTableWidget_ != nullptr ? resultsTableWidget_->currentRow() : -1;
    if (currentRow < 0 || currentRow >= candidates_.size()) {
        return std::nullopt;
    }

    return candidates_.at(currentRow);
}

void BitstreamSyncDiscoveryDialog::startDiscovery() {
    if (dataSource_ == nullptr || !dataSource_->hasData()) {
        progressPhaseLabel_->setVisible(true);
        progressDetailLabel_->setVisible(true);
        progressPhaseLabel_->setText(QStringLiteral("No file loaded"));
        progressDetailLabel_->setText(QStringLiteral("Load a file before running bitstream sync discovery."));
        progressBar_->setValue(0);
        return;
    }

    stopWorkerThread();
    candidates_.clear();
    appliedCandidateIndex_ = -1;
    previewedCandidateKey_.clear();
    resultsTableWidget_->setRowCount(0);
    clearPreview();

    features::bitstream_sync_discovery::BitstreamSyncDiscoverySettings settings;
    settings.minimumPatternBits = minimumBitsSpinBox_->value();
    settings.maximumPatternBits = maximumBitsSpinBox_->value();
    settings.minimumMatchCount = minimumMatchesSpinBox_->value();
    settings.maximumCandidatesPerWidth = candidatesPerWidthSpinBox_->value();
    settings.minimumExpectedFrameBits = minimumFrameBitsSpinBox_->value();
    settings.entropyThreshold = entropyThresholdSpinBox_->value();
    settings.maximumGapCoefficientVariation = maximumGapCvSpinBox_->value();
    settings.searchEffort = static_cast<features::bitstream_sync_discovery::BitstreamSyncDiscoverySearchEffort>(
        searchEffortComboBox_->currentData().toInt()
    );

    workerThread_ = new QThread(this);
    worker_ = new features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker(dataSource_, settings);
    worker_->moveToThread(workerThread_);

    connect(workerThread_, &QThread::started, worker_, &features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker::start);
    connect(worker_, &features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker::progressChanged,
            this, &BitstreamSyncDiscoveryDialog::handleProgressChanged);
    connect(worker_, &features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker::partialResultsReady,
            this, &BitstreamSyncDiscoveryDialog::handlePartialResultsReady);
    connect(worker_, &features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker::finished,
            this, &BitstreamSyncDiscoveryDialog::handleDiscoveryFinished);
    connect(worker_, &features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker::failed,
            this, &BitstreamSyncDiscoveryDialog::handleDiscoveryFailed);
    connect(worker_, &features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker::canceled,
            this, &BitstreamSyncDiscoveryDialog::handleDiscoveryCanceled);

    connect(worker_, &features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker::finished,
            workerThread_, &QThread::quit);
    connect(worker_, &features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker::failed,
            workerThread_, &QThread::quit);
    connect(worker_, &features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker::canceled,
            workerThread_, &QThread::quit);
    setScanningState(true);
    progressPhaseLabel_->setVisible(true);
    progressDetailLabel_->setVisible(true);
    progressPhaseLabel_->setText(QStringLiteral("Bitstream Sync Discovery"));
    progressDetailLabel_->setText(QStringLiteral("Scanning bitstream candidates..."));
    progressBar_->setValue(0);
    workerThread_->start();
}

void BitstreamSyncDiscoveryDialog::cancelDiscovery() {
    if (worker_ != nullptr) {
        worker_->cancel();
    }
}

void BitstreamSyncDiscoveryDialog::handleProgressChanged(
    const features::bitstream_sync_discovery::BitstreamSyncDiscoveryProgressUpdate& progressUpdate
) {
    progressPhaseLabel_->setVisible(true);
    progressDetailLabel_->setVisible(true);
    progressPhaseLabel_->setText(progressUpdate.phaseLabel);
    progressDetailLabel_->setText(progressUpdate.detailLabel);
    progressBar_->setValue(qBound(0, progressUpdate.percentComplete, 100));
}

void BitstreamSyncDiscoveryDialog::handlePartialResultsReady(
    const features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidateList& candidates
) {
    const QString preferredCandidateKey = selectedCandidate().has_value()
        ? candidateKey(*selectedCandidate())
        : QString();
    candidates_ = candidates;
    refreshResultsTable(preferredCandidateKey);
}

void BitstreamSyncDiscoveryDialog::handleDiscoveryFinished(
    const features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidateList& candidates
) {
    candidates_ = candidates;
    refreshResultsTable();
    setScanningState(false);

    progressPhaseLabel_->setVisible(true);
    progressDetailLabel_->setVisible(true);
    progressPhaseLabel_->setText(QStringLiteral("Discovery complete"));
    progressDetailLabel_->setText(
        QStringLiteral("Prepared %1 ranked candidate%2")
            .arg(candidates_.size())
            .arg(candidates_.size() == 1 ? QString() : QStringLiteral("s"))
    );
    progressBar_->setValue(100);

    if (resultsTableWidget_->rowCount() > 0 && resultsTableWidget_->currentRow() < 0) {
        resultsTableWidget_->selectRow(0);
    }

    stopWorkerThread();
}

void BitstreamSyncDiscoveryDialog::handleDiscoveryFailed(const QString& errorMessage) {
    setScanningState(false);
    progressPhaseLabel_->setVisible(true);
    progressDetailLabel_->setVisible(true);
    progressPhaseLabel_->setText(QStringLiteral("Discovery failed"));
    progressDetailLabel_->setText(errorMessage);
    progressBar_->setValue(0);
    if (candidates_.isEmpty()) {
        clearPreview();
    }
    stopWorkerThread();
}

void BitstreamSyncDiscoveryDialog::handleDiscoveryCanceled() {
    setScanningState(false);
    progressPhaseLabel_->setVisible(true);
    progressDetailLabel_->setVisible(true);
    progressPhaseLabel_->setText(QStringLiteral("Discovery canceled"));
    progressDetailLabel_->setText(QStringLiteral("Bitstream sync discovery was canceled."));
    progressBar_->setValue(0);
    stopWorkerThread();
}

void BitstreamSyncDiscoveryDialog::updatePreviewForSelection() {
    updatePreviewForCandidateIndex(resultsTableWidget_->currentRow());
}

void BitstreamSyncDiscoveryDialog::applySelectedCandidate() {
    const int currentRow = resultsTableWidget_->currentRow();
    if (currentRow < 0 || currentRow >= candidates_.size()) {
        return;
    }

    appliedCandidateIndex_ = currentRow;
    accept();
}

void BitstreamSyncDiscoveryDialog::buildLayout() {
    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(10, 10, 10, 10);
    rootLayout->setSpacing(8);

    QHBoxLayout* controlsLayout = new QHBoxLayout();
    controlsLayout->setSpacing(8);
    controlsLayout->addWidget(new QLabel(QStringLiteral("Min bits:"), this));

    minimumBitsSpinBox_ = new QSpinBox(this);
    minimumBitsSpinBox_->setRange(1, features::bitstream_sync_discovery::kMaximumSupportedSyncPatternBits);
    minimumBitsSpinBox_->setValue(8);
    controlsLayout->addWidget(minimumBitsSpinBox_);

    controlsLayout->addWidget(new QLabel(QStringLiteral("Max bits:"), this));
    maximumBitsSpinBox_ = new QSpinBox(this);
    maximumBitsSpinBox_->setRange(1, features::bitstream_sync_discovery::kMaximumSupportedSyncPatternBits);
    maximumBitsSpinBox_->setValue(50);
    controlsLayout->addWidget(maximumBitsSpinBox_);

    controlsLayout->addWidget(new QLabel(QStringLiteral("Min matches:"), this));
    minimumMatchesSpinBox_ = new QSpinBox(this);
    minimumMatchesSpinBox_->setRange(3, 1024);
    minimumMatchesSpinBox_->setValue(3);
    controlsLayout->addWidget(minimumMatchesSpinBox_);

    controlsLayout->addWidget(new QLabel(QStringLiteral("Top / Width:"), this));
    candidatesPerWidthSpinBox_ = new QSpinBox(this);
    candidatesPerWidthSpinBox_->setRange(1, 32);
    candidatesPerWidthSpinBox_->setValue(3);
    controlsLayout->addWidget(candidatesPerWidthSpinBox_);

    controlsLayout->addWidget(new QLabel(QStringLiteral("Search:"), this));
    searchEffortComboBox_ = new QComboBox(this);
    searchEffortComboBox_->addItem(
        QStringLiteral("Fast"),
        static_cast<int>(features::bitstream_sync_discovery::BitstreamSyncDiscoverySearchEffort::Fast)
    );
    searchEffortComboBox_->addItem(
        QStringLiteral("Balanced"),
        static_cast<int>(features::bitstream_sync_discovery::BitstreamSyncDiscoverySearchEffort::Balanced)
    );
    searchEffortComboBox_->addItem(
        QStringLiteral("Exhaustive"),
        static_cast<int>(features::bitstream_sync_discovery::BitstreamSyncDiscoverySearchEffort::Exhaustive)
    );
    searchEffortComboBox_->setCurrentIndex(1);
    controlsLayout->addWidget(searchEffortComboBox_);

    controlsLayout->addStretch();

    discoverButton_ = new QPushButton(QStringLiteral("Discover"), this);
    connect(discoverButton_, &QPushButton::clicked, this, &BitstreamSyncDiscoveryDialog::startDiscovery);
    controlsLayout->addWidget(discoverButton_);

    cancelDiscoveryButton_ = new QPushButton(QStringLiteral("Cancel Search"), this);
    connect(cancelDiscoveryButton_, &QPushButton::clicked, this, &BitstreamSyncDiscoveryDialog::cancelDiscovery);
    controlsLayout->addWidget(cancelDiscoveryButton_);

    rootLayout->addLayout(controlsLayout);

    QHBoxLayout* advancedControlsLayout = new QHBoxLayout();
    advancedControlsLayout->setSpacing(8);
    advancedControlsLayout->addWidget(new QLabel(QStringLiteral("Min Frame Bits:"), this));

    minimumFrameBitsSpinBox_ = new QSpinBox(this);
    minimumFrameBitsSpinBox_->setRange(1, 524288);
    minimumFrameBitsSpinBox_->setValue(64);
    advancedControlsLayout->addWidget(minimumFrameBitsSpinBox_);

    advancedControlsLayout->addWidget(new QLabel(QStringLiteral("Entropy TH:"), this));
    entropyThresholdSpinBox_ = new QDoubleSpinBox(this);
    entropyThresholdSpinBox_->setRange(0.05, 0.95);
    entropyThresholdSpinBox_->setDecimals(2);
    entropyThresholdSpinBox_->setSingleStep(0.05);
    entropyThresholdSpinBox_->setValue(0.30);
    advancedControlsLayout->addWidget(entropyThresholdSpinBox_);

    advancedControlsLayout->addWidget(new QLabel(QStringLiteral("Max Gap CV:"), this));
    maximumGapCvSpinBox_ = new QDoubleSpinBox(this);
    maximumGapCvSpinBox_->setRange(0.10, 10.0);
    maximumGapCvSpinBox_->setDecimals(2);
    maximumGapCvSpinBox_->setSingleStep(0.10);
    maximumGapCvSpinBox_->setValue(2.0);
    advancedControlsLayout->addWidget(maximumGapCvSpinBox_);

    advancedControlsLayout->addStretch();
    rootLayout->addLayout(advancedControlsLayout);

    progressPhaseLabel_ = new QLabel(QStringLiteral("Bitstream Sync Discovery"), this);
    progressPhaseLabel_->setVisible(false);
    rootLayout->addWidget(progressPhaseLabel_);

    progressDetailLabel_ = new QLabel(
        QStringLiteral("Choose the search range and v2 framing filters, then click Discover."),
        this
    );
    progressDetailLabel_->setWordWrap(true);
    progressDetailLabel_->setVisible(false);
    rootLayout->addWidget(progressDetailLabel_);

    progressBar_ = new QProgressBar(this);
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    rootLayout->addWidget(progressBar_);

    QSplitter* contentSplitter = new QSplitter(Qt::Horizontal, this);

    resultsTableWidget_ = new QTableWidget(this);
    resultsTableWidget_->setColumnCount(9);
    resultsTableWidget_->setHorizontalHeaderLabels({
        QStringLiteral("Rank"),
        QStringLiteral("Pattern"),
        QStringLiteral("Bits"),
        QStringLiteral("Matches"),
        QStringLiteral("Frames"),
        QStringLiteral("Median Bytes"),
        QStringLiteral("Groups"),
        QStringLiteral("Cliffs"),
        QStringLiteral("Score")
    });
    resultsTableWidget_->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
    resultsTableWidget_->setSelectionMode(QAbstractItemView::SelectionMode::SingleSelection);
    resultsTableWidget_->setEditTriggers(QAbstractItemView::EditTrigger::NoEditTriggers);
    resultsTableWidget_->setAlternatingRowColors(true);
    resultsTableWidget_->setMinimumWidth(620);
    resultsTableWidget_->verticalHeader()->setVisible(false);
    resultsTableWidget_->horizontalHeader()->setStretchLastSection(false);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kRankColumn, QHeaderView::ResizeMode::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kPatternColumn, QHeaderView::ResizeMode::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kBitsColumn, QHeaderView::ResizeMode::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kMatchesColumn, QHeaderView::ResizeMode::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kFramesColumn, QHeaderView::ResizeMode::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kMedianBytesColumn, QHeaderView::ResizeMode::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kGroupsColumn, QHeaderView::ResizeMode::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kCliffsColumn, QHeaderView::ResizeMode::ResizeToContents);
    resultsTableWidget_->horizontalHeader()->setSectionResizeMode(kScoreColumn, QHeaderView::ResizeMode::ResizeToContents);
    connect(resultsTableWidget_, &QTableWidget::itemSelectionChanged, this, &BitstreamSyncDiscoveryDialog::updatePreviewForSelection);
    contentSplitter->addWidget(resultsTableWidget_);

    QWidget* previewWidget = new QWidget(this);
    QVBoxLayout* previewLayout = new QVBoxLayout(previewWidget);
    previewLayout->setContentsMargins(0, 0, 0, 0);
    previewLayout->setSpacing(8);

    QHBoxLayout* previewOrderLayout = new QHBoxLayout();
    previewOrderLayout->addWidget(new QLabel(QStringLiteral("Row Order:"), previewWidget));
    previewChronologicalOrderButton_ = new QPushButton(previewWidget);
    connect(previewChronologicalOrderButton_, &QPushButton::clicked, this, &BitstreamSyncDiscoveryDialog::cyclePreviewChronologicalOrder);
    previewOrderLayout->addWidget(previewChronologicalOrderButton_);
    previewLengthOrderButton_ = new QPushButton(previewWidget);
    connect(previewLengthOrderButton_, &QPushButton::clicked, this, &BitstreamSyncDiscoveryDialog::cyclePreviewLengthOrder);
    previewOrderLayout->addWidget(previewLengthOrderButton_);
    previewOrderLayout->addStretch();
    previewLayout->addLayout(previewOrderLayout);

    previewSummaryLabel_ = new QLabel(QStringLiteral("Top-ranked candidate preview will appear here."), previewWidget);
    previewSummaryLabel_->setWordWrap(true);
    previewLayout->addWidget(previewSummaryLabel_);

    QScrollArea* previewScrollArea = new QScrollArea(previewWidget);
    previewScrollArea->setWidgetResizable(false);
    previewScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    previewScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    previewBitViewer_ = new LiveBitViewerWidget(previewScrollArea);
    previewBitViewer_->setCellSize(14);
    previewBitViewer_->setMinimumHeight(520);
    previewScrollArea->setWidget(previewBitViewer_);
    previewLayout->addWidget(previewScrollArea, 1);

    contentSplitter->addWidget(previewWidget);
    contentSplitter->setChildrenCollapsible(false);
    contentSplitter->setStretchFactor(0, 0);
    contentSplitter->setStretchFactor(1, 1);
    contentSplitter->setSizes({560, 620});
    rootLayout->addWidget(contentSplitter, 1);

    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();

    applyButton_ = new QPushButton(QStringLiteral("Apply"), this);
    connect(applyButton_, &QPushButton::clicked, this, &BitstreamSyncDiscoveryDialog::applySelectedCandidate);
    buttonLayout->addWidget(applyButton_);

    QPushButton* closeButton = new QPushButton(QStringLiteral("Close"), this);
    connect(closeButton, &QPushButton::clicked, this, &QDialog::reject);
    buttonLayout->addWidget(closeButton);

    rootLayout->addLayout(buttonLayout);

    updatePreviewRowOrderButtons();
}

void BitstreamSyncDiscoveryDialog::setScanningState(bool isScanning) {
    minimumBitsSpinBox_->setEnabled(!isScanning);
    maximumBitsSpinBox_->setEnabled(!isScanning);
    minimumMatchesSpinBox_->setEnabled(!isScanning);
    candidatesPerWidthSpinBox_->setEnabled(!isScanning);
    searchEffortComboBox_->setEnabled(!isScanning);
    minimumFrameBitsSpinBox_->setEnabled(!isScanning);
    entropyThresholdSpinBox_->setEnabled(!isScanning);
    maximumGapCvSpinBox_->setEnabled(!isScanning);
    discoverButton_->setEnabled(!isScanning);
    cancelDiscoveryButton_->setEnabled(isScanning);
    applyButton_->setEnabled(!isScanning && resultsTableWidget_->currentRow() >= 0);
}

void BitstreamSyncDiscoveryDialog::stopWorkerThread() {
    if (worker_ != nullptr) {
        worker_->cancel();
    }

    if (workerThread_ != nullptr) {
        workerThread_->quit();
        workerThread_->wait();
        delete worker_;
        delete workerThread_;
    }

    worker_ = nullptr;
    workerThread_ = nullptr;
}

void BitstreamSyncDiscoveryDialog::refreshResultsTable(const QString& preferredCandidateKey) {
    QSignalBlocker selectionBlocker(resultsTableWidget_);

    resultsTableWidget_->setRowCount(candidates_.size());
    int candidateRowToSelect = candidates_.isEmpty() ? -1 : 0;

    for (int candidateIndex = 0; candidateIndex < candidates_.size(); ++candidateIndex) {
        const features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidate = candidates_.at(candidateIndex);

        if (!preferredCandidateKey.isEmpty() && candidateKey(candidate) == preferredCandidateKey) {
            candidateRowToSelect = candidateIndex;
        }

        auto makeItem = [](const QString& text) {
            QTableWidgetItem* item = new QTableWidgetItem(text);
            item->setTextAlignment(Qt::AlignCenter);
            return item;
        };

        resultsTableWidget_->setItem(candidateIndex, 0, makeItem(QString::number(candidateIndex + 1)));
        resultsTableWidget_->setItem(candidateIndex, 1, makeItem(candidate.displayPattern));
        resultsTableWidget_->setItem(candidateIndex, 2, makeItem(QString::number(candidate.refinedPattern.bitWidth)));
        resultsTableWidget_->setItem(candidateIndex, 3, makeItem(QString::number(candidate.matchStartBits.size())));
        resultsTableWidget_->setItem(candidateIndex, 4, makeItem(QString::number(candidate.frameSpans.size())));
        resultsTableWidget_->setItem(
            candidateIndex,
            5,
            makeItem(QString::number((candidate.frameLengthSummary.medianLengthBits + 7) / 8))
        );
        resultsTableWidget_->setItem(
            candidateIndex,
            6,
            makeItem(QString::number(candidate.detectedGroupCount))
        );
        resultsTableWidget_->setItem(
            candidateIndex,
            7,
            makeItem(QString::number(candidate.detectedEntropyCliffCount))
        );
        resultsTableWidget_->setItem(
            candidateIndex,
            8,
            makeItem(QString::number(candidate.confidenceScore, 'f', 1))
        );
    }

    if (candidateRowToSelect >= 0) {
        resultsTableWidget_->selectRow(candidateRowToSelect);
        resultsTableWidget_->setCurrentCell(candidateRowToSelect, 0);
    }

    applyButton_->setEnabled(resultsTableWidget_->currentRow() >= 0);

    if (candidateRowToSelect >= 0) {
        const QString selectedCandidateKey = candidateKey(candidates_.at(candidateRowToSelect));
        if (selectedCandidateKey != previewedCandidateKey_) {
            updatePreviewForCandidateIndex(candidateRowToSelect);
        }
    } else {
        clearPreview();
    }
}

void BitstreamSyncDiscoveryDialog::updatePreviewForCandidateIndex(int candidateIndex) {
    if (candidateIndex < 0 || candidateIndex >= candidates_.size()) {
        clearPreview();
        return;
    }

    const features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidate = candidates_.at(candidateIndex);

    const QVector<features::framing::FrameSpan> previewFrameSpans = previewFrameSpansForCandidate(candidate);
    previewFrameLayout_->setFrames(previewFrameSpans);

    const int previewFrameBitWidth = qMax(
        candidate.refinedPattern.bitWidth,
        static_cast<int>(candidate.frameLengthSummary.maximumLengthBits)
    );
    const QVector<features::columns::VisibleByteColumn> syncPreviewColumns =
        previewColumnsForBitRange(0, previewFrameBitWidth - 1);
    const int previewRowCount = previewFrameSpans.size();
    previewBitViewer_->setPreviewSource(
        dataSource_,
        previewFrameLayout_,
        syncPreviewColumns,
        0,
        previewRowCount
    );

    previewSummaryLabel_->setText(candidateSummaryText(candidate, previewRowCount));
    previewSummaryLabel_->setVisible(true);
    previewedCandidateKey_ = candidateKey(candidate);
    applyButton_->setEnabled(true);
}

void BitstreamSyncDiscoveryDialog::clearPreview() {
    previewFrameLayout_->clearFrame();
    previewBitViewer_->setPreviewSource(nullptr, nullptr, {}, 0, 0);
    previewSummaryLabel_->setText(QStringLiteral("Select a candidate to preview its framing and sync column."));
    previewSummaryLabel_->setVisible(true);
    previewedCandidateKey_.clear();
    applyButton_->setEnabled(false);
}

void BitstreamSyncDiscoveryDialog::cyclePreviewChronologicalOrder() {
    const bool descending = previewFrameLayout_->rowOrderMode() == features::framing::FrameLayout::RowOrderMode::Chronological
        ? !previewFrameLayout_->rowOrderDescending()
        : previewChronologicalDescending_;
    previewChronologicalDescending_ = descending;
    applyPreviewRowOrder(features::framing::FrameLayout::RowOrderMode::Chronological, descending);
}

void BitstreamSyncDiscoveryDialog::cyclePreviewLengthOrder() {
    const bool descending = previewFrameLayout_->rowOrderMode() == features::framing::FrameLayout::RowOrderMode::Length
        ? !previewFrameLayout_->rowOrderDescending()
        : previewLengthDescending_;
    previewLengthDescending_ = descending;
    applyPreviewRowOrder(features::framing::FrameLayout::RowOrderMode::Length, descending);
}

void BitstreamSyncDiscoveryDialog::applyPreviewRowOrder(
    features::framing::FrameLayout::RowOrderMode rowOrderMode,
    bool descending
) {
    previewFrameLayout_->setRowOrder(rowOrderMode, descending);
    updatePreviewRowOrderButtons();
    const int currentRow = resultsTableWidget_ != nullptr ? resultsTableWidget_->currentRow() : -1;
    if (currentRow >= 0 && currentRow < candidates_.size()) {
        updatePreviewForCandidateIndex(currentRow);
    } else if (previewBitViewer_ != nullptr) {
        previewBitViewer_->update();
    }
}

void BitstreamSyncDiscoveryDialog::updatePreviewRowOrderButtons() {
    if (previewChronologicalOrderButton_ != nullptr) {
        previewChronologicalOrderButton_->setText(
            rowOrderButtonLabel(QStringLiteral("Chronological"), previewChronologicalDescending_)
        );
        previewChronologicalOrderButton_->setDefault(
            previewFrameLayout_ != nullptr
            && previewFrameLayout_->rowOrderMode() == features::framing::FrameLayout::RowOrderMode::Chronological
        );
    }
    if (previewLengthOrderButton_ != nullptr) {
        previewLengthOrderButton_->setText(
            rowOrderButtonLabel(QStringLiteral("Frame Size"), previewLengthDescending_)
        );
        previewLengthOrderButton_->setDefault(
            previewFrameLayout_ != nullptr
            && previewFrameLayout_->rowOrderMode() == features::framing::FrameLayout::RowOrderMode::Length
        );
    }
}

QString BitstreamSyncDiscoveryDialog::candidateKey(
    const features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidate
) const {
    return QStringLiteral("%1|%2|%3")
        .arg(candidate.displayPattern)
        .arg(candidate.refinedPattern.bitWidth)
        .arg(candidate.matchStartBits.isEmpty() ? -1 : candidate.matchStartBits.first());
}

QVector<features::framing::FrameSpan> BitstreamSyncDiscoveryDialog::previewFrameSpansForCandidate(
    const features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidate
) const {
    QVector<features::framing::FrameSpan> previewFrameSpans;
    if (candidate.frameSpans.isEmpty()) {
        return previewFrameSpans;
    }

    qsizetype maximumDetectedFrameCount = 0;
    for (const features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidateEntry : candidates_) {
        maximumDetectedFrameCount = qMax(maximumDetectedFrameCount, static_cast<qsizetype>(candidateEntry.frameSpans.size()));
    }
    const double relativeFrameSupport = maximumDetectedFrameCount > 0
        ? static_cast<double>(candidate.frameSpans.size()) / static_cast<double>(maximumDetectedFrameCount)
        : 1.0;
    const qsizetype totalBitCount = dataSource_ != nullptr ? dataSource_->bitCount() : 0;
    const qsizetype estimatedFrameCount = estimatedFrameCountFromMedianLength(candidate, totalBitCount);
    const double expectedFrameCoverage = estimatedFrameCount > 0
        ? static_cast<double>(candidate.frameSpans.size()) / static_cast<double>(estimatedFrameCount)
        : 1.0;
    if (relativeFrameSupport >= 0.85 || expectedFrameCoverage >= 0.80) {
        return candidate.frameSpans;
    }

    const qsizetype frameLengthBits = representativeFrameLengthBits(candidate);
    const qsizetype splitThresholdBits = frameLengthBits + (frameLengthBits / 2);

    previewFrameSpans.reserve(candidate.frameSpans.size() * 2);
    for (const features::framing::FrameSpan& detectedFrameSpan : candidate.frameSpans) {
        if (detectedFrameSpan.lengthBits <= frameLengthBits || frameLengthBits <= 0) {
            previewFrameSpans.append(detectedFrameSpan);
            continue;
        }

        qsizetype remainingLengthBits = detectedFrameSpan.lengthBits;
        qsizetype currentStartBit = detectedFrameSpan.startBit;
        while (remainingLengthBits > splitThresholdBits) {
            features::framing::FrameSpan previewFrameSpan;
            previewFrameSpan.startBit = currentStartBit;
            previewFrameSpan.lengthBits = frameLengthBits;
            previewFrameSpans.append(previewFrameSpan);

            currentStartBit += frameLengthBits;
            remainingLengthBits -= frameLengthBits;
        }

        features::framing::FrameSpan finalPreviewFrameSpan;
        finalPreviewFrameSpan.startBit = currentStartBit;
        finalPreviewFrameSpan.lengthBits = remainingLengthBits;
        previewFrameSpans.append(finalPreviewFrameSpan);
    }

    return previewFrameSpans;
}

QString BitstreamSyncDiscoveryDialog::candidateSummaryText(
    const features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidate,
    int previewFrameCount
) const {
    const QString previewLabel = previewFrameCount != candidate.frameSpans.size()
        ? QStringLiteral("Preview: expanded sparse candidate from %1 detected frames into %2 preview rows using median frame size\n")
              .arg(candidate.frameSpans.size())
              .arg(previewFrameCount)
        : QStringLiteral("Preview: scroll to inspect all %1 framed rows\n")
              .arg(candidate.frameSpans.size());

    return previewLabel + QStringLiteral(
               "Pattern: %1 | Tail: %2 | Width: %3 bits | Matches: %4 | Frames: %5 | Groups: %6 | Cliffs: %7\n"
               "Frame lengths: min %8 | median %9 | max %10 | Class: %11 | Order: %12\n"
               "Spacing %13% | Cliff %14% | Unique %15% | Distributed %16% | Shared Header %17% | Boost %18 | Score %19")
        .arg(candidate.displayHexPattern.isEmpty() ? candidate.displayPattern : candidate.displayHexPattern)
        .arg(candidate.trailingBitsText.isEmpty() ? QStringLiteral("-") : candidate.trailingBitsText)
        .arg(candidate.refinedPattern.bitWidth)
        .arg(candidate.matchStartBits.size())
        .arg(candidate.frameSpans.size())
        .arg(candidate.detectedGroupCount)
        .arg(candidate.detectedEntropyCliffCount)
        .arg(bitLengthLabel(candidate.frameLengthSummary.minimumLengthBits))
        .arg(bitLengthLabel(candidate.frameLengthSummary.medianLengthBits))
        .arg(bitLengthLabel(candidate.frameLengthSummary.maximumLengthBits))
        .arg(candidate.protocolClassification.isEmpty() ? QStringLiteral("-") : candidate.protocolClassification)
        .arg(
            previewFrameLayout_->rowOrderMode() == features::framing::FrameLayout::RowOrderMode::Length
                ? QStringLiteral("frame size %1").arg(previewFrameLayout_->rowOrderDescending() ? QStringLiteral("desc") : QStringLiteral("asc"))
                : QStringLiteral("chronological %1").arg(previewFrameLayout_->rowOrderDescending() ? QStringLiteral("desc") : QStringLiteral("asc"))
        )
        .arg(QString::number(candidate.spacingRegularityScore * 100.0, 'f', 1))
        .arg(QString::number(candidate.cliffSharpnessScore * 100.0, 'f', 1))
        .arg(QString::number(candidate.patternUniquenessScore * 100.0, 'f', 1))
        .arg(QString::number(candidate.distributedConstantsScore * 100.0, 'f', 1))
        .arg(QString::number(candidate.crossGroupAgreementScore * 100.0, 'f', 1))
        .arg(QString::number(candidate.validationBoostScore, 'f', 1))
        .arg(QString::number(candidate.confidenceScore, 'f', 1));
}

QVector<features::columns::VisibleByteColumn> BitstreamSyncDiscoveryDialog::previewColumnsForBitRange(
    int startBit,
    int endBit
) const {
    QVector<features::columns::VisibleByteColumn> previewColumns;
    if (startBit > endBit) {
        return previewColumns;
    }

    int segmentStartBit = startBit;
    while (segmentStartBit <= endBit) {
        const int segmentEndBit = qMin(endBit, (((segmentStartBit / 8) + 1) * 8) - 1);

        features::columns::VisibleByteColumn visibleColumn;
        visibleColumn.byteIndex = segmentStartBit / 8;
        visibleColumn.byteEndIndex = segmentEndBit / 8;
        visibleColumn.bitStart = segmentStartBit % 8;
        visibleColumn.bitEnd = segmentEndBit % 8;
        visibleColumn.absoluteStartBit = segmentStartBit;
        visibleColumn.absoluteEndBit = segmentEndBit;
        previewColumns.append(visibleColumn);

        segmentStartBit = segmentEndBit + 1;
    }

    return previewColumns;
}

}  // namespace bitabyte::ui
