#include "ui/bitstream_sync_discovery_dialog.h"

#include "data/byte_data_source.h"
#include "features/classification/frame_field_classification.h"
#include "features/bitstream_sync_discovery/bitstream_sync_discovery_worker.h"
#include "features/columns/visible_byte_column.h"
#include "features/framing/frame_layout.h"
#include "ui/frame_field_hints_panel.h"
#include "ui/live_bit_viewer_widget.h"

#include <QtConcurrent>

#include <QAbstractItemView>
#include <QComboBox>
#include <QColor>
#include <QDoubleSpinBox>
#include <QFutureWatcher>
#include <QGroupBox>
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
constexpr int kMaximumDiscoveryPreviewRows = 128;

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

const QColor kCounterHighlightColor(212, 232, 255);
const QColor kConstantHighlightColor(255, 247, 196);

QString previewFieldLabel(const features::columns::VisibleByteColumn& visibleColumn) {
    const bool startsOnByteBoundary = visibleColumn.absoluteStartBit % 8 == 0;
    const bool endsOnByteBoundary = visibleColumn.absoluteEndBit % 8 == 7;
    const int startByteIndex = visibleColumn.absoluteStartBit / 8;
    const int endByteIndex = visibleColumn.absoluteEndBit / 8;

    if (startsOnByteBoundary && endsOnByteBoundary) {
        return startByteIndex == endByteIndex
            ? QStringLiteral("Byte %1").arg(startByteIndex)
            : QStringLiteral("Bytes %1-%2").arg(startByteIndex).arg(endByteIndex);
    }

    return QStringLiteral("Bits %1-%2")
        .arg(visibleColumn.absoluteStartBit)
        .arg(visibleColumn.absoluteEndBit);
}

QVector<LiveBitViewerWidget::PreviewBitHighlight> previewBitHighlightRanges(
    const features::classification::FrameFieldClassificationResult& classificationResult
) {
    QVector<LiveBitViewerWidget::PreviewBitHighlight> previewBitHighlights;
    previewBitHighlights.reserve(
        classificationResult.constantHints.size() + classificationResult.counterHints.size()
    );

    for (const features::classification::FrameFieldHint& constantHint : classificationResult.constantHints) {
        LiveBitViewerWidget::PreviewBitHighlight bitHighlight;
        bitHighlight.absoluteStartBit = constantHint.absoluteStartBit;
        bitHighlight.absoluteEndBit = constantHint.absoluteEndBit;
        bitHighlight.color = kConstantHighlightColor;
        previewBitHighlights.append(bitHighlight);
    }
    for (const features::classification::FrameFieldHint& counterHint : classificationResult.counterHints) {
        LiveBitViewerWidget::PreviewBitHighlight bitHighlight;
        bitHighlight.absoluteStartBit = counterHint.absoluteStartBit;
        bitHighlight.absoluteEndBit = counterHint.absoluteEndBit;
        bitHighlight.color = kCounterHighlightColor;
        previewBitHighlights.append(bitHighlight);
    }

    std::sort(
        previewBitHighlights.begin(),
        previewBitHighlights.end(),
        [](const LiveBitViewerWidget::PreviewBitHighlight& leftHighlight,
           const LiveBitViewerWidget::PreviewBitHighlight& rightHighlight) {
            if (leftHighlight.absoluteStartBit != rightHighlight.absoluteStartBit) {
                return leftHighlight.absoluteStartBit < rightHighlight.absoluteStartBit;
            }
            return leftHighlight.absoluteEndBit < rightHighlight.absoluteEndBit;
        }
    );
    return previewBitHighlights;
}

QVector<features::framing::FrameSpan> limitedPreviewFrameSpans(
    const QVector<features::framing::FrameSpan>& previewFrameSpans,
    features::framing::FrameLayout::RowOrderMode rowOrderMode,
    bool descending
) {
    if (previewFrameSpans.size() <= kMaximumDiscoveryPreviewRows) {
        return previewFrameSpans;
    }

    QVector<int> orderedIndices;
    orderedIndices.reserve(previewFrameSpans.size());
    for (int frameIndex = 0; frameIndex < previewFrameSpans.size(); ++frameIndex) {
        orderedIndices.append(frameIndex);
    }

    std::stable_sort(
        orderedIndices.begin(),
        orderedIndices.end(),
        [&previewFrameSpans, rowOrderMode, descending](int leftIndex, int rightIndex) {
            if (leftIndex < 0 || leftIndex >= previewFrameSpans.size()
                || rightIndex < 0 || rightIndex >= previewFrameSpans.size()) {
                return leftIndex < rightIndex;
            }

            const features::framing::FrameSpan& leftFrameSpan = previewFrameSpans.at(leftIndex);
            const features::framing::FrameSpan& rightFrameSpan = previewFrameSpans.at(rightIndex);
            if (rowOrderMode == features::framing::FrameLayout::RowOrderMode::Length
                && leftFrameSpan.lengthBits != rightFrameSpan.lengthBits) {
                return descending
                    ? leftFrameSpan.lengthBits > rightFrameSpan.lengthBits
                    : leftFrameSpan.lengthBits < rightFrameSpan.lengthBits;
            }

            return descending ? leftIndex > rightIndex : leftIndex < rightIndex;
        }
    );

    QVector<features::framing::FrameSpan> limitedFrameSpans;
    limitedFrameSpans.reserve(kMaximumDiscoveryPreviewRows);
    QVector<int> selectedIndices;
    selectedIndices.reserve(kMaximumDiscoveryPreviewRows);
    for (int orderedIndex = 0; orderedIndex < kMaximumDiscoveryPreviewRows; ++orderedIndex) {
        selectedIndices.append(orderedIndices.at(orderedIndex));
    }
    std::sort(selectedIndices.begin(), selectedIndices.end());
    for (int selectedIndex : selectedIndices) {
        limitedFrameSpans.append(previewFrameSpans.at(selectedIndex));
    }
    return limitedFrameSpans;
}

features::classification::FrameFieldClassificationResult analyzePreviewFields(
    const data::ByteDataSource& dataSource,
    const features::framing::FrameLayout& frameLayout,
    const QVector<features::classification::FrameFieldColumnSnapshot>& columnSnapshots,
    int universalEndBit
) {
    features::classification::FrameFieldClassificationResult previewClassificationResult =
        features::classification::classifyFramedVisibleColumns(
            dataSource,
            frameLayout,
            columnSnapshots
        );
    if (universalEndBit < 0) {
        return previewClassificationResult;
    }

    const QVector<features::classification::FrameFieldHint> discoveredCounterHints =
        features::classification::discoverCounterHintsByBitWindow(
            dataSource,
            frameLayout,
            columnSnapshots,
            0,
            universalEndBit
        );
    for (const features::classification::FrameFieldHint& counterHint : discoveredCounterHints) {
        bool alreadyCovered = false;
        for (const features::classification::FrameFieldHint& existingHint : previewClassificationResult.counterHints) {
            if (!(counterHint.absoluteEndBit < existingHint.absoluteStartBit
                || existingHint.absoluteEndBit < counterHint.absoluteStartBit)) {
                alreadyCovered = true;
                break;
            }
        }
        if (alreadyCovered) {
            continue;
        }

        previewClassificationResult.counterHints.append(counterHint);
        for (int visibleColumnIndex : counterHint.visibleColumnIndices) {
            previewClassificationResult.counterVisibleColumnIndices.insert(visibleColumnIndex);
        }
    }
    std::sort(
        previewClassificationResult.counterHints.begin(),
        previewClassificationResult.counterHints.end(),
        [](const features::classification::FrameFieldHint& leftHint,
           const features::classification::FrameFieldHint& rightHint) {
            if (leftHint.absoluteStartBit != rightHint.absoluteStartBit) {
                return leftHint.absoluteStartBit < rightHint.absoluteStartBit;
            }
            return leftHint.absoluteEndBit < rightHint.absoluteEndBit;
        }
    );
    return previewClassificationResult;
}

}  // namespace

BitstreamSyncDiscoveryDialog::BitstreamSyncDiscoveryDialog(data::ByteDataSource* dataSource, QWidget* parent)
    : QDialog(parent),
      dataSource_(dataSource),
      previewFrameLayout_(new features::framing::FrameLayout()),
      previewAnalysisWatcher_(new QFutureWatcher<features::classification::FrameFieldClassificationResult>(this)) {
    qRegisterMetaType<features::bitstream_sync_discovery::BitstreamSyncDiscoveryProgressUpdate>();
    qRegisterMetaType<features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate>();
    qRegisterMetaType<features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidateList>();

    setModal(true);
    setWindowTitle(QStringLiteral("Find Frames"));
    resize(1280, 760);
    buildLayout();
    setScanningState(false);

    connect(
        previewAnalysisWatcher_,
        &QFutureWatcher<features::classification::FrameFieldClassificationResult>::finished,
        this,
        [this]() {
            if (previewAnalysisWatcher_ == nullptr) {
                return;
            }

            if (activePreviewAnalysisRequestId_ == previewAnalysisRequestId_
                && activePreviewAnalysisCandidateKey_ == previewedCandidateKey_) {
                const features::classification::FrameFieldClassificationResult classificationResult =
                    previewAnalysisWatcher_->result();
                if (previewBitViewer_ != nullptr) {
                    previewBitViewer_->setPreviewColumnHighlights({});
                    previewBitViewer_->setPreviewBitHighlights(previewBitHighlightRanges(classificationResult));
                }
                if (previewHintsPanel_ != nullptr) {
                    previewHintsPanel_->setHints(classificationResult);
                }
                return;
            }

            const int currentRow = resultsTableWidget_ != nullptr ? resultsTableWidget_->currentRow() : -1;
            if (currentRow >= 0 && currentRow < candidates_.size()) {
                updatePreviewForCandidateIndex(currentRow);
            } else {
                clearPreview();
            }
        }
    );
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
        progressDetailLabel_->setText(QStringLiteral("Load a file before running Find Frames."));
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
    progressPhaseLabel_->setText(QStringLiteral("Find Frames"));
    progressDetailLabel_->setText(QStringLiteral("Scanning framing candidates..."));
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
    const std::optional<features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate> currentCandidate =
        selectedCandidate();
    const QString preferredCandidateKey = currentCandidate.has_value()
        ? candidateKey(*currentCandidate)
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
    progressDetailLabel_->setText(QStringLiteral("Find Frames was canceled."));
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

    progressPhaseLabel_ = new QLabel(QStringLiteral("Find Frames"), this);
    progressPhaseLabel_->setVisible(false);
    rootLayout->addWidget(progressPhaseLabel_);

    progressDetailLabel_ = new QLabel(
        QStringLiteral("Choose the search range and framing filters, then click Discover."),
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
    QWidget* leftPaneWidget = new QWidget(this);
    QVBoxLayout* leftPaneLayout = new QVBoxLayout(leftPaneWidget);
    leftPaneLayout->setContentsMargins(0, 0, 0, 0);
    leftPaneLayout->setSpacing(8);

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
    leftPaneLayout->addWidget(resultsTableWidget_, 1);

    QGroupBox* previewHintsGroup = new QGroupBox(QStringLiteral("Detected Fields"), leftPaneWidget);
    QVBoxLayout* previewHintsLayout = new QVBoxLayout(previewHintsGroup);
    previewHintsLayout->setContentsMargins(8, 8, 8, 8);
    previewHintsLayout->setSpacing(6);
    previewHintsPanel_ = new FrameFieldHintsPanel(previewHintsGroup);
    previewHintsLayout->addWidget(previewHintsPanel_);
    leftPaneLayout->addWidget(previewHintsGroup);
    contentSplitter->addWidget(leftPaneWidget);

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
    previewBitViewer_->setAutoFitEnabled(false);
    previewBitViewer_->setCellSize(8);
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
    const QVector<features::framing::FrameSpan> displayedPreviewFrameSpans = limitedPreviewFrameSpans(
        previewFrameSpans,
        previewFrameLayout_->rowOrderMode(),
        previewFrameLayout_->rowOrderDescending()
    );
    previewFrameLayout_->setFrames(displayedPreviewFrameSpans);

    int previewFrameBitWidth = candidate.refinedPattern.bitWidth;
    for (const features::framing::FrameSpan& previewFrameSpan : displayedPreviewFrameSpans) {
        previewFrameBitWidth = qMax(previewFrameBitWidth, static_cast<int>(previewFrameSpan.lengthBits));
    }
    const QVector<features::columns::VisibleByteColumn> syncPreviewColumns =
        previewColumnsForBitRange(candidate.refinedPattern.bitWidth, previewFrameBitWidth);
    const QVector<features::classification::FrameFieldColumnSnapshot> previewColumnSnapshotsForCandidate =
        previewColumnSnapshots(syncPreviewColumns);
    const int previewRowCount = displayedPreviewFrameSpans.size();
    previewBitViewer_->setPreviewSource(
        dataSource_,
        previewFrameLayout_,
        syncPreviewColumns,
        0,
        previewRowCount
    );
    previewBitViewer_->setPreviewColumnHighlights({});
    if (previewHintsPanel_ != nullptr) {
        previewHintsPanel_->setPendingAnalysis();
    }

    previewBitViewer_->setPreviewBitHighlights({});
    previewSummaryLabel_->setText(candidateSummaryText(candidate, previewFrameSpans.size(), previewRowCount));
    previewSummaryLabel_->setVisible(true);
    previewedCandidateKey_ = candidateKey(candidate);
    applyButton_->setEnabled(true);
    features::framing::FrameLayout analysisFrameLayout;
    analysisFrameLayout.setFrames(previewFrameSpans);
    analysisFrameLayout.setRowOrder(
        previewFrameLayout_->rowOrderMode(),
        previewFrameLayout_->rowOrderDescending()
    );
    startPreviewAnalysis(
        previewedCandidateKey_,
        analysisFrameLayout,
        previewColumnSnapshotsForCandidate,
        static_cast<int>(candidate.frameLengthSummary.minimumLengthBits) - 1
    );
}

void BitstreamSyncDiscoveryDialog::startPreviewAnalysis(
    const QString& candidateKey,
    const features::framing::FrameLayout& analysisFrameLayout,
    const QVector<features::classification::FrameFieldColumnSnapshot>& columnSnapshots,
    int universalEndBit
) {
    ++previewAnalysisRequestId_;
    if (dataSource_ == nullptr
        || !dataSource_->hasData()
        || !analysisFrameLayout.isFramed()
        || columnSnapshots.isEmpty()) {
        if (previewHintsPanel_ != nullptr) {
            previewHintsPanel_->setHints(features::classification::FrameFieldClassificationResult{});
        }
        return;
    }
    if (previewAnalysisWatcher_ == nullptr || previewAnalysisWatcher_->isRunning()) {
        return;
    }

    activePreviewAnalysisRequestId_ = previewAnalysisRequestId_;
    activePreviewAnalysisCandidateKey_ = candidateKey;
    const data::ByteDataSource dataSourceSnapshot = *dataSource_;
    const features::framing::FrameLayout frameLayoutSnapshot = analysisFrameLayout;
    const QVector<features::classification::FrameFieldColumnSnapshot> columnSnapshotsSnapshot = columnSnapshots;
    previewAnalysisWatcher_->setFuture(QtConcurrent::run(
        [dataSourceSnapshot, frameLayoutSnapshot, columnSnapshotsSnapshot, universalEndBit]() {
            return analyzePreviewFields(
                dataSourceSnapshot,
                frameLayoutSnapshot,
                columnSnapshotsSnapshot,
                universalEndBit
            );
        }
    ));
}

void BitstreamSyncDiscoveryDialog::clearPreview() {
    ++previewAnalysisRequestId_;
    activePreviewAnalysisCandidateKey_.clear();
    previewFrameLayout_->clearFrame();
    previewBitViewer_->setPreviewSource(nullptr, nullptr, {}, 0, 0);
    previewBitViewer_->setPreviewColumnHighlights({});
    previewBitViewer_->setPreviewBitHighlights({});
    if (previewHintsPanel_ != nullptr) {
        previewHintsPanel_->showUnavailable();
    }
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
    int previewFrameCount,
    int displayedPreviewRowCount
) const {
    QString previewLabel;
    if (displayedPreviewRowCount < previewFrameCount) {
        previewLabel = previewFrameCount != candidate.frameSpans.size()
            ? QStringLiteral("Preview: expanded sparse candidate from %1 detected frames into %2 rows; showing %3 rows\n")
                  .arg(candidate.frameSpans.size())
                  .arg(previewFrameCount)
                  .arg(displayedPreviewRowCount)
            : QStringLiteral("Preview: showing %1 of %2 framed rows\n")
                  .arg(displayedPreviewRowCount)
                  .arg(previewFrameCount);
    } else {
        previewLabel = previewFrameCount != candidate.frameSpans.size()
            ? QStringLiteral("Preview: expanded sparse candidate from %1 detected frames into %2 preview rows using median frame size\n")
                  .arg(candidate.frameSpans.size())
                  .arg(previewFrameCount)
            : QStringLiteral("Preview: scroll to inspect all %1 framed rows\n")
                  .arg(candidate.frameSpans.size());
    }

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
    int syncBitWidth,
    int previewFrameBitWidth
) const {
    QVector<features::columns::VisibleByteColumn> previewColumns;
    if (previewFrameBitWidth <= 0) {
        return previewColumns;
    }

    auto appendSegment = [&previewColumns](int segmentStartBit, int segmentEndBit, bool isUndefined) {
        features::columns::VisibleByteColumn visibleColumn;
        visibleColumn.byteIndex = segmentStartBit / 8;
        visibleColumn.byteEndIndex = segmentEndBit / 8;
        visibleColumn.bitStart = segmentStartBit % 8;
        visibleColumn.bitEnd = segmentEndBit % 8;
        visibleColumn.absoluteStartBit = segmentStartBit;
        visibleColumn.absoluteEndBit = segmentEndBit;
        visibleColumn.isUndefined = isUndefined;
        previewColumns.append(visibleColumn);
    };

    const int boundedSyncWidth = qBound(0, syncBitWidth, previewFrameBitWidth);
    if (boundedSyncWidth > 0) {
        appendSegment(0, boundedSyncWidth - 1, false);
    }

    const int widthBytes = (previewFrameBitWidth + 7) / 8;
    for (int byteIndex = 0; byteIndex < widthBytes; ++byteIndex) {
        const int byteStartBit = byteIndex * 8;
        const int byteEndBit = qMin(previewFrameBitWidth - 1, byteStartBit + 7);
        if (byteEndBit < byteStartBit) {
            continue;
        }

        const bool byteCoveredBySync = boundedSyncWidth > 0
            && byteStartBit < boundedSyncWidth
            && byteEndBit >= 0;
        if (!byteCoveredBySync) {
            appendSegment(byteStartBit, byteEndBit, false);
            continue;
        }

        const int coveredStartBit = qMax(byteStartBit, 0);
        const int coveredEndBit = qMin(byteEndBit, boundedSyncWidth - 1);
        if (coveredStartBit > byteStartBit) {
            appendSegment(byteStartBit, coveredStartBit - 1, true);
        }
        if (coveredEndBit < byteEndBit) {
            appendSegment(coveredEndBit + 1, byteEndBit, true);
        }
    }

    return previewColumns;
}

QVector<features::classification::FrameFieldColumnSnapshot> BitstreamSyncDiscoveryDialog::previewColumnSnapshots(
    const QVector<features::columns::VisibleByteColumn>& previewColumns
) const {
    QVector<features::classification::FrameFieldColumnSnapshot> columnSnapshots;
    columnSnapshots.reserve(previewColumns.size());
    for (int visibleColumnIndex = 0; visibleColumnIndex < previewColumns.size(); ++visibleColumnIndex) {
        const features::columns::VisibleByteColumn& visibleColumn = previewColumns.at(visibleColumnIndex);
        features::classification::FrameFieldColumnSnapshot columnSnapshot;
        columnSnapshot.visibleColumnIndex = visibleColumnIndex;
        columnSnapshot.visibleColumn = visibleColumn;
        columnSnapshot.label = previewFieldLabel(visibleColumn);
        columnSnapshot.displayFormat = visibleColumn.bitWidth() == 1 ? QStringLiteral("binary") : QStringLiteral("hex");
        columnSnapshots.append(columnSnapshot);
    }
    return columnSnapshots;
}

}  // namespace bitabyte::ui
