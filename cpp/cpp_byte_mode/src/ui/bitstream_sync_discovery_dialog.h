#pragma once

#include <QDialog>
#include <QVector>
#include <QtTypes>

#include <optional>

#include "features/classification/frame_field_classification.h"
#include "features/columns/byte_column_definition.h"
#include "features/bitstream_sync_discovery/bitstream_sync_discovery_types.h"

class QLabel;
class QProgressBar;
class QPushButton;
class QComboBox;
class QDoubleSpinBox;
class QSpinBox;
class QTableWidget;
class QThread;
template <typename T>
class QFutureWatcher;

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::features::columns {
struct VisibleByteColumn;
}

namespace bitabyte::features::classification {
struct FrameFieldClassificationResult;
}

namespace bitabyte::features::bitstream_sync_discovery {
class BitstreamSyncDiscoveryWorker;
}

namespace bitabyte::features::framing {
class FrameLayout;
}

namespace bitabyte::ui {

class LiveBitViewerWidget;
class FrameFieldHintsPanel;

class BitstreamSyncDiscoveryDialog final : public QDialog {
    Q_OBJECT

public:
    explicit BitstreamSyncDiscoveryDialog(
        data::ByteDataSource* dataSource,
        const QVector<features::columns::ByteColumnDefinition>* existingColumnDefinitions = nullptr,
        QWidget* parent = nullptr
    );
    ~BitstreamSyncDiscoveryDialog() override;

    [[nodiscard]] std::optional<features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate> selectedCandidate() const;
    [[nodiscard]] QVector<features::classification::FrameFieldHint> selectedColumnHints() const;

private slots:
    void startDiscovery();
    void cancelDiscovery();
    void handleProgressChanged(
        const bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryProgressUpdate& progressUpdate
    );
    void handlePartialResultsReady(
        const bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidateList& candidates
    );
    void handleDiscoveryFinished(
        const bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidateList& candidates
    );
    void handleDiscoveryFailed(const QString& errorMessage);
    void handleDiscoveryCanceled();
    void updatePreviewForSelection();
    void applySelectedCandidate();
    void cyclePreviewChronologicalOrder();
    void cyclePreviewLengthOrder();

private:
    void buildLayout();
    void setScanningState(bool isScanning);
    void stopWorkerThread();
    void refreshResultsTable(const QString& preferredCandidateKey = QString());
    void updatePreviewForCandidateIndex(int candidateIndex);
    void startPreviewAnalysis(
        const QString& candidateKey,
        const features::framing::FrameLayout& analysisFrameLayout,
        const QVector<features::classification::FrameFieldColumnSnapshot>& columnSnapshots,
        int universalEndBit
    );
    void clearPreview();
    [[nodiscard]] QVector<features::framing::FrameSpan> previewFrameSpansForCandidate(
        const bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidate
    ) const;
    [[nodiscard]] QString candidateKey(
        const bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidate
    ) const;
    [[nodiscard]] QString candidateSummaryText(
        const bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidate& candidate,
        int previewFrameCount,
        int displayedPreviewRowCount
    ) const;
    void applyPreviewRowOrder(
        features::framing::FrameLayout::RowOrderMode rowOrderMode,
        bool descending
    );
    void updatePreviewRowOrderButtons();
    [[nodiscard]] QVector<features::columns::VisibleByteColumn> previewColumnsForBitRange(
        int syncBitWidth,
        int previewFrameBitWidth
    ) const;
    [[nodiscard]] QVector<features::classification::FrameFieldColumnSnapshot> previewColumnSnapshots(
        const QVector<features::columns::VisibleByteColumn>& previewColumns
    ) const;

    data::ByteDataSource* dataSource_ = nullptr;
    const QVector<features::columns::ByteColumnDefinition>* existingColumnDefinitions_ = nullptr;
    features::framing::FrameLayout* previewFrameLayout_ = nullptr;
    LiveBitViewerWidget* previewBitViewer_ = nullptr;
    FrameFieldHintsPanel* previewHintsPanel_ = nullptr;
    QTableWidget* resultsTableWidget_ = nullptr;
    QSpinBox* minimumBitsSpinBox_ = nullptr;
    QSpinBox* maximumBitsSpinBox_ = nullptr;
    QSpinBox* minimumMatchesSpinBox_ = nullptr;
    QSpinBox* candidatesPerWidthSpinBox_ = nullptr;
    QComboBox* searchEffortComboBox_ = nullptr;
    QSpinBox* minimumFrameBitsSpinBox_ = nullptr;
    QDoubleSpinBox* entropyThresholdSpinBox_ = nullptr;
    QDoubleSpinBox* maximumGapCvSpinBox_ = nullptr;
    QLabel* progressPhaseLabel_ = nullptr;
    QLabel* progressDetailLabel_ = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QLabel* previewSummaryLabel_ = nullptr;
    QPushButton* previewChronologicalOrderButton_ = nullptr;
    QPushButton* previewLengthOrderButton_ = nullptr;
    QPushButton* discoverButton_ = nullptr;
    QPushButton* cancelDiscoveryButton_ = nullptr;
    QPushButton* applyButton_ = nullptr;
    QThread* workerThread_ = nullptr;
    features::bitstream_sync_discovery::BitstreamSyncDiscoveryWorker* worker_ = nullptr;
    QFutureWatcher<features::classification::FrameFieldClassificationResult>* previewAnalysisWatcher_ = nullptr;
    features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidateList candidates_;
    QString previewedCandidateKey_;
    QString activePreviewAnalysisCandidateKey_;
    int appliedCandidateIndex_ = -1;
    bool previewChronologicalDescending_ = false;
    bool previewLengthDescending_ = false;
    quint64 previewAnalysisRequestId_ = 0;
    quint64 activePreviewAnalysisRequestId_ = 0;
};

}  // namespace bitabyte::ui
