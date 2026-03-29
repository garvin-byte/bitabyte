#include "features/bitstream_sync_discovery/bitstream_sync_discovery_worker.h"

#include "data/byte_data_source.h"
#include "features/bitstream_sync_discovery/bitstream_sync_discovery_engine.h"
#include "features/bitstream_sync_discovery/bitstream_sync_discovery_formatter.h"

namespace bitabyte::features::bitstream_sync_discovery {

BitstreamSyncDiscoveryWorker::BitstreamSyncDiscoveryWorker(
    const data::ByteDataSource* dataSource,
    const BitstreamSyncDiscoverySettings& settings,
    QObject* parent
)
    : QObject(parent),
      dataSource_(dataSource),
      settings_(settings) {
}

void BitstreamSyncDiscoveryWorker::cancel() {
    cancelRequested_.store(true, std::memory_order_relaxed);
}

void BitstreamSyncDiscoveryWorker::start() {
    if (dataSource_ == nullptr || !dataSource_->hasData()) {
        emit failed(QStringLiteral("Load a file before running bitstream sync discovery."));
        return;
    }

    QString errorMessage;
    const BitstreamSyncDiscoveryCandidateList candidates = BitstreamSyncDiscoveryEngine::discover(
        *dataSource_,
        settings_,
        &cancelRequested_,
        [this](const BitstreamSyncDiscoveryProgressUpdate& progressUpdate,
               const BitstreamSyncDiscoveryCandidateList& partialCandidates) {
            emit progressChanged(progressUpdate);
            emit partialResultsReady(formatCandidatesForDisplay(partialCandidates));
        },
        &errorMessage
    );

    if (cancelRequested_.load(std::memory_order_relaxed)) {
        emit canceled();
        return;
    }

    if (candidates.isEmpty()) {
        emit failed(errorMessage.isEmpty()
                        ? QStringLiteral("No bitstream sync candidates were found.")
                        : errorMessage);
        return;
    }

    emit finished(formatCandidatesForDisplay(candidates));
}

}  // namespace bitabyte::features::bitstream_sync_discovery
