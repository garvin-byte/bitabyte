#include "features/bitstream_sync_discovery/bitstream_sync_discovery_worker.h"

#include "data/byte_data_source.h"
#include "features/bitstream_sync_discovery/bitstream_sync_discovery_engine.h"
#include "features/bitstream_sync_discovery/bitstream_sync_discovery_formatter.h"

namespace bitabyte::features::bitstream_sync_discovery {
namespace {

[[nodiscard]] bool candidateMatchesLengthMode(
    const BitstreamSyncDiscoveryCandidate& candidate,
    BitstreamSyncDiscoveryLengthMode lengthMode
) {
    switch (lengthMode) {
    case BitstreamSyncDiscoveryLengthMode::VariableAndStatic:
        return true;
    case BitstreamSyncDiscoveryLengthMode::StaticOnly:
        return candidate.protocolClassification == QStringLiteral("fixed-length")
            || candidate.protocolClassification == QStringLiteral("multi-type fixed");
    }

    return true;
}

[[nodiscard]] BitstreamSyncDiscoveryCandidateList filteredCandidatesForLengthMode(
    const BitstreamSyncDiscoveryCandidateList& candidates,
    BitstreamSyncDiscoveryLengthMode lengthMode
) {
    BitstreamSyncDiscoveryCandidateList filteredCandidates;
    filteredCandidates.reserve(candidates.size());
    for (const BitstreamSyncDiscoveryCandidate& candidate : candidates) {
        if (candidateMatchesLengthMode(candidate, lengthMode)) {
            filteredCandidates.append(candidate);
        }
    }
    return filteredCandidates;
}

}  // namespace

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
        emit failed(QStringLiteral("Load a file before running Find Frames."));
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
            emit partialResultsReady(formatCandidatesForDisplay(
                filteredCandidatesForLengthMode(partialCandidates, settings_.lengthMode)
            ));
        },
        &errorMessage
    );

    if (cancelRequested_.load(std::memory_order_relaxed)) {
        emit canceled();
        return;
    }

    const BitstreamSyncDiscoveryCandidateList filteredCandidates =
        filteredCandidatesForLengthMode(candidates, settings_.lengthMode);

    if (filteredCandidates.isEmpty()) {
        emit failed(errorMessage.isEmpty()
                        ? (settings_.lengthMode == BitstreamSyncDiscoveryLengthMode::StaticOnly
                              ? QStringLiteral("No static frame candidates were found.")
                              : QStringLiteral("No frame candidates were found."))
                        : errorMessage);
        return;
    }

    emit finished(formatCandidatesForDisplay(filteredCandidates));
}

}  // namespace bitabyte::features::bitstream_sync_discovery
