#pragma once

#include <QObject>

#include <atomic>

#include "features/bitstream_sync_discovery/bitstream_sync_discovery_types.h"

class QString;

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::features::bitstream_sync_discovery {

class BitstreamSyncDiscoveryWorker final : public QObject {
    Q_OBJECT

public:
    explicit BitstreamSyncDiscoveryWorker(
        const data::ByteDataSource* dataSource,
        const BitstreamSyncDiscoverySettings& settings,
        QObject* parent = nullptr
    );

    void cancel();

public slots:
    void start();

signals:
    void progressChanged(const bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryProgressUpdate& progressUpdate);
    void partialResultsReady(
        const bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidateList& candidates
    );
    void finished(
        const bitabyte::features::bitstream_sync_discovery::BitstreamSyncDiscoveryCandidateList& candidates
    );
    void failed(const QString& errorMessage);
    void canceled();

private:
    const data::ByteDataSource* dataSource_ = nullptr;
    BitstreamSyncDiscoverySettings settings_;
    std::atomic_bool cancelRequested_ = false;
};

}  // namespace bitabyte::features::bitstream_sync_discovery
