#pragma once

#include <atomic>
#include <functional>

#include "features/bitstream_sync_discovery/bitstream_sync_discovery_types.h"

class QString;

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::features::bitstream_sync_discovery {

class BitstreamSyncDiscoveryEngine {
public:
    using ProgressCallback = std::function<void(
        const BitstreamSyncDiscoveryProgressUpdate& progressUpdate,
        const BitstreamSyncDiscoveryCandidateList& partialCandidates
    )>;

    [[nodiscard]] static BitstreamSyncDiscoveryCandidateList discover(
        const data::ByteDataSource& dataSource,
        const BitstreamSyncDiscoverySettings& settings,
        const std::atomic_bool* cancelRequested,
        const ProgressCallback& progressCallback,
        QString* errorMessage = nullptr
    );
};

}  // namespace bitabyte::features::bitstream_sync_discovery
