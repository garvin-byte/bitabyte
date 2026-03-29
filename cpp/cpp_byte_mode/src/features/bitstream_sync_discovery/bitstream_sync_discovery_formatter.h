#pragma once

#include "features/bitstream_sync_discovery/bitstream_sync_discovery_types.h"

namespace bitabyte::features::bitstream_sync_discovery {

void formatCandidateForDisplay(BitstreamSyncDiscoveryCandidate* candidate);

[[nodiscard]] BitstreamSyncDiscoveryCandidateList formatCandidatesForDisplay(
    const BitstreamSyncDiscoveryCandidateList& candidates
);

}  // namespace bitabyte::features::bitstream_sync_discovery
