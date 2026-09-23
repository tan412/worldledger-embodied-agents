# Transactional verification

This document describes the public integration boundary used by the WorldLedger report. A context is locked before a candidate is proposed. The candidate carries the robot profile, initial state, scene, task, protocol, and action identity needed for verification.

The control sequence is:

1. Build and hash a context.
2. Rank or select candidates without granting execution authority.
3. Submit one candidate against the current accepted base revision.
4. Execute a fresh simulation run under the declared protocol.
5. Save the applied controls and independently replay them.
6. Check receipt identity, mandatory claims, evidence hashes, thresholds, and base-state consistency.
7. Commit only a complete pass; retain rejection and insufficient-evidence outcomes.

The public example records six simulated worlds, 24 candidate attempts, 10 accepted candidates, 14 rejected candidates, and 12 unevaluated control or setup events. All six world histories pass the saved-event replay check. This is an integration example, not a held-out generalization benchmark and not a hardware execution record.

The source package keeps compatibility modules under `organoid_kernel/`. A separate external control service is optional for local experiments; no private endpoint or deployment credential is part of this release.
