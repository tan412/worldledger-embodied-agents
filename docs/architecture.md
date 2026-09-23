# WorldLedger architecture

WorldLedger is the public name for a verification-centered organization of embodied-agent workflows. The release connects four responsibilities:

1. **Propose** a context-bound action with declared body, scene, initial state, task, and protocol.
2. **Execute** the native backend and retain applied controls, state, sensor-like streams, and runtime metadata.
3. **Record** evidence, hashes, validation receipts, and replay artifacts without turning execution into acceptance.
4. **Verify** claims through a separately defined authority that preserves accepted, rejected, and unevaluated outcomes.

The architecture deliberately separates policy proposal from authorization. A learned model, planner, script, or human-authored candidate can use the same envelope when it satisfies a backend contract. The validator remains responsible for the claims it can actually evaluate. Missing evidence is recorded as `not_evaluated`; it is not converted into a failure or a zero label.

The current release includes a multi-format evidence kernel, simulation-backed trajectory tools, a transaction-oriented integration example, and a thin MCP service interface. It does not include internal server deployment material, private data paths, raw production archives, or hardware safety claims.
