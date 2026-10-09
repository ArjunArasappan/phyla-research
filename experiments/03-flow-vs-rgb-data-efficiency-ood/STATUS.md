# Experiment 3 status

**All 12 bounded LoRA training runs completed.** The last successful node check observed 25 of 36 evaluation conditions complete (125 finished rollouts). The node subsequently stopped responding to SSH before final retrieval; the final count is unknown.

All 12 learned policy bundles and 64 raw demonstrations are checksum-verified locally in `exp03-core.tar.gz`. Final optimizer/RNG checkpoints were archived on the node but their upload did not start. The user reports provider-confirmed node/disk deletion: full evaluation arrays/videos and node-only optimizer/RNG checkpoints are lost. Repeat evaluations from the saved learned policies on new compute. A partial record of 19 measured condition summaries is retained in `results/ai/partial-evaluation-observations.json`; it is not a complete or fully audited comparison.

No flow-supervision efficacy, sample-efficiency, or OOD superiority claim is supported yet. Five paired reset seeds, two optimizer seeds, one subset seed, one PushCube task, and a fixed 1,000-update budget define this exploratory pilot. LTX policy SFT was executed; Wan policy SFT was not.

## New H100 recovery

All twelve learned policies restored on the new persistent volume. Simulator RGB/native8D validation and53 repository tests passed. Pinned model assets are restored and converted. The first saved policy passed exact frozen-base verification and began its ID evaluation. All36 groups /180rollouts are queued; no completed new success rate is asserted yet. Setup and renderer/software provenance: `../recovery/README.md`. Persistent progress: `/workspace/phyla-research-runs/evaluation-status.json`.
