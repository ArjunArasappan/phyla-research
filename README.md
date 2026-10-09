# Phyla GPU pilot scientific backup

This branch preserves completed scientific archives that exceeded laptop storage. Code and readable experiment reports are on `main`. Archives are split into files below GitHub's per-file limit. Every chunk and the reconstructed archive have SHA-256 checksums in `archives.json`.

Restore on a disk with adequate space:

```bash
git clone --single-branch --branch artifacts/gpu-pilot-2026-10-09 git@github.com:ArjunArasappan/phyla-research.git phyla-scientific-backup
cd phyla-scientific-backup
python3 restore_archives.py /path/to/durable/storage
```

The restore script validates chunk and complete-archive checksums. It does not automatically extract archives. Original relative paths are preserved inside each archive. Learned adapter exports are separate from downloadable frozen base checkpoints. Full optimizer checkpoints are not included in these scientific backups.
