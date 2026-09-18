# Provenance of the released checkpoints

Every file here was traced back to the exact run and epoch that produced the results in the
manuscript, by re-encoding the benchmarks and matching embeddings (max|delta| ~1e-07) and by
nearest-neighbour parameter search for the training parents.

## Lineage of the reported model

    qm9_pretrained/{backbone_U.pt, readout_U.pt}
      -> denoising_{ligand,pocket}/backbone_epoch_{29,26}.pt      (stage 3)
        -> clip_sair_pretrained/dimenet_clip_epoch_5.pth           (stage 4)
          -> clip_pdbbind_finetuned/dimenet_clip_epoch_3.pth       (stage 5)  <-- reported model

| File | Source | md5 (12) |
|---|---|---|
| `clip_pdbbind_finetuned/dimenet_clip_epoch_3.pth` | `clip-models-pdbbind-no-fp-3/dimenet_clip_epoch_3.pth` | `0fff6f39fe86` |
| `clip_sair_pretrained/dimenet_clip_epoch_5.pth` | `clip-models-pretrained-no-fp/dimenet_clip_epoch_5.pth` | `401fc4dbf619` |
| `denoising_ligand/backbone_epoch_29.pt` | `ligand_models/backbone_epoch_29.pt` | `4e9be4d7aadd` |
| `denoising_pocket/backbone_epoch_26.pt` | `pocket_models/backbone_epoch_26.pt` | `400bd6cb28e3` |

(sources relative to `/home/ziqiaoxu/the-benchmark/dimenet/`)

Epoch 3 is the validation minimum of its run (val 22.98; the run recorded 9 epochs).
The SAIR parent was identified by parameter L2 against the fine-tuned epoch 0: nearest
candidate 2.30, runner-up 9.03 (3.9x gap), so the identification is unambiguous.

## What was previously released, and why it was wrong

The earlier release shipped `clip_pdbbind_finetuned/dimenet_clip_epoch_2.pth`
(md5 `efcf2237b902`, from `clip-models-pdbbind-no-fp-ag-2/`) and
`clip_sair_pretrained/dimenet_clip_epoch_98.pth` (md5 `1454db5d5755`, from
`clip-models-pretrained/`). Neither is on the lineage of the published results:

- The published DUD-E embeddings were written 2026-03-15 19:07; that PDBBind checkpoint was
  not written until 2026-03-16 01:14, six hours later. Re-encoding with it gives AUC 0.715 /
  EF1 8.72 against the paper's 0.697 / 7.49.
- The PDBBind run behind the paper wrote its epoch 0 at 17:56 on 2026-03-15; that SAIR
  checkpoint was written at 18:03, so it cannot have been the initialization.

Both are recoverable from git history if needed.

## A second issue this uncovered

As published, Table 1 mixed two models: the DUD-E row came from
`clip-models-pdbbind-no-fp-2/dimenet_clip_epoch_2_b.pth` (val 23.15, parent
`clip-models-pretrained-no-fp/epoch_7`) while the LIT-PCBA row came from
`clip-models-pdbbind-no-fp-3/dimenet_clip_epoch_3.pth` (val 22.98, parent `.../epoch_5`).
These are two replicates of the same configuration, each evaluated at its own validation
optimum, encoded 50 minutes apart on 2026-03-15.

All CLIP results are now regenerated from the single lowest-validation-loss model
(`no-fp-3` epoch 3), selected on validation loss before any test metric was inspected.
