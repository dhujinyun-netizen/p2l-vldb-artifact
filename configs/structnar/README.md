# StructNAR configuration scope

- `final_p2l_d3_rqc_cosine_w20.yaml` is the canonical paper evaluation
  configuration: P2L with prefix length `s=3`, standardized cosine PCAA with
  `lambda=20`, deterministic query sampling, batch size 8, and the epoch-99
  checkpoint.
- `train.yaml` reproduces the 100-epoch training schedule used to obtain the
  released checkpoint.
- `eval.yaml` is a base template retained for diagnostic scripts. It is not a
  reported paper configuration and intentionally does not encode the final P2L
  and PCAA policy.

Historical implementation keys such as `tcis`, `rrg`, and `rqc` are preserved
for checkpoint compatibility. The paper terminology is P2L and PCAA.
