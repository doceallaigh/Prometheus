| run | architecture | params | seq_len | latest_val_loss | best_val_loss | latest_val_ppl |
| --- | --- | --- | --- | --- | --- | --- |
| baseline-tiny-20260529-054742 | dense | 2699328 | 128 | 0.3189 | 0.3189 | 1.3756 |
| variant-modular-dense-matched-20260529-054630 | modular | 2672229 | 128 | 2.0042 | 2.0042 | 7.4205 |
| variant-modular-sparse-matched-20260529-054655 | modular | 2672229 | 128 | 2.1817 | 2.1817 | 8.8617 |
| variant-modular-uneven-matched-20260529-054701 | modular | 2719581 | 128 | 2.0627 | 2.0627 | 7.8674 |
| variant-modular-uneven-sparse-matched-20260529-054809 | modular | 2719581 | 128 | 2.6187 | 2.6187 | 13.7175 |