# Paper-ready result summary

| dataset_variant | method | split | unit | n | sacrebleu | chrf_corpus | chrf_mean | rouge_l_mean | token_f1_mean | bertscore_f1_mean | mean_latency_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_enriched | retrieval_tfidf | test | case | 100 | 8.266 | 26.990 | 0.286 | 0.216 | 0.256 | 0.716 |  |
| dataset_enriched | retrieval_e5 | test | case | 100 | 8.555 | 30.091 | 0.308 | 0.212 | 0.259 | 0.714 |  |
| dataset_enriched | retrieval_sbert | test | case | 100 | 8.795 | 29.816 | 0.309 | 0.222 | 0.269 | 0.720 |  |
| dataset_enriched | vlm_lora_case_avg | test | case | 100 | 5.022 | 34.532 | 0.368 | 0.250 | 0.314 | 0.737 | 46.996 |
| dataset_longest_answer | retrieval_tfidf_train_only | test | case | 100 | 0.462 | 13.160 | 0.148 | 0.056 | 0.072 |  | 0.001 |
| dataset_longest_answer | vlm_zero_shot | test | case | 100 | 0.409 | 20.548 | 0.182 | 0.072 | 0.092 | 0.628 | 26.736 |
| dataset_longest_answer | vlm_lora | test | case | 100 | 0.328 | 11.082 | 0.157 | 0.112 | 0.131 | 0.667 | 12.541 |
| dataset_short_answer | retrieval_tfidf_train_only | test | case | 100 | 0.773 | 9.958 | 0.089 | 0.013 | 0.013 |  | 0.001 |

Notes:
- `chrf_corpus` is on a 0-100 scale when copied from sacreBLEU corpus chrF.
- `chrf_mean`, `ROUGE-L`, `token-F1`, and `BERTScore F1` are on a 0-1 scale.
- The main table uses case-level rows; raw image-level enriched VLM rows remain in the long metrics table.