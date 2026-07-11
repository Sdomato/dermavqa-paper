# Paper-ready result summary

| dataset_variant | method | split | unit | n | sacrebleu | chrf_corpus | chrf_mean | rouge_l_mean | token_f1_mean | bertscore_f1_mean | mean_latency_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_enriched | retrieval_tfidf | test | case | 100 | 8.266 | 26.990 | 0.286 | 0.216 | 0.256 | 0.716 |  |
| dataset_enriched | retrieval_e5 | test | case | 100 | 8.555 | 30.091 | 0.308 | 0.212 | 0.259 | 0.714 |  |
| dataset_enriched | retrieval_sbert | test | case | 100 | 8.795 | 29.816 | 0.309 | 0.222 | 0.269 | 0.720 |  |
| dataset_enriched | vlm_lora_case_avg | test | case | 100 | 5.022 | 34.532 | 0.368 | 0.250 | 0.314 | 0.737 | 46.996 |
| dataset_enriched | vlm_zero_shot | test | case | 100 |  |  | 0.2938 | 0.1202 | 0.1849 | 0.6789 | 14.697 |
| dataset_enriched | vlm_zero_shot_rag_e5_small_enriched | test | case | 100 |  |  | 0.3075 | 0.1351 | 0.1939 | 0.6811 | 18.066 |
| dataset_enriched | vlm_lora_rag_e5_small_enriched | test | case | 100 |  |  | 0.3035 | 0.2322 | 0.2757 | 0.7254 | 11.716 |
| dataset_longest_answer | retrieval_tfidf_train_only | test | case | 100 | 0.462 | 13.160 | 0.148 | 0.056 | 0.072 |  | 0.001 |
| dataset_longest_answer | vlm_lora | test | case | 100 | 0.328 | 11.082 | 0.157 | 0.112 | 0.131 |  | 12.541 |
| dataset_longest_answer | vlm_zero_shot_by_image | test | case | 100 |  |  | 0.2042 | 0.0799 | 0.1026 | 0.6428 | 15.016 |
| dataset_longest_answer | vlm_zero_shot_by_image_rag_e5_small_longest | test | case | 100 |  |  | 0.1960 | 0.0797 | 0.1018 | 0.6396 | 18.034 |
| dataset_longest_answer | vlm_lora_by_image | test | case | 100 |  |  | 0.1608 | 0.0798 | 0.0879 | 0.6179 | 20.449 |
| dataset_longest_answer | vlm_lora_by_image_rag_e5_small_longest | test | case | 100 |  |  | 0.1220 | 0.0445 | 0.0380 | 0.5691 | 29.387 |
| dataset_short_answer | retrieval_tfidf_train_only | test | case | 100 | 0.773 | 9.958 | 0.089 | 0.013 | 0.013 |  | 0.001 |

Notes:
- `chrf_corpus` is on a 0-100 scale when copied from sacreBLEU corpus chrF.
- `chrf_mean`, `ROUGE-L`, `token-F1`, and `BERTScore F1` are on a 0-1 scale.
- All VLM results are case-level (n=100): for methods evaluated per image, metrics are averaged across images within each encounter_id before computing the mean over cases.
- `sacrebleu` and `chrf_corpus` are not available for the new per-image methods aggregated to case level (would require re-running corpus-level scoring on aggregated predictions).