# Paper-ready result summary

| dataset_variant | method | split | unit | n | sacrebleu | chrf_corpus | chrf_mean | rouge_l_mean | token_f1_mean | bertscore_f1_mean | mean_latency_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_enriched | retrieval_tfidf | test | case | 100 | 8.266 | 26.990 | 0.286 | 0.216 | 0.256 | 0.716 |  |
| dataset_enriched | retrieval_e5 | test | case | 100 | 8.555 | 30.091 | 0.308 | 0.212 | 0.259 | 0.714 |  |
| dataset_enriched | retrieval_sbert | test | case | 100 | 8.795 | 29.816 | 0.309 | 0.222 | 0.269 | 0.720 |  |
| dataset_enriched | vlm_lora_case_avg | test | case | 100 | 5.022 | 34.532 | 0.368 | 0.250 | 0.314 | 0.737 | 46.996 |
| dataset_enriched | vlm_zero_shot | test | image | 314 | 1.401 | 31.408 | 0.298 | 0.123 | 0.189 | 0.680 | 14.697 |
| dataset_enriched | vlm_zero_shot_rag_e5_small_enriched | test | image | 314 | 1.712 | 32.470 | 0.310 | 0.136 | 0.199 | 0.682 | 18.066 |
| dataset_enriched | vlm_lora_rag_e5_small_enriched | test | image | 314 | 8.799 | 29.170 | 0.302 | 0.228 | 0.274 | 0.724 | 11.716 |
| dataset_longest_answer | retrieval_tfidf_train_only | test | case | 100 | 0.462 | 13.160 | 0.148 | 0.056 | 0.072 |  | 0.001 |
| dataset_longest_answer | vlm_lora | test | case | 100 | 0.328 | 11.082 | 0.157 | 0.112 | 0.131 |  | 12.541 |
| dataset_longest_answer | vlm_zero_shot_by_image | test | image | 314 | 0.624 | 21.977 | 0.211 | 0.083 | 0.110 | 0.646 | 15.016 |
| dataset_longest_answer | vlm_zero_shot_by_image_rag_e5_small_longest | test | image | 314 | 0.609 | 21.226 | 0.202 | 0.083 | 0.107 | 0.642 | 18.034 |
| dataset_longest_answer | vlm_lora_by_image | test | image | 314 | 0.299 | 14.474 | 0.168 | 0.081 | 0.092 | 0.619 | 20.449 |
| dataset_longest_answer | vlm_lora_by_image_rag_e5_small_longest | test | image | 314 | 0.078 | 10.541 | 0.121 | 0.041 | 0.033 | 0.560 | 29.387 |
| dataset_short_answer | retrieval_tfidf_train_only | test | case | 100 | 0.773 | 9.958 | 0.089 | 0.013 | 0.013 |  | 0.001 |

Notes:
- `chrf_corpus` is on a 0-100 scale when copied from sacreBLEU corpus chrF.
- `chrf_mean`, `ROUGE-L`, `token-F1`, and `BERTScore F1` are on a 0-1 scale.
- The main table uses case-level rows; raw image-level enriched VLM rows remain in the long metrics table.