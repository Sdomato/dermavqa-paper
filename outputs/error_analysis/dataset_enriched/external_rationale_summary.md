# Contrastive explanation run summary

These explanations are post-hoc observable justifications, not hidden chain-of-thought traces.

| method | n | parse success | minimum length | mean tokens | answer stability F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lora | 25 | 1.000 | 1.000 | 159.4 | 1.000 |
| lora_rag | 25 | 1.000 | 1.000 | 161.7 | 1.000 |
| lora_rag_aware | 25 | 1.000 | 1.000 | 162.0 | 1.000 |
| zero_shot | 25 | 1.000 | 1.000 | 161.5 | 1.000 |
| zero_shot_rag | 25 | 0.920 | 0.920 | 158.3 | 1.000 |