# Contrastive explanation run summary

These explanations are post-hoc observable justifications, not hidden chain-of-thought traces.

| method | n | parse success | minimum length | mean tokens | answer stability F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lora | 23 | 0.000 | 0.000 | 0.0 | 0.000 |
| lora_rag | 23 | 0.043 | 0.043 | 33.8 | 0.014 |
| zero_shot | 23 | 0.652 | 0.652 | 105.3 | 0.287 |
| zero_shot_rag | 23 | 0.739 | 0.739 | 114.6 | 0.250 |