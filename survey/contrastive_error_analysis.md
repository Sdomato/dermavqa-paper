# Contrastive error analysis

## Objective

Explain why the enriched and longest-answer VLM conditions perform differently
without treating a generated explanation as a faithful trace of hidden model
reasoning.

The comparison covers:

1. base VLM;
2. base VLM with E5 RAG;
3. VLM with LoRA;
4. VLM with LoRA plus E5 RAG at inference.
5. for the enriched dataset, VLM with LoRA trained and evaluated with RAG.

## Case selection

`src/build_contrastive_explanation_sample.py` joins predictions and per-case
metrics by `encounter_id` and `image_id` independently for each dataset. It
selects cases where:

- LoRA produces the largest gains or losses against the base VLM;
- RAG produces the largest gains or losses for the base VLM;
- RAG produces the largest gains or losses for the LoRA VLM;
- BERTScore and lexical metrics disagree;
- the reference is unusually short or long;
- images from the same encounter produce inconsistent scores.

The same script reports aggregate performance by reference length, question
length and number of images. These strata provide quantitative evidence before
reading individual examples.

## Structured re-inference

`src/vlm_explain_contrastive_cases.py` re-runs only the selected cases. The
model receives the same current image and question, plus the same retrieved
contexts for RAG conditions. It does not receive the reference answer or the
old prediction.

The output is JSON with:

- `answer_es`;
- `explanation`, with a default minimum of 100 tokenizer tokens;
- observable visual evidence;
- explicit evidence from the question;
- uncertainty;
- reported use of RAG context.

The prompt explicitly asks for an observable clinical justification rather
than chain-of-thought. A short or malformed explanation is retried once and
flagged if it still fails.

Because adding an explanation changes the prompt, the new answer is not used
to replace the original paper metrics. `original_vs_reanalysis_token_f1`
measures answer stability and helps detect cases where the diagnostic prompt
itself changed model behavior.

## Interpretation protocol

Use three evidence layers:

1. **Quantitative strata:** determine whether gains depend on answer length,
   question length or multi-image encounters.
2. **Contrastive examples:** inspect cases where two methods differ most on
   the same image and reference.
3. **Blinded review:** score clinical correctness, evidence grounding,
   answer-explanation consistency, hallucination and whether RAG helped or
   hurt.

Suggested error labels:

- wrong diagnosis;
- overly generic answer;
- unsupported specificity;
- image evidence ignored;
- question evidence ignored;
- copied or misleading RAG context;
- safe uncertainty;
- answer-length mismatch.

The explanations can support hypotheses about failure modes, but causal claims
must be based on repeated patterns across cases and reviewer agreement.

## Commands

Build the sample and summaries locally:

```bash
python3 -m src.build_contrastive_explanation_sample --dataset enriched
python3 -m src.build_contrastive_explanation_sample --dataset longest
```

Validate prompts without loading the model:

```bash
DRY_RUN=1 LIMIT=2 bash scripts/run_contrastive_explanation_analysis.sh
```

Run all four conditions on a GPU:

```bash
DATASET_VARIANT=enriched \
ADAPTER=outputs/results/dataset_enriched/vlm_lora/final_adapter \
RAG_AWARE_ADAPTER=outputs/results/dataset_enriched/vlm_lora_rag_aware/final_adapter \
MIN_EXPLANATION_TOKENS=100 \
nohup bash scripts/run_contrastive_explanation_analysis.sh \
  > outputs/error_analysis/dataset_enriched/full_run.log 2>&1 &
```

For longest answer:

```bash
DATASET_VARIANT=longest \
ADAPTER=outputs/results/dataset_longest_answer/vlm_lora_by_image/final_adapter \
MIN_EXPLANATION_TOKENS=100 \
nohup bash scripts/run_contrastive_explanation_analysis.sh \
  > outputs/error_analysis/dataset_longest_answer/full_run.log 2>&1 &
```

Main outputs, under the corresponding dataset directory:

- `outputs/error_analysis/dataset_enriched/contrastive_cases_test.csv`;
- `outputs/error_analysis/dataset_enriched/metric_strata_summary.csv`;
- `outputs/error_analysis/dataset_enriched/pairwise_effect_summary.csv`;
- `outputs/error_analysis/dataset_enriched/contrastive_explanation_review.csv`;
- `outputs/error_analysis/dataset_enriched/contrastive_explanation_summary.md`.
