PYTHON ?= python3

.PHONY: help all data retrieval eval-retrieval paper dry-run

help:
	@echo ""
	@echo "DermaVQA pipeline — targets disponibles:"
	@echo ""
	@echo "  make data            Construye todos los datasets desde los JSON crudos (CPU)"
	@echo "  make retrieval       Corre todos los baselines de retrieval (CPU/GPU)"
	@echo "  make eval-retrieval  Evalúa los baselines de retrieval y genera métricas"
	@echo "  make paper           Genera tablas y figuras paper-ready en outputs/paper/"
	@echo "  make dry-run         Valida prompts e imágenes sin cargar modelos (CPU)"
	@echo "  make all             Ejecuta data + retrieval + eval-retrieval + paper en orden"
	@echo ""
	@echo "  Pasos GPU (ver scripts/):"
	@echo "    bash scripts/run_enriched_vlm_lora.sh       Fine-tuning enriched + infer + eval"
	@echo "    bash scripts/run_longest_by_image_vlm_lora.sh  Fine-tuning longest_by_image + infer + eval"
	@echo "    bash scripts/run_vlm_rag_comparison.sh      Zero-shot y LoRA con RAG"
	@echo ""

all: data retrieval eval-retrieval paper

# ── Fase 1: construcción de datasets ──────────────────────────────────────────

data:
	@echo "\n[data 1/2] Construyendo dataset_longest_answer y dataset_short_answer..."
	$(PYTHON) -m src.build_answer_datasets
	@echo "\n[data 2/2] Expandiendo a dataset_longest_answer_by_image (una fila por imagen)..."
	$(PYTHON) -m src.build_longest_by_image_dataset

# ── Fase 2: baselines de retrieval ────────────────────────────────────────────

retrieval: data
	@echo "\n[retrieval 1/5] TF-IDF (longest y short)..."
	$(PYTHON) -m src.tfidf_retrieval
	$(PYTHON) -m src.tfidf_retrieval_short

	@echo "\n[retrieval 2/5] Sentence-BERT (longest y short)..."
	$(PYTHON) -m src.sbert_retrieval
	$(PYTHON) -m src.sbert_retrieval_short

	@echo "\n[retrieval 3/5] Multilingual E5 (longest y short)..."
	$(PYTHON) -m src.e5_retrieval
	$(PYTHON) -m src.e5_retrieval_short

	@echo "\n[retrieval 4/5] Visual BiomedCLIP (longest y short — requiere imágenes)..."
	$(PYTHON) -m src.visual_retrieval
	$(PYTHON) -m src.visual_retrieval_short

	@echo "\n[retrieval 5/5] Multimodal late-fusion alpha=0.6 (longest y short)..."
	$(PYTHON) -m src.multimodal_retrieval --alpha 0.6
	$(PYTHON) -m src.multimodal_retrieval_short --alpha 0.6

# ── Fase 3: evaluación de retrieval ──────────────────────────────────────────

eval-retrieval: retrieval
	@echo "\n[eval-retrieval 1/2] Métricas all-split (longest y short)..."
	$(PYTHON) -m src.evaluate_retrieval --dataset longest_answer
	$(PYTHON) -m src.evaluate_retrieval --dataset short_answer

	@echo "\n[eval-retrieval 2/2] Retrieval held-out sin data leakage (train-only)..."
	$(PYTHON) -m src.evaluate_retrieval_heldout --dataset all

# ── Fase 4: tablas y figuras paper-ready ─────────────────────────────────────

paper:
	@echo "\n[paper] Consolidando métricas y generando tablas/figuras SVG..."
	$(PYTHON) -m src.build_paper_results

# ── Validación sin GPU ────────────────────────────────────────────────────────

dry-run:
	@echo "\n[dry-run 1/2] Validando prompts e imágenes de vlm_infer_longest_by_image (sin modelo)..."
	$(PYTHON) -m src.vlm_infer_longest_by_image --split valid --limit 5 --dry-run

	@echo "\n[dry-run 2/2] Validando formato chat de train_longest_by_image (sin modelo)..."
	$(PYTHON) -m src.train_longest_by_image --dry-run --limit 5
