# Paper draft

Esta carpeta contiene el borrador narrativo del trabajo.

- `draft.md`: paper completo en Markdown, listo para convertir luego a LaTeX/ACL.

Artefactos usados por el draft:

- `outputs/paper/tables/paper_main_test_comparison.csv`
- `outputs/paper/tables/paper_all_metrics_long.csv`
- `outputs/paper/tables/paper_clinical_review_20.csv`
- `outputs/paper/tables/paper_clinical_review_summary.csv`
- `outputs/paper/figures/*.svg`

Para regenerar tablas y figuras:

```bash
python -m src.evaluate_retrieval_heldout --dataset all
python -m src.build_paper_results
```
