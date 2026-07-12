# Paper

## Versión ACL en LaTeX

La versión científica compilable está en [`acl/main.tex`](acl/main.tex). Desde
PowerShell se genera el PDF con:

```powershell
cd paper\acl
.\compile.ps1
```

El borrador Markdown se conserva como base narrativa e historial de la
evolución del trabajo.

## Borrador narrativo

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
