# Paper ACL en LaTeX

Esta carpeta contiene la versión científica del trabajo en formato ACL y en
español.

## Compilar en Windows

Desde PowerShell:

```powershell
cd paper\acl
.\compile.ps1
```

Si PowerShell bloquea scripts en la sesión actual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\compile.ps1
```

También se puede ejecutar directamente:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Para regenerar el gráfico de efectos pareados desde los CSV de evaluación:

```powershell
python -m pip install pandas matplotlib
python make_figures.py
```

El PDF resultante queda en `paper/acl/main.pdf`.

## Archivos

- `main.tex`: manuscrito principal.
- `references.bib`: bibliografía.
- `acl.sty` y `acl_natbib.bst`: estilo oficial de
  [ACL](https://github.com/acl-org/acl-style-files).
- `compile.ps1`: compilación reproducible con MiKTeX/TeX Live.
- `make_figures.py`: regenera las figuras del paper desde resultados versionados.
- `figures/paired_effects.pdf`: efectos pareados usados en el análisis de
  explicaciones post-hoc.

La versión principal está diseñada para un máximo de ocho páginas de contenido
en formato ACL; referencias, limitaciones y consideraciones éticas se mantienen
como secciones separadas según la convención de ACL.
