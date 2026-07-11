# Handoff → Santino: generar el cache de embeddings (multimodal)

**Qué te pido:** correr un script una vez en tu VM con GPU para generar un archivo
de ~5 MB con los embeddings de los 998 casos. Eso habilita el **retrieval
multimodal** (texto + imagen) del servicio de ingeniería (DermaAssist).

**Por qué vos:** tu VM ya tiene GPU + las deps pesadas (`open_clip`, `torch`) y las
imágenes. El servicio NO calcula esto: solo carga el `.npz` que generes, así que el
cómputo pesado se hace una sola vez en tu máquina y nunca en la nuestra.

**Tiempo estimado:** segundos en GPU (minutos si fuera CPU). Es una corrida única.

---

## Contexto (1 párrafo)

El servicio de ingeniería recupera "casos similares" a una consulta nueva. Hoy lo hace
solo por texto. Para hacerlo **multimodal** (también por imagen, como el baseline
`multimodal_retrieval.py` del paper, α=0.6) el servicio necesita los embeddings de cada
caso precalculados: E5 para el texto y BiomedCLIP para las imágenes. El script reutiliza
**tus mismas funciones** de `src/multimodal_retrieval.py`, solo que guarda los embeddings
crudos (no la matriz de similitud).

## Requisitos en la VM

- Rama **`dev-ing`** del repo (es donde vive el script; incluye `src/` igual que `develop`).
- Las imágenes en `data/iiyi/images_final/` (las que ya usaste).
- Deps: `torch`, `transformers`, `open_clip_torch`, `pillow` (las del entorno de tu corrida).

## Pasos

```bash
# 1. En la VM, traer la rama de ingeniería
git fetch origin
git checkout dev-ing
git pull

# 2. (si falta) instalar open_clip
pip install open_clip_torch

# 3. Prueba rápida con 5 casos (verifica que todo arranca: descarga modelos, lee imágenes)
python ing/backend/scripts/build_case_embeddings.py --limit 5

# 4. Corrida completa (los 998 casos)
python ing/backend/scripts/build_case_embeddings.py
```

## Qué genera

Un único archivo:

```
outputs/embeddings/case_embeddings.npz   (~5 MB)
```

Con: `encounter_ids`, `text_emb` (E5), `visual_emb` (BiomedCLIP), `has_image` y `meta`
(model ids, dims, α). Al terminar, el script imprime un resumen tipo:

```json
{
  "text_model": "intfloat/multilingual-e5-base",
  "visual_model": "BiomedCLIP (open_clip) con fallback ViT-B-32",
  "text_dim": 768,
  "visual_dim": 512,
  "alpha_text": 0.6,
  "n": 998,
  "n_con_imagen": 998
}
```

> Chequeo rápido: `n_con_imagen` debería ser cercano a 998 (casi todos los casos tienen
> imagen). Si da muy bajo, probablemente las imágenes no estén en `data/iiyi/images_final/`.

## Cómo devolvérnoslo

Cualquiera de las dos:

- **Pushear el archivo a `dev-ing`** (pesa ~5 MB, entra sin problema):
  ```bash
  git add outputs/embeddings/case_embeddings.npz
  git commit -m "data(ing): cache de embeddings de casos (E5 + BiomedCLIP) para retrieval multimodal"
  git push origin dev-ing
  ```
- O **mandarnos el `.npz`** por donde sea y nosotros lo commiteamos.

## Notas

- Si BiomedCLIP no estuviera disponible, el script cae al fallback `ViT-B-32` (igual que
  tu baseline). Avisanos si pasó, porque cambia los números respecto del paper.
- El servicio usa el prefijo `"passage: "` para los casos (lo que generás acá) y
  `"query: "` para la consulta nueva, siguiendo la convención de E5.
- Cualquier cosa rara, mandanos la salida de la prueba con `--limit 5`.

¡Gracias! 🙌
