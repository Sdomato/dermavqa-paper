# Models to compare

## Retrieval textual

Primera linea:

- TF-IDF con normalizacion simple (`sklearn.feature_extraction.text.TfidfVectorizer`).
- Multilingual E5 liviano: `intfloat/multilingual-e5-small`.
- Sentence-BERT multilingue liviano: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

Criterios:

- buen soporte para espanol;
- embeddings rapidos de calcular;
- posibilidad de correr localmente;
- comparacion transparente contra TF-IDF.

Implementacion inicial:

- indexar solo `question_es` de los casos `train`;
- para cada pregunta de `valid`/`test`, recuperar el caso de `train` mas parecido;
- devolver `answer_es` del vecino mas cercano como respuesta baseline;
- evaluar por caso unico, no por imagen duplicada, para que los casos con mas imagenes no pesen mas;
- reportar `sacreBLEU`, `chrF`, ejemplos manuales y vecinos top-k.

## Retrieval visual

Candidatos:

- CLIP;
- OpenCLIP;
- BiomedCLIP si esta disponible localmente o por Hugging Face;
- encoder visual del VLM elegido, si es facil extraer embeddings.

Criterios:

- calidad en imagenes dermatologicas;
- costo de indexar 2.945 imagenes;
- compatibilidad con GPU/CPU disponible;
- facilidad para reproducir embeddings.

## VLM zero-shot y LoRA

Candidatos del PDF:

- Qwen2.5-VL-3B;
- PaliGemma;
- LLaVA-Med como opcion biomedica;
- Qwen2-VL o InternVL si encajan mejor con el hardware.

Criterios:

- modelo suficientemente pequeno para entrenar con LoRA/QLoRA;
- buen seguimiento de instrucciones en espanol;
- facilidad de procesar imagen + texto;
- soporte estable en `transformers`;
- costo de inferencia razonable.

## Recomendacion inicial

Empezar por baselines de retrieval antes de fine-tuning. Son mas baratos,
permiten validar el dataset y generan una referencia fuerte para decidir si
vale la pena gastar GPU en LoRA/QLoRA.
