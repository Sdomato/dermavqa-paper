# Risks

## Riesgos metodologicos

- Las referencias son heterogeneas: algunas son diagnosticos de una palabra y
  otras son recomendaciones mas largas.
- Usar siempre la respuesta mas larga puede favorecer verbosidad por encima de
  precision medica.
- Las traducciones al espanol pueden introducir errores o estilo poco natural.
- El split local debe auditarse antes de reportar resultados.
- Las metricas automaticas pueden penalizar respuestas correctas redactadas de
  forma distinta.

## Riesgos medicos

- El modelo puede generar diagnosticos no respaldados por la imagen o consulta.
- Las respuestas recuperadas pueden contener recomendaciones inseguras.
- La ausencia de contexto clinico completo limita cualquier conclusion medica.
- El sistema debe describirse como experimento de NLP/VQA, no como herramienta
  de decision clinica.

## Riesgos operativos

- Fine-tuning de VLMs puede exceder el presupuesto de GPU.
- Algunos modelos candidatos pueden tener soporte incompleto en el entorno.
- BiomedCLIP u otros checkpoints pueden requerir dependencias adicionales.
- Imagenes grandes o muchas imagenes por caso pueden aumentar costo de
  inferencia.

## Mitigaciones

- Empezar con retrieval textual y visual antes de LoRA.
- Mantener prompts con cautela medica y sin afirmar certeza diagnostica.
- Reportar limitaciones explicitamente.
- Guardar predicciones, configuraciones y seeds.
- Separar validacion de test y no ajustar hiperparametros en test.
