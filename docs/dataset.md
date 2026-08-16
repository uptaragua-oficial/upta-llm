# Dataset UPTA-LLM

## 1. Descripción

Dataset conversacional para el ajuste fino supervisado (SFT) del modelo
Qwen3-8B mediante QLoRA.

El dataset fue construido a partir de información institucional de la
Universidad Politécnica Territorial del Estado Aragua "Federico Brito
Figueroa" (UPT Aragua).

La información procede de:

- Documentos institucionales.
- Información oficial publicada en el portal web institucional.
- Documentos PDF institucionales.
- Información académica y administrativa seleccionada del portal.

---

## 2. Pipeline de construcción

El dataset fue construido mediante cuatro etapas principales:

1. Limpieza y curación del corpus institucional.
2. Curación del contenido seleccionado del portal web.
3. Integración y normalización semántica de información procedente de
   portal y documentos PDF.
4. Generación de preguntas y respuestas a partir de fragmentos semánticos.

Los notebooks utilizados fueron:

- `01_textoLimpioUPTA.ipynb`
- `02_portalTextoLimpio.ipynb`
- `03_pdfYPortal.ipynb`
- `04_generacionQA.ipynb`

---

## 3. Corpus semántico

Archivo principal:

```text
data/corpus/fragmentos_semanticos.jsonl