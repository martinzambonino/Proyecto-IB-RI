# Sistema de Recuperación de Información

Proyecto de Recuperación de Información orientado a comparar modelos clásicos y modernos sobre un corpus textual.

El sistema permite ingresar consultas libres desde una interfaz de línea de comandos y muestra automáticamente los resultados recuperados por cuatro modelos:

1. Jaccard con vectores binarios.
2. TF-IDF con similitud coseno.
3. BM25.
4. Recuperación semántica con embeddings y FAISS.

---

## 1. Descripción general

Este proyecto implementa un motor de búsqueda básico sobre un corpus de noticias financieras y económicas. El usuario escribe una consulta en texto libre y el sistema devuelve rankings de documentos ordenados por relevancia.

La idea principal no es solo recuperar documentos, sino comparar cómo responde cada modelo ante la misma consulta.

Ejemplo de consulta:

```text
japan trade
```

El sistema muestra resultados separados para:

```text
[1] MODELO BINARIO - JACCARD
[2] MODELO VECTORIAL - TF-IDF + COSENO
[3] MODELO PROBABILÍSTICO - BM25
[4] MODELO SEMÁNTICO - EMBEDDINGS
```

---

## 2. Estructura del proyecto

La estructura esperada del proyecto es:

```text
.
├── proyectoIB.py
├── evaluacion.py
├── README.md
└── data/
    └── ModApte_train.csv
```

### Archivos principales

```text
proyectoIB.py
```

Contiene:

- carga del corpus,
- preprocesamiento,
- implementación de modelos de recuperación,
- índice invertido,
- recuperación semántica con FAISS,
- interfaz CLI.

```text
evaluacion.py
```

Contiene:

- consultas de prueba,
- generación de qrels desde la columna `topics`,
- cálculo de Precision,
- cálculo de Recall,
- cálculo de Average Precision,
- cálculo de MAP.

```text
data/ModApte_train.csv
```

Corpus usado por el sistema. Debe estar ubicado dentro de la carpeta `data`.

---

## 3. Requisitos

El proyecto usa Python y las siguientes librerías:

```text
pandas
numpy
scikit-learn
sentence-transformers
faiss-cpu
```

---

## 4. Instalación de dependencias

Desde una terminal, ejecutar:

```bash
pip install -r requirements.txt
```

Si el sistema usa `python3` y `pip3`, ejecutar:

```bash
pip3 install -r requirements.txt
```

---

## 5. Corpus utilizado

El sistema usa el archivo:

```text
data/ModApte_train.csv
```

El archivo debe contener al menos estas columnas:

```text
title
text
topics
```

### Significado de cada columna

```text
title
```

Título del documento. Se muestra en los resultados recuperados.

```text
text
```

Contenido textual del documento. Se usa para indexar, vectorizar y generar embeddings.

```text
topics
```

Categorías temáticas del documento. Se usan para la evaluación formal con qrels.

---

## 6. Cómo ejecutar el sistema

Desde la carpeta raíz del proyecto, ejecutar:

```bash
python proyectoIB.py
```

Si el sistema usa `python3`, ejecutar:

```bash
python3 proyectoIB.py
```

Al iniciar, el sistema carga el corpus, construye las estructuras de recuperación y genera los embeddings necesarios para la búsqueda semántica.

La primera ejecución puede tardar más porque el modelo de embeddings `all-MiniLM-L6-v2` puede descargarse y almacenarse en caché.

---

## 7. Interfaz de usuario

Al ejecutar el programa, se muestra una interfaz de línea de comandos con este formato:

```text
╔════════════════════════════════════════════════════════════╗
║        SISTEMA DE RECUPERACIÓN DE INFORMACIÓN             ║
║        Jaccard | TF-IDF | BM25 | Embeddings               ║
╚════════════════════════════════════════════════════════════╝

Escribe una consulta libre y presiona ENTER.
Comandos: :ayuda | :eval | :top N | :salir

Buscar>
```

El usuario puede escribir una consulta normal o un comando interno.

---

## 8. Consultas libres

Una consulta libre es cualquier texto escrito por el usuario.

Ejemplos:

```text
Buscar> coffee
Buscar> gold
Buscar> crude
Buscar> japan trade
Buscar> coffee export prices in brazil
Buscar> crude oil production and economic impact
```

Cuando se ingresa una consulta normal, el sistema ejecuta automáticamente los cuatro modelos y muestra los resultados por separado.

No es necesario escoger manualmente un modelo.

---

## 9. Comandos disponibles

### `:ayuda`

Muestra instrucciones de uso y ejemplos.

```text
Buscar> :ayuda
```

### `:top N`

Cambia la cantidad de resultados visibles por modelo.

Ejemplo:

```text
Buscar> :top 5
```

Después de ejecutar ese comando, cada modelo mostrará 5 resultados.

El valor máximo permitido es 10, porque los modelos recuperan hasta 10 resultados por consulta.

### `:eval`

Ejecuta la evaluación formal del sistema.

```text
Buscar> :eval
```

Este comando calcula métricas usando consultas de prueba y documentos relevantes conocidos.

### `:salir`

Cierra el sistema.

```text
Buscar> :salir
```

---

## 10. Evaluación del sistema

La evaluación formal se puede ejecutar de dos formas.

### Opción 1: desde la CLI

Ejecutar el programa:

```bash
python proyectoIB.py
```

Luego escribir:

```text
Buscar> :eval
```

### Opción 2: directamente desde el archivo de evaluación

```bash
python evaluacion.py
```

O:

```bash
python3 evaluacion.py
```

---

## 11. Consultas usadas en la evaluación

La evaluación usa cinco consultas basadas en categorías reales del corpus:

```text
coffee
gold
acq
earn
crude
```

Los documentos relevantes se obtienen desde la columna `topics`.

Un documento se considera relevante para una consulta si su columna `topics` contiene la categoría correspondiente.

Ejemplo:

```text
Consulta: coffee
Documento relevante: documento cuyo campo topics contiene coffee
```

---

## 12. Métricas calculadas

Para cada consulta y modelo se calculan:

```text
Precision
Recall
Average Precision
```

Para todo el sistema se calcula:

```text
MAP
```

### Precision

Mide qué proporción de los documentos recuperados son realmente relevantes.

```text
Precision = documentos relevantes recuperados / documentos recuperados
```

En este proyecto se evalúa sobre Top-10.

### Recall

Mide qué proporción del total de documentos relevantes fue recuperada.

```text
Recall = documentos relevantes recuperados / total de documentos relevantes
```

### Average Precision

Mide la calidad del ranking considerando las posiciones donde aparecen documentos relevantes.

### MAP

MAP significa `Mean Average Precision`.

Es el promedio del Average Precision de todas las consultas evaluadas.

---

## 13. Modelos implementados

### 13.1 Jaccard

Modelo basado en vectores binarios.

Representa documentos y consultas como conjuntos de términos.

La similitud se calcula con:

```text
Jaccard(D, Q) = |D ∩ Q| / |D ∪ Q|
```

Este modelo solo considera si un término aparece o no aparece. No usa frecuencia ni ponderación global del término.

---

### 13.2 TF-IDF + similitud coseno

Modelo vectorial basado en `TfidfVectorizer`.

Convierte documentos y consultas en vectores TF-IDF y calcula similitud coseno entre la consulta y cada documento.

Funciona bien cuando la consulta comparte términos exactos con los documentos relevantes.

---

### 13.3 BM25

Modelo probabilístico de recuperación.

La implementación usa:

- frecuencia de término,
- frecuencia documental inversa,
- longitud del documento,
- longitud promedio del corpus,
- parámetro `k1 = 1.5`,
- parámetro `b = 0.75`.

BM25 usa índice invertido para obtener documentos candidatos antes de calcular el score.

---

### 13.4 Embeddings + FAISS

Modelo de recuperación semántica.

Usa `sentence-transformers` con el modelo:

```text
all-MiniLM-L6-v2
```

Los documentos se transforman en embeddings y se almacenan en FAISS usando:

```text
IndexFlatIP
```

Los vectores se normalizan con L2 para que el producto interno funcione como similitud coseno.

Este modelo puede recuperar documentos relacionados por significado, aunque no compartan exactamente las mismas palabras con la consulta.

---
## 14. Decisiones de diseño

- **BM25 k1=1.5, b=0.75**: valores estándar de la literatura; b=0.75 penaliza documentos largos sin eliminar su aporte.
- **Modelo de embeddings `all-MiniLM-L6-v2`**: equilibrio entre velocidad y calidad semántica para corpus en inglés de tamaño mediano.
- **FAISS IndexFlatIP**: búsqueda exacta por producto interno; apropiado dado el tamaño del corpus.
- **Índice invertido en Jaccard y BM25**: reduce candidatos antes de calcular scores, mejora eficiencia.

---
## 15. Índice invertido

El sistema construye un índice invertido con la estructura:

```text
término → documento → frecuencia
```

Ejemplo:

```text
coffee → doc_1: 3, doc_8: 1, doc_20: 5
```

Esto significa que el término `coffee` aparece:

```text
3 veces en doc_1
1 vez en doc_8
5 veces en doc_20
```

El índice invertido se usa para recuperar documentos candidatos en modelos basados en términos.

---

## 16. Diferencia entre búsqueda libre y evaluación

La búsqueda libre sirve para usar el sistema como buscador.

Ejemplo:

```text
Buscar> japan trade
```

Esta búsqueda muestra rankings de los cuatro modelos, pero no calcula métricas.

La evaluación formal se ejecuta con:

```text
Buscar> :eval
```

La evaluación sí calcula métricas porque usa consultas de prueba con documentos relevantes conocidos.

La diferencia principal es:

```text
Consulta libre → muestra resultados.
:eval          → mide calidad con qrels y métricas.
```

---

## 17. Ejemplo de uso

Ejecutar el sistema:

```bash
python proyectoIB.py
```

Cambiar el número de resultados a 5:

```text
Buscar> :top 5
```

Realizar una consulta:

```text
Buscar> japan trade
```

Ejecutar evaluación:

```text
Buscar> :eval
```

Salir:

```text
Buscar> :salir
```

---

## 18. Notas importantes

- El sistema no usa Elasticsearch, Solr ni Whoosh.
- La recuperación semántica usa FAISS.
- Los modelos clásicos se implementan o configuran directamente en el código.
- La primera ejecución puede tardar por la generación de embeddings.
- El archivo `ModApte_train.csv` debe estar dentro de la carpeta `data`.
- Si el archivo del corpus no está en la ruta correcta, el sistema no podrá iniciar.

---

## 19. Posibles problemas

### Error: no se encontró el archivo del corpus

Verificar que exista esta ruta:

```text
data/ModApte_train.csv
```

La estructura debe ser:

```text
.
├── proyectoIB.py
├── evaluacion.py
└── data/
    └── ModApte_train.csv
```

### La primera ejecución tarda mucho

Esto puede pasar porque se están generando embeddings para todos los documentos.

También puede ocurrir si el modelo `all-MiniLM-L6-v2` se descarga por primera vez.

### Advertencias de Hugging Face en Windows

Puede aparecer una advertencia sobre `symlinks` o caché.

Mientras el sistema continúe y muestre:

```text
Total de documentos indexados en FAISS
Sistema listo
```

la advertencia no impide ejecutar el proyecto.

---

## 20. Autores

Proyecto realizado por:

```text
Erick Carcelen
Kenneth Yar
Martin Zambonino
```

Materia:

```text
Recuperación de Información
```
