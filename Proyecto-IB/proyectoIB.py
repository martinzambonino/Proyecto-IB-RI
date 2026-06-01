import pandas as pd
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def normalizador_base(texto):
    #Convertir todo el texto a minúsculas.
    return str(texto).lower()

def tokenizador_base(texto):

    # Extraer secuencias de caracteres alfanuméricos
    tokens = re.findall(r'\b\w+\b', str(texto))
   
    # Filtrar contra la lista de ENGLISH_STOP_WORDS
    tokens_limpios = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
   
    return tokens_limpios

class MotorRecuperacion:
    def __init__(self, ruta_train_csv):
         # Leer el corpus en texto plano
        df = pd.read_csv(ruta_train_csv)

        # Llenamos valores nulos con strings vacíos para evitar errores
        self.textos = df['text'].fillna("").tolist()
        self.titulos = df['title'].fillna("No Title").tolist()
       
        self.vectorizador_tfidf = TfidfVectorizer(
            preprocessor=normalizador_base,
            tokenizer=tokenizador_base,
            token_pattern=None
        )

        self.matriz_tfidf = self.vectorizador_tfidf.fit_transform(self.textos)
       
        # ESPACIO PARA EL RESTO DE MODELOS (Inicialización)
  
        # JACCARD
        self.modelo_jaccard = ModeloJaccard(self.textos, self.titulos)
        # BM25
        self.modelo_bm25 = ModeloBM25(self.textos, self.titulos)
        
        self.modelo_semantico = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = self.modelo_semantico.encode(self.textos, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dimensiones = embeddings.shape[1] 
        self.base_vectorial = faiss.IndexFlatIP(dimensiones) 
        faiss.normalize_L2(embeddings)
        self.base_vectorial.add(embeddings)
        print(f"Total de documentos indexados en FAISS: {self.base_vectorial.ntotal}")

    def buscar_tfidf(self, consulta_texto):

            # Transformar la consulta de texto libre
            vector_consulta = self.vectorizador_tfidf.transform([consulta_texto])
            # Calcular similitud coseno contra todos los documentos
            similitudes = cosine_similarity(vector_consulta, self.matriz_tfidf).flatten()

            resultados = []
            for indice, score in enumerate(similitudes):
                if score > 0.0:
                    resultados.append((score, self.titulos[indice], self.textos[indice]))
                   
            # Mostrar un ranking de documentos ordenados por relevancia
            resultados.sort(reverse=True, key=lambda x: x[0])
            return resultados[:10]
    
    def buscar_jaccard(self, consulta_texto):
        return self.modelo_jaccard.buscar(consulta_texto, top_k=10)
    def buscar_bm25(self, consulta_texto):
        return self.modelo_bm25.buscar(consulta_texto, top_k=10)
    
    def buscar_semantico(self, consulta_texto):
        # Generar embedding para la consulta
        vector_consulta = self.modelo_semantico.encode([consulta_texto])
        vector_consulta = np.array(vector_consulta).astype("float32")
        faiss.normalize_L2(vector_consulta)
        
        top_k = 10
        distancias, indices = self.base_vectorial.search(vector_consulta, top_k)
        
        # Emparejar resultados
        resultados = []
        for i in range(top_k):
            score = distancias[0][i]
            idx_doc = indices[0][i]
            if idx_doc != -1: 
                resultados.append((score, self.titulos[idx_doc], self.textos[idx_doc]))
                
        return resultados


class ModeloJaccard:
 
    def __init__(self, textos: list[str], titulos: list[str]):

        self.titulos = titulos
        self.textos = textos
 
        # ── Paso 1: tokenización completa para obtener frecuencias y vectores binarios
        documentos_tokenizados = [
            tokenizador_base(normalizador_base(texto)) for texto in textos
        ]

        self.conjuntos_docs: list[set[str]] = [
            set(tokens) for tokens in documentos_tokenizados
        ]
 
        # ── Paso 2: índice invertido  término → {doc_id: frecuencia}
        self.indice_invertido: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for doc_id, tokens in enumerate(documentos_tokenizados):
            for token in tokens:
                self.indice_invertido[token][doc_id] += 1
 
 
    # ── Similitud Jaccard ─────────────────────────────────────────
    @staticmethod
    def similitud_jaccard(set_doc: set, set_consulta: set) -> float:
        """
        Jaccard(D, Q) = |D ∩ Q| / |D ∪ Q|
 
        Retorna 0.0 cuando ambos conjuntos son vacíos.
        """
        interseccion = len(set_doc & set_consulta)
        if interseccion == 0:
            return 0.0
        union = len(set_doc | set_consulta)
        return interseccion / union
 
    # ── Búsqueda con índice invertido ────────────────────────────
    def buscar(self, consulta_texto: str, top_k: int = 10) -> list[tuple]:

        set_consulta = set(tokenizador_base(normalizador_base(consulta_texto)))
 
        if not set_consulta:
            return []
 
        # ── Candidatos: docs con al menos un término en común
        candidatos: set[int] = set()
        for termino in set_consulta:
            if termino in self.indice_invertido:
                candidatos |= set(self.indice_invertido[termino].keys())
 
        if not candidatos:
            return []
 
        # ── Calcular Jaccard solo para candidatos
        resultados = []
        for doc_id in candidatos:
            score = self.similitud_jaccard(self.conjuntos_docs[doc_id], set_consulta)
            if score > 0.0:
                resultados.append((score, self.titulos[doc_id], self.textos[doc_id]))
 
        # ── Ranking descendente
        resultados.sort(reverse=True, key=lambda x: x[0])
        return resultados[:top_k]
 
    # ── Estadísticas del índice ───────────────────────────────────
    def info_termino(self, termino: str) -> dict:
        """Devuelve estadísticas de un término en el índice."""
        t = normalizador_base(termino)
        docs = self.indice_invertido.get(t, {})
        return {
            "termino": t,
            "doc_frequency": len(docs),
            "doc_ids": sorted(docs.keys())[:10],   # solo los primeros 10
        }
        
        
class ModeloBM25:
    def __init__(self, textos, titulos, k1=1.5, b=0.75):
        self.textos = textos
        self.titulos = titulos

        self.k1 = k1
        self.b = b

        # Tokenización usando el mismo tokenizador del sistema
        self.documentos_tokenizados = [
            tokenizador_base(normalizador_base(texto)) for texto in textos
        ]

        # Número total de documentos
        self.N = len(self.documentos_tokenizados)

        # Longitud de cada documento
        self.longitudes_documentos = [
            len(documento) for documento in self.documentos_tokenizados
        ]

        # Longitud promedio del corpus
        self.avgdl = sum(self.longitudes_documentos) / self.N

        # Frecuencia de términos por documento
        self.frecuencias_documentos = [
            Counter(documento) for documento in self.documentos_tokenizados
        ]

        # df: cantidad de documentos donde aparece cada término
        self.df = defaultdict(int)

        for frecuencias in self.frecuencias_documentos:
            for termino in frecuencias.keys():
                self.df[termino] += 1

        # IDF de cada término
        self.idf = {}

        for termino, frecuencia_documental in self.df.items():
            self.idf[termino] = math.log(
                1 + (self.N - frecuencia_documental + 0.5) /
                (frecuencia_documental + 0.5)
            )

        # Índice invertido: término → {doc_id: frecuencia}
        self.indice_invertido: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for doc_id, frecuencias in enumerate(self.frecuencias_documentos):
            for termino, freq in frecuencias.items():
                self.indice_invertido[termino][doc_id] = freq

    def calcular_score(self, consulta_tokens, indice_documento):
        score = 0.0

        frecuencias = self.frecuencias_documentos[indice_documento]
        longitud_documento = self.longitudes_documentos[indice_documento]

        for termino in consulta_tokens:
            if termino not in self.idf:
                continue

            frecuencia_termino = frecuencias.get(termino, 0)

            if frecuencia_termino == 0:
                continue

            numerador = frecuencia_termino * (self.k1 + 1)

            denominador = frecuencia_termino + self.k1 * (
                1 - self.b + self.b * (longitud_documento / self.avgdl)
            )

            score += self.idf[termino] * (numerador / denominador)

        return score

    def buscar(self, consulta_texto, top_k=10):
        consulta_tokens = tokenizador_base(normalizador_base(consulta_texto))

        if not consulta_tokens:
            return []

        # Usar índice invertido para obtener documentos candidatos
        candidatos: set[int] = set()
        for termino in consulta_tokens:
            if termino in self.indice_invertido:
                candidatos |= set(self.indice_invertido[termino].keys())

        if not candidatos:
            return []

        resultados = []

        for indice_documento in candidatos:
            score = self.calcular_score(consulta_tokens, indice_documento)

            if score > 0.0:
                resultados.append((
                    score,
                    self.titulos[indice_documento],
                    self.textos[indice_documento]
                ))

        resultados.sort(reverse=True, key=lambda x: x[0])

        return resultados[:top_k]

def mostrar_inicio():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        SISTEMA DE RECUPERACIÓN DE INFORMACIÓN             ║")
    print("║        Jaccard | TF-IDF | BM25 | Embeddings               ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print("Escribe una consulta libre y presiona ENTER.")
    print("Comandos: :ayuda | :eval | :top N | :salir")
    print()


def mostrar_ayuda():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                         AYUDA                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print("Uso del sistema:")
    print()
    print("Consulta normal  → muestra resultados de los 4 modelos.")
    print(":eval            → ejecuta evaluación formal con métricas.")
    print(":top N           → cambia cantidad de resultados visibles.")
    print(":ayuda           → explica el uso.")
    print(":salir           → cierra el sistema.")
    print()
    print("Ejemplos de palabras libres:")
    print()
    print("Buscar> coffee")
    print("Buscar> gold")
    print("Buscar> crude")
    print()
    print("Ejemplos de consultas más largas:")
    print()
    print("Buscar> coffee export prices in brazil")
    print("Buscar> gold market changes and international prices")
    print("Buscar> crude oil production and economic impact")
    print()


def crear_resumen(texto, limite=160):
    texto = str(texto).replace("\n", " ").strip()
    texto = " ".join(texto.split())

    if len(texto) > limite:
        texto = texto[:limite].rstrip() + "..."

    return texto


def imprimir_resultados_modelo(titulo_modelo, resultados, top_k):
    print(titulo_modelo)
    print("-" * 60)

    resultados_mostrados = resultados[:top_k]

    if not resultados_mostrados:
        print("No se encontraron documentos relevantes.")
        print()
        return

    for i, resultado in enumerate(resultados_mostrados, start=1):
        score, titulo, texto = resultado
        resumen = crear_resumen(texto)

        print(f"#{i}  {titulo}")
        print(f"    Score: {float(score):.4f}")

        if resumen:
            print(f"    {resumen}")

        print()


def imprimir_resultados_consulta(consulta, resultados_jaccard, resultados_tfidf, resultados_bm25, resultados_semantico, top_k):
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print(f"║ Consulta: {consulta[:48]:<48} ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print(f"Top K actual: {top_k}")
    print()

    imprimir_resultados_modelo("[1] MODELO BINARIO - JACCARD", resultados_jaccard, top_k)
    imprimir_resultados_modelo("[2] MODELO VECTORIAL - TF-IDF + COSENO", resultados_tfidf, top_k)
    imprimir_resultados_modelo("[3] MODELO PROBABILÍSTICO - BM25", resultados_bm25, top_k)
    imprimir_resultados_modelo("[4] MODELO SEMÁNTICO - EMBEDDINGS", resultados_semantico, top_k)


def procesar_comando_top(consulta, top_k_actual):
    partes = consulta.split()

    if len(partes) != 2:
        print()
        print("Uso incorrecto del comando.")
        print()
        print("Formato correcto:")
        print(":top N")
        print()
        print("Ejemplo:")
        print("Buscar> :top 5")
        print()
        return top_k_actual

    try:
        nuevo_top_k = int(partes[1])
    except ValueError:
        print()
        print("El valor de N debe ser un número entero mayor que 0.")
        print()
        print("Ejemplo:")
        print("Buscar> :top 3")
        print()
        return top_k_actual

    if nuevo_top_k <= 0:
        print()
        print("El valor de N debe ser mayor que 0.")
        print()
        print("Ejemplo:")
        print("Buscar> :top 3")
        print()
        return top_k_actual

    if nuevo_top_k > 10:
        print()
        print("El sistema actualmente recupera hasta 10 resultados por modelo.")
        print("Por eso, el valor máximo permitido para :top N es 10.")
        print()
        print("Ejemplo:")
        print("Buscar> :top 10")
        print()
        return top_k_actual

    print()
    print("Top K actualizado correctamente.")
    print()
    print(f"Ahora se mostrarán {nuevo_top_k} resultados por cada modelo.")
    print()

    return nuevo_top_k


def ejecutar_cli():
    archivo_corpus = './data/ModApte_train.csv'
    top_k = 3

    try:
        print("Cargando motor y procesando índices...")
        print("Esto puede tardar unos momentos.")
        print()

        motor = MotorRecuperacion(archivo_corpus)

        print()
        print("Sistema listo.")
        print()

    except FileNotFoundError:
        print(f"\nERROR: No se encontró el archivo '{archivo_corpus}'.")
        return

    mostrar_inicio()

    while True:
        consulta = input("Buscar> ").strip()

        if not consulta:
            print()
            print("Ingresa una consulta o escribe :ayuda para ver instrucciones.")
            print()
            continue

        consulta_minuscula = consulta.lower()

        if consulta_minuscula == ":salir":
            print()
            print("Cerrando sistema...")
            print()
            print("Gracias por usar el Sistema de Recuperación de Información.")
            break

        if consulta_minuscula == ":ayuda":
            mostrar_ayuda()
            continue

        if consulta_minuscula.startswith(":top"):
            top_k = procesar_comando_top(consulta_minuscula, top_k)
            continue

        if consulta_minuscula == ":eval":
            print()
            print("╔════════════════════════════════════════════════════════════╗")
            print("║              EVALUACIÓN FORMAL DE MODELOS                 ║")
            print("╚════════════════════════════════════════════════════════════╝")
            print()

            try:
                from evaluacion import ejecutar_evaluacion
                ejecutar_evaluacion()
            except Exception as error:
                print()
                print("Ocurrió un error al ejecutar la evaluación.")
                print(f"Detalle: {error}")
                print()

            print()
            continue

        if consulta.startswith(":"):
            print()
            print(f"Comando no reconocido: {consulta}")
            print()
            print("Escribe :ayuda para ver los comandos disponibles.")
            print()
            continue

        try:
            resultados_jaccard = motor.buscar_jaccard(consulta)
            resultados_tfidf = motor.buscar_tfidf(consulta)
            resultados_bm25 = motor.buscar_bm25(consulta)
            resultados_semantico = motor.buscar_semantico(consulta)

            imprimir_resultados_consulta(
                consulta=consulta,
                resultados_jaccard=resultados_jaccard,
                resultados_tfidf=resultados_tfidf,
                resultados_bm25=resultados_bm25,
                resultados_semantico=resultados_semantico,
                top_k=top_k
            )

        except Exception as error:
            print()
            print("Ocurrió un error al procesar la consulta.")
            print(f"Detalle: {error}")
            print()

if __name__ == "__main__":
    ejecutar_cli()
