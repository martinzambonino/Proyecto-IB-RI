import pandas as pd
from proyectoIB import MotorRecuperacion

def calcular_metricas(resultados_obtenidos, qrels_reales, top_k=10):
    # Extraer los títulos de los resultados recuperados
    titulos_recuperados = [res[1] for res in resultados_obtenidos[:top_k]]
    
    relevantes_recuperados = 0
    suma_precisiones = 0.0
    
    # Evaluar relevancia posición por posición para el Average Precision (AP)
    for rank, titulo in enumerate(titulos_recuperados, start=1):
        if titulo in qrels_reales:
            relevantes_recuperados += 1
            suma_precisiones += relevantes_recuperados / rank
            
    # Cálculos finales
    precision = relevantes_recuperados / top_k
    total_reales = len(qrels_reales)
    
    recall = relevantes_recuperados / total_reales if total_reales > 0 else 0
    ap = suma_precisiones / total_reales if total_reales > 0 else 0
    
    return precision, recall, ap

def ejecutar_evaluacion():
    archivo_corpus = './data/ModApte_train.csv'
    
    print("Cargando motor y procesando índices (esto tomará unos momentos)...")
    motor = MotorRecuperacion(archivo_corpus)
    
    df = pd.read_csv(archivo_corpus)
    df['title'] = df['title'].fillna("No Title")
    df['topics'] = df['topics'].fillna("")
    
    # Consultas de prueba basadas en las categorías reales del dataset
    consultas_prueba = ["coffee", "gold", "acq", "earn", "crude"]
    
    # Almacenamiento de APs para calcular el MAP final
    metricas = {
        "TF-IDF": [],
        "Jaccard": [],
        "BM25": [],
        "Semántico": []
    }
    
    print("\n" + "="*60)
    print("INICIANDO EVALUACIÓN DE MODELOS (TOP 10)")
    print("="*60)
    
    for consulta in consultas_prueba:
        print(f"\nConsulta: '{consulta}'")
        
        # Generar QRELS: Identificar documentos que realmente pertenecen al tópico
        qrels_reales = set(df[df['topics'].str.contains(f"'{consulta}'", na=False)]['title'])
        print(f"Documentos relevantes totales en el corpus: {len(qrels_reales)}")
        
        if len(qrels_reales) == 0:
            continue
            
        # Ejecutar búsqueda en los 4 modelos
        resultados_modelos = {
            "TF-IDF": motor.buscar_tfidf(consulta),
            "Jaccard": motor.buscar_jaccard(consulta),
            "BM25": motor.buscar_bm25(consulta),
            "Semántico": motor.buscar_semantico(consulta)
        }
        
        # Calcular e imprimir métricas por modelo para la consulta actual
        for nombre, resultados in resultados_modelos.items():
            p, r, ap = calcular_metricas(resultados, qrels_reales)
            metricas[nombre].append(ap)
            print(f"  {nombre:10} -> Precisión: {p:.2f} | Recall: {r:.4f} | AP: {ap:.4f}")
            
    print("\n" + "="*60)
    print("RESULTADOS FINALES DEL SISTEMA (MAP)")
    print("="*60)
    
    # Calcular y mostrar el Mean Average Precision (MAP) de cada modelo
    for modelo, lista_ap in metricas.items():
        map_score = sum(lista_ap) / len(lista_ap) if lista_ap else 0
        print(f"MAP {modelo:10}: {map_score:.4f}")

if __name__ == "__main__":
    ejecutar_evaluacion()