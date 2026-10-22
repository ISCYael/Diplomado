import numpy as np

def generar_archivos_projector():
    print("Generando datos para TensorFlow Projector...")
    
    # 1. Generar datos ficticios en 3 Dimensiones (3 variables)
    np.random.seed(42)
    
    # Creamos 3 grupos de datos (150 puntos en total)
    gatos = np.random.randn(50, 3) + np.array([3, 3, 3])
    perros = np.random.randn(50, 3) + np.array([-3, -3, -3])
    aves = np.random.randn(50, 3) + np.array([3, -3, 3])
    
    # Juntamos todos los vectores en una sola matriz
    vectores = np.vstack([gatos, perros, aves])
    
    # Creamos las etiquetas correspondientes a cada punto
    metadatos = ["Gatos"] * 50 + ["Perros"] * 50 + ["Aves"] * 50

    # 2. Guardar el archivo de Vectores (Puros números, sin encabezado)
    with open("vectores.tsv", "w", encoding="utf-8") as f_vec:
        for vector in vectores:
            # Convierte los números a texto separados por tabulación (\t)
            linea = "\t".join([str(val) for val in vector])
            f_vec.write(linea + "\n")
            
    # 3. Guardar el archivo de Metadatos (Etiquetas)
    with open("metadatos.tsv", "w", encoding="utf-8") as f_meta:
        f_meta.write("Clase\n") # Encabezado
        for etiqueta in metadatos:
            f_meta.write(etiqueta + "\n")
            
    print("¡Listo! Se crearon 'vectores.tsv' y 'metadatos.tsv' en tu carpeta.")

if __name__ == "__main__":
    generar_archivos_projector()