import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Agregado para que funcione en WSL
import matplotlib.pyplot as plt

def kmeans_visual(k=3, epochs=10):
    # 1. Generar una nube de puntos tipo "rocío" con ruido
    np.random.seed(42)
    # Creamos 3 grupos base de datos
    c1 = np.random.randn(30, 2) + np.array([2, 2])
    c2 = np.random.randn(30, 2) + np.array([-2, -2])
    c3 = np.random.randn(30, 2) + np.array([2, -2])
    X = np.vstack([c1, c2, c3]) # Nube de puntos total con ruido mezclado

    # 2. Inicializar centroides de forma aleatoria (de otro color en la gráfica)
    # Seleccionamos k puntos aleatorios de los mismos datos como centroides iniciales
    random_indices = np.random.choice(len(X), k, replace=False)
    centroids = X[random_indices]

    plt.figure(figsize=(7, 7))

    for epoch in range(epochs):
        # 3. Calcular la Distancia Euclidiana de cada punto a cada centroide
        # Usamos broadcasting de NumPy para calcular distancias
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # 4. Asignar cada punto al centroide más cercano (menor distancia)
        labels = np.argmin(distances, axis=1)

        # --- DIBUJAR LA GRÁFICA EN CADA ÉPOCA ---
        plt.clf()
        
        # Dibujar los puntos asignados a cada grupo con colores diferentes
        colores = ['teal', 'purple', 'gold']
        for i in range(k):
            puntos_grupo = X[labels == i]
            plt.scatter(puntos_grupo[:, 0], puntos_grupo[:, 1], c=colores[i], s=40, alpha=0.6, edgecolors='black')
            
        # Dibujar los centroides (Las X rojas grandes)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black', label=f'Centroides (Época {epoch + 1})')

        plt.title(f"K-Means Iterativo - Época {epoch + 1}\n(Los centroides buscan el centro de masa de la nube)")
        plt.xlabel("Dimensión X1")
        plt.ylabel("Dimensión X2")
        plt.legend(loc='upper right')
        
        plt.draw()
        plt.pause(1.0) # Pausa dramática de 1 segundo para ver el movimiento en vivo

        # 5. Actualizar la posición de los centroides (calcular la media de los puntos asignados)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Condición de parada: si los centroides ya no se mueven (convergencia)
        if np.all(centroids == new_centroids):
            print(f"¡K-Means convergió perfectamente en la época {epoch + 1}!")
            break
            
        centroids = new_centroids

    plt.show()

# Ejecutar la simulación en clase
if __name__ == "__main__":
    print("Iniciando visualización interactiva de K-Means...")
    kmeans_visual(k=3, epochs=8)