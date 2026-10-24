import cv2
import numpy as np
import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def graficar_imagen_y_rgb():
    print("Procesando imagen y calculando coordenadas 3D...")
    
    # Ruta exacta de tu imagen
    ruta_imagen = '/home/Fxzzy/Documentos/Diplomado/images.jpeg'
    
    # 1. Cargar la imagen usando la ruta directa
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        print(f"Error: No se encontró la imagen en: {ruta_imagen}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Preparar datos: Redu'tamaño y aplanar para K-Means y gráfico 3D
    img_pequena = cv2.resize(img_rgb, (100, 100))
    X = img_pequena.reshape(-1, 3) 
    
    # =========================================================================
    # IMPLEMENTACIÓN DE K-MEANS MANUAL (Basado en la clase)
    # =========================================================================
    
    K = 15
    epochs = 1000 

    # 1. Inicializar centroides seleccionando K píxeles aleatorios de la matriz X
    np.random.seed(42)
    random_indices = np.random.choice(X.shape[0], K, replace=False)
    centroids = X[random_indices].astype(float)
    
    print(f"=== INICIANDO K-MEANS PARA CUANTIZACIÓN DE COLOR (K={K}) ===")
    print(f"Centroides iniciales (Colores RGB):\n{centroids.astype(int)}\n")
    
    for epoch in range(epochs):
        # 2. CÁLCULO DE LA MATRIZ DE DISTANCIAS EUCLIDIANAS
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # 3. MATRIZ DE PERTENENCIA / ASIGNACIÓN
        labels = np.argmin(distances, axis=1)
        
        if epoch == 0:
            print(f"Dimensiones de la matriz de distancias: {distances.shape}")
            print(f"Primeros 5 registros de la Matriz de Pertenencia: {labels[:5]}")
            print("-" * 60)
            
        # 4. ACTUALIZACIÓN DE CENTROIDES
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(K)])
        
        # Condición de convergencia
        if np.allclose(centroids, new_centroids):
            print(f"¡K-Means convergió matemáticamente en la época {epoch + 1}!")
            break
            
        centroids = new_centroids
        
    print("¡Proceso de agrupamiento por K-Means completado con éxito!")
    
    # =========================================================================
    
    # Reconstruir imagen cuantizada
    img_cuantizada = centroids[labels].astype(np.uint8).reshape(img_pequena.shape)

    # Preparar datos para el gráfico 3D
    R = X[:, 0]
    G = X[:, 1]
    B = X[:, 2]

    # 3. Crear ventana con tres secciones
    fig = plt.figure(figsize=(15, 5))
    
    # --- Lado Izquierdo: Imagen Original ---
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_pequena)
    ax1.set_title("Imagen Original")
    ax1.axis('off')
    
    # --- Centro: Imagen Cuantizada (K-Means) ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img_cuantizada)
    ax2.set_title(f"Imagen Cuantizada (K={K})")
    ax2.axis('off')
    
    # --- Lado Derecho: Gráfica 3D con colores K-Means ---
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    colores_kmeans = centroids[labels] / 255.0
    ax3.scatter(R, G, B, c=colores_kmeans, marker='.', alpha=0.5)
    
    ax3.set_xlabel('Canal Rojo (R)', color='red', fontweight='bold')
    ax3.set_ylabel('Canal Verde (G)', color='green', fontweight='bold')
    ax3.set_zlabel('Canal Azul (B)', color='blue', fontweight='bold')
    
    ax3.set_xlim([0, 255])
    ax3.set_ylim([0, 255])
    ax3.set_zlim([0, 255])
    ax3.set_title("Espacio RGB 3D (K-Means)")
    
    plt.tight_layout()
    print("¡Listo! Puedes usar el ratón para rotar el gráfico 3D.")
    plt.show()

if __name__ == "__main__":
    graficar_imagen_y_rgb() 