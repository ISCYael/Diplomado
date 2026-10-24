import cv2
import numpy as np
from numba import njit, prange
import matplotlib
import os
import time
# Cambiado a TkAgg para compatibilidad con tu entorno en Bazzite
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# =========================================================================
# FUNCIONES COMPILADAS EN CÓDIGO MÁQUINA CON NUMBA (@njit)
# =========================================================================

@njit(parallel=True, fastmath=True)
def calcular_distancias_minimas_sq(X, centroids_actuales):
    n_samples = X.shape[0]
    k_actual = centroids_actuales.shape[0]
    dist_sq = np.empty(n_samples, dtype=np.float32)
    
    for i in prange(n_samples):
        min_d = 1e30
        r, g, b = X[i, 0], X[i, 1], X[i, 2]
        for j in range(k_actual):
            dr = r - centroids_actuales[j, 0]
            dg = g - centroids_actuales[j, 1]
            db = b - centroids_actuales[j, 2]
            d = dr*dr + dg*dg + db*db
            if d < min_d:
                min_d = d
        dist_sq[i] = min_d
    return dist_sq

def inicializar_kmeans_plusplus_numba(X, K, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    centroids = np.empty((K, 3), dtype=np.float32)
    
    primer_indice = np.random.randint(0, n_samples)
    centroids[0] = X[primer_indice]
    
    for k in range(1, K):
        dist_sq = calcular_distancias_minimas_sq(X, centroids[:k])
        probabilidades = dist_sq / np.sum(dist_sq)
        siguiente_indice = np.random.choice(n_samples, p=probabilidades)
        centroids[k] = X[siguiente_indice]
        
    return centroids

@njit(parallel=True, fastmath=True)
def asignar_clusters_numba(X, centroids):
    n_samples = X.shape[0]
    K = centroids.shape[0]
    labels = np.empty(n_samples, dtype=np.int64)
    
    for i in prange(n_samples):
        min_dist = 1e30
        best_cluster = 0
        r, g, b = X[i, 0], X[i, 1], X[i, 2]
        
        for j in range(K):
            dr = r - centroids[j, 0]
            dg = g - centroids[j, 1]
            db = b - centroids[j, 2]
            dist = dr*dr + dg*dg + db*db
            if dist < min_dist:
                min_dist = dist
                best_cluster = j
                
        labels[i] = best_cluster
    return labels

@njit(fastmath=True)
def actualizar_centroides_numba(X, labels, K, old_centroids):
    new_centroids = np.zeros((K, 3), dtype=np.float32)
    counts = np.zeros(K, dtype=np.int64)
    
    for i in range(X.shape[0]):
        c = labels[i]
        new_centroids[c, 0] += X[i, 0]
        new_centroids[c, 1] += X[i, 1]
        new_centroids[c, 2] += X[i, 2]
        counts[c] += 1
        
    for j in range(K):
        if counts[j] > 0:
            new_centroids[j, 0] /= counts[j]
            new_centroids[j, 1] /= counts[j]
            new_centroids[j, 2] /= counts[j]
        else:
            new_centroids[j] = old_centroids[j]
            
    return new_centroids

# =========================================================================
# FUNCIÓN PRINCIPAL CON INTERACTIVIDAD
# =========================================================================

def graficar_imagen_y_rgb():
    print("🚀 Preparando compilador JIT de Numba...")

    # 1. Obtener ruta dinámica (Corregido a __file__)
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    # Apuntamos a tu imagen actual
    ruta_imagen = os.path.join(directorio_actual, 'images.jpeg')
    
    # 2. Cargar la imagen
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        print(f"Error: No se encontró la imagen en: {ruta_imagen}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. Preparar datos (250x250)
    img_pequena = cv2.resize(img_rgb, (250, 250), interpolation=cv2.INTER_NEAREST)
    X = np.ascontiguousarray(img_pequena.reshape(-1, 3), dtype=np.float32)
    
    K = 15 # Ajustado para el cubo de rubik
    epochs = 1000

    # Calentamiento JIT
    _ = asignar_clusters_numba(X[:10], X[:2])
    _ = actualizar_centroides_numba(X[:10], np.zeros(10, dtype=np.int64), 2, X[:2])

    inicio_tiempo = time.time()
    centroids = inicializar_kmeans_plusplus_numba(X, K, seed=42)
    
    for epoch in range(epochs):
        labels = asignar_clusters_numba(X, centroids)
        new_centroids = actualizar_centroides_numba(X, labels, K, centroids)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids

    print(f"⚡ Entrenamiento completado en {time.time() - inicio_tiempo:.4f} s.")
    
    # Reconstrucción en resolución original
    X_orig = np.ascontiguousarray(img_rgb.reshape(-1, 3), dtype=np.float32)
    labels_original = asignar_clusters_numba(X_orig, centroids)
    img_cuantizada = np.round(centroids[labels_original]).astype(np.uint8).reshape(img_rgb.shape)
    
    # Reorganizar etiquetas al tamaño 2D de la imagen original para crear máscaras rápidas
    labels_2d = labels_original.reshape(img_rgb.shape[0], img_rgb.shape[1])

    # Pre-calcular una versión oscurecida en blanco y negro para el fondo no seleccionado
    img_gris = cv2.cvtColor(img_cuantizada, cv2.COLOR_RGB2GRAY)
    img_fondo_atenuado = (np.stack([img_gris]*3, axis=-1) * 0.25).astype(np.uint8)

    # Preparar datos para el gráfico 3D
    indices_grafica = np.random.choice(X.shape[0], size=min(10000, X.shape[0]), replace=False)
    R = X[indices_grafica, 0]
    G = X[indices_grafica, 1]
    B = X[indices_grafica, 2]
    labels_grafica = labels[indices_grafica]
    colores_kmeans = centroids[labels_grafica] / 255.0

    # 4. Crear la ventana de visualización
    fig = plt.figure(figsize=(16, 5))
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Imagen Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(1, 3, 2)
    im_cuantizada_plot = ax2.imshow(img_cuantizada)
    ax2.set_title(f"Imagen Cuantizada (K={K})\n[Haz clic en un centroide 3D ➡️]")
    ax2.axis('off')
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    scatter_puntos = ax3.scatter(R, G, B, c=colores_kmeans, marker='.', alpha=0.3)
    
    # Habilitar "picker=True" y "pickradius=5" en los centroides para que detecten el clic del ratón
    scatter_centroides = ax3.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c=centroids / 255.0, s=120, marker='o', edgecolors='black', 
        linewidths=1.5, label='Centroides (Clic aquí)', picker=True, pickradius=5
    )
    
    ax3.set_xlabel('Canal Rojo (R)', color='red', fontweight='bold')
    ax3.set_ylabel('Canal Verde (G)', color='green', fontweight='bold')
    ax3.set_zlabel('Canal Azul (B)', color='blue', fontweight='bold')
    ax3.set_xlim([0, 255])
    ax3.set_ylim([0, 255])
    ax3.set_zlim([0, 255])
    ax3.set_title("Espacio RGB 3D (Interactivo)")
    ax3.legend()

    # =========================================================================
    # LÓGICA DE EVENTOS (CLIC EN CENTROIDES)
    # =========================================================================
    estado = {'cluster_activo': None}

    def restaurar_vista():
        estado['cluster_activo'] = None
        im_cuantizada_plot.set_data(img_cuantizada)
        ax2.set_title(f"Imagen Cuantizada (K={K})\n[Haz clic en un centroide 3D ➡️]")
        colores_rgba = np.column_stack([colores_kmeans, np.full(len(colores_kmeans), 0.3)])
        scatter_puntos.set_facecolors(colores_rgba)
        scatter_centroides.set_sizes(np.full(K, 120))
        fig.canvas.draw_idle()

    def al_hacer_clic_centroide(event):
        if event.artist != scatter_centroides:
            return
            
        cluster_idx = event.ind[0]
        
        if estado['cluster_activo'] == cluster_idx:
            restaurar_vista()
            return
            
        estado['cluster_activo'] = cluster_idx
        color_rgb = np.round(centroids[cluster_idx]).astype(int)
        
        mascara = (labels_2d == cluster_idx)
        porcentaje = (np.sum(mascara) / mascara.size) * 100
        
        img_resaltada = img_fondo_atenuado.copy()
        img_resaltada[mascara] = img_cuantizada[mascara] 
        
        im_cuantizada_plot.set_data(img_resaltada)
        ax2.set_title(
            f"Cluster #{cluster_idx} | RGB: {tuple(color_rgb)} ({porcentaje:.1f}% de la img)\n"
            f"[Clic en el mismo centroide o en la imagen para restaurar]"
        )
        
        alphas = np.where(labels_grafica == cluster_idx, 0.9, 0.02)
        colores_rgba = np.column_stack([colores_kmeans, alphas])
        scatter_puntos.set_facecolors(colores_rgba)
        
        tamanos = np.full(K, 60)
        tamanos[cluster_idx] = 300
        scatter_centroides.set_sizes(tamanos)
        
        print(f"🎯 Seleccionado Cluster #{cluster_idx} -> Color RGB: {tuple(color_rgb)} | Área: {porcentaje:.2f}%")
        fig.canvas.draw_idle()

    def al_hacer_clic_ventana(event):
        if event.inaxes in [ax1, ax2] and estado['cluster_activo'] is not None:
            restaurar_vista()

    fig.canvas.mpl_connect('pick_event', al_hacer_clic_centroide)
    fig.canvas.mpl_connect('button_press_event', al_hacer_clic_ventana)

    plt.tight_layout()
    print("✨ ¡Modo interactivo listo! Haz clic en las esferas del gráfico 3D para aislar regiones.")
    plt.show()

# Corregido a __name__ == "__main__"
if __name__ == "__main__":
    graficar_imagen_y_rgb()