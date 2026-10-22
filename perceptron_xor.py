import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class RedNeuronalVisualXOR:
    def __init__(self, learning_rate=1.2, epochs=10000):
        self.lr = learning_rate
        self.epochs = epochs
        # 4 Neuronas Ocultas para mayor potencia cerebral
        self.W1 = np.random.uniform(-1, 1, (2, 4))
        self.b1 = np.random.uniform(-1, 1, (1, 4))
        self.W2 = np.random.uniform(-1, 1, (4, 1))
        self.b2 = np.random.uniform(-1, 1, (1, 1))

    def predict(self, X):
        hidden_layer = sigmoid(np.dot(X, self.W1) + self.b1)
        output_layer = sigmoid(np.dot(hidden_layer, self.W2) + self.b2)
        return output_layer

    def train_and_animate(self, X, y):
        fig, ax = plt.subplots(figsize=(8, 7))
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]

        print("Entrenando Red Neuronal Turbo (XOR) en cámara lenta...")
        
        for epoch in range(self.epochs):
            # --- FORWARD PASS ---
            hidden_layer = sigmoid(np.dot(X, self.W1) + self.b1)
            output_layer = sigmoid(np.dot(hidden_layer, self.W2) + self.b2)

            # --- BACKPROPAGATION ---
            error_output = y - output_layer
            d_output = error_output * sigmoid_derivative(output_layer)
            
            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * sigmoid_derivative(hidden_layer)

            # Ajuste de pesos
            self.W2 += hidden_layer.T.dot(d_output) * self.lr
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
            self.W1 += X.T.dot(d_hidden) * self.lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

            # --- VERIFICACIÓN ESTRICTA PARA FRENAR ---
            error_absoluto = np.abs(error_output)
            es_exitoso = np.all(error_absoluto < 0.1)

            # --- DIBUJAR ANIMACIÓN (LENTO Y DETALLADO) ---
            # El % 10 hace que dibuje un cuadro cada 10 intentos
            if epoch % 10 == 0 or es_exitoso:
                ax.clear()
                predicciones_grid = self.predict(grid).reshape(xx.shape)
                ax.contourf(xx, yy, predicciones_grid, levels=50, cmap='RdBu', alpha=0.6)
                
                if es_exitoso:
                    ax.set_title(f"Solución Multicapa para XOR", fontsize=12)
                    ax.text(0.5, 1.35, f"✓ ¡ÉXITO LOGRADO EN ÉPOCA {epoch}! ✓", 
                            color='white', weight='bold', fontsize=14, ha='center',
                            bbox=dict(facecolor='#2ECC71', edgecolor='green', boxstyle='round,pad=0.5'))
                    
                    for spine in ax.spines.values():
                        spine.set_color('#2ECC71')
                        spine.set_linewidth(5)
                        
                    colores = ['red' if val == 0 else 'blue' for val in y.flatten()]
                    ax.scatter(X[:, 0], X[:, 1], c=colores, s=180, edgecolors='#2ECC71', linewidths=3.5, zorder=5)
                    
                    ax.set_xlim(-0.5, 1.5)
                    ax.set_ylim(-0.5, 1.5)
                    plt.draw()
                    
                    # --- LA EXPLICACIÓN PARA LA CLASE ---
                    print(f"\n✅ ¡Récord destruido! La IA aprendió el XOR en solo {epoch} épocas.")
                    print("=" * 70)
                    print("¿POR QUÉ LA GRÁFICA DEMUESTRA UN ÉXITO TOTAL?")
                    print("-" * 70)
                    print("1. El fondo de la gráfica representa cómo las Capas Ocultas 'doblaron' el espacio.")
                    print("2. PUNTOS CERO (Los iguales): La IA agrupó correctamente en las esquinas ROJAS a:")
                    print("   -> Coordenada (0,0) - Esquina inferior izquierda.")
                    print("   -> Coordenada (1,1) - Esquina superior derecha.")
                    print("3. PUNTOS UNO (Los diferentes): La IA construyó un 'pasillo' AZUL para aislar a:")
                    print("   -> Coordenada (0,1) - Esquina superior izquierda.")
                    print("   -> Coordenada (1,0) - Esquina inferior derecha.")
                    print("\nCONCLUSIÓN: Se venció la limitante de la 'línea recta' del Perceptrón Simple.")
                    print("=" * 70)
                    break 
                else:
                    ax.set_title(f"Buscando solución... Época {epoch}", fontsize=12)
                    colores = ['red' if val == 0 else 'blue' for val in y.flatten()]
                    ax.scatter(X[:, 0], X[:, 1], c=colores, s=150, edgecolors='black', zorder=5)
                
                ax.set_xlim(-0.5, 1.5)
                ax.set_ylim(-0.5, 1.5)
                plt.draw()
                # Pausa aumentada a 0.1 segundos para que se alcance a apreciar el movimiento
                plt.pause(0.1)

if __name__ == "__main__":
    X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_xor = np.array([[0], [1], [1], [0]]) 

    red = RedNeuronalVisualXOR(learning_rate=1.2, epochs=10000)
    red.train_and_animate(X_xor, y_xor)
    plt.show()