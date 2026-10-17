import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Motor gráfico corregido para WSL
import matplotlib.pyplot as plt

class PerceptronVisual:
    def __init__(self, learning_rate=0.1, epochs=15):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(2)
        self.bias = 0.0

    def activation(self, x):
        # Función escalón: si el número es positivo devuelve 1, si es negativo devuelve 0
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    def train_and_plot(self, X, y, title):
        # Pesos iniciales desacomodados a propósito para ver cómo la IA los arregla
        self.weights = np.array([0.5, -0.5]) 
        self.bias = -0.2

        plt.figure(figsize=(6, 6))

        for epoch in range(self.epochs):
            error_flag = False
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                if error != 0:
                    # Regla de aprendizaje del perceptrón (ajuste de la línea)
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    error_flag = True

            # --- DIBUJAR LA GRÁFICA EN CADA ÉPOCA ---
            plt.clf()
            plt.title(f"{title} - Época {epoch + 1}") # Título corregido
            plt.xlim(-0.5, 1.5)
            plt.ylim(-0.5, 1.5)
            
            # Dibujar los puntos de datos (rojo para 0, azul para 1)
            for i, xi in enumerate(X):
                color = 'blue' if y[i] == 1 else 'red'
                plt.scatter(xi[0], xi[1], color=color, s=100, zorder=5)

            # Dibujar la línea verde (frontera de decisión)
            if self.weights[1] != 0:
                x1_vals = np.array([-0.5, 1.5])
                x2_vals = -(self.weights[0] * x1_vals + self.bias) / self.weights[1]
                plt.plot(x1_vals, x2_vals, color='green', linestyle='--')

            plt.draw()
            plt.pause(0.8) # Pausa de casi un segundo para ver la animación

            # Si ya no hubo errores, la IA aprendió a separar los colores
            if not error_flag:
                print(f"\n¡Éxito! Entrenamiento completado para {title} en la época {epoch + 1}.")
                break

        plt.show()

if __name__ == "__main__":
    # Inicializamos nuestra IA
    perceptron = PerceptronVisual(learning_rate=0.1, epochs=15)

    # --- LA CORRECCIÓN DE LA COMPUERTA AND ---
    X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_and = np.array([0, 0, 0, 1])
    
    print("Ejecutando visualización para COMPUERTA AND...")
    perceptron.train_and_plot(X_and, y_and, "Compuerta AND")