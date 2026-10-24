from skimage import data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Lista para guardar los puntos que vayamos tocando
coordenadas = []

def al_hacer_click(evento):
    # Verificar que el click fue dentro de la imagen y no afuera
    if evento.xdata is not None and evento.ydata is not None:
        x = int(evento.xdata)
        y = int(evento.ydata)
        coordenadas.append((x, y))
        print(f"Punto registrado en el píxel: X={x}, Y={y}")
        
        # Dibujar un punto rojo ('ro') en la gráfica
        plt.plot(x, y, 'ro', markersize=6)
        plt.draw() # Actualizar la ventana en vivo

def seleccionar_puntos():
    print("Abriendo imagen... Haz click en cualquier parte de la foto.")
    imagen = data.chelsea() # Usamos al mismo gatito
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(imagen)
    ax.set_title("Selección Interactiva\nHaz click para poner puntos. Cierra la ventana al terminar.")
    ax.axis('off')
    
    # Aquí está la magia: conectamos tu ratón con nuestra función
    fig.canvas.mpl_connect('button_press_event', al_hacer_click)
    
    plt.show()
    
    # Al cerrar la ventana, imprimimos todos los puntos recolectados
    print("\nResumen de coordenadas seleccionadas:")
    print(coordenadas)

if __name__ == "__main__":
    seleccionar_puntos()