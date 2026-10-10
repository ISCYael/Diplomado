import pandas as pd
import matplotlib.pyplot as plt

# 1. Crear una pequeña tabla de datos con Pandas
datos = {'Meses': ['Enero', 'Febrero', 'Marzo'], 'IA_Proyectos': [2, 5, 4]}
df = pd.DataFrame(datos)
print("Datos cargados correctamente en Pandas:")
print(df)

# 2. Hacer una gráfica con Matplotlib
df.plot(x='Meses', y='IA_Proyectos', kind='bar', color='blue')
plt.title('Prueba de Matplotlib')

print("\nAbriendo gráfica... Cierra la ventana de la gráfica para terminar el programa.")
plt.savefig('mi_grafica.png')