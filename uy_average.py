#Importamos librerías y dependencias
import pandas as pd
import matplotlib.pyplot as plt

#Cargamos el archivo CSV generado con las emociones y los timestamps
df = pd.read_csv("uy_timestamps.csv")

#Contamos la frecuencia de cada emoción
emotion_counts = df['emotion'].value_counts()

#Creamos el piechart con una paleta de colores interesante
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFFF99', '#FF99FF']
plt.pie(emotion_counts, labels=emotion_counts.index, colors=colors, autopct='%1.1f%%')
plt.axis('equal')  # Para que el gráfico sea un círculo en lugar de una elipse
plt.title('Frecuencia de Emociones')
plt.show()