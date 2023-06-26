# Importamos todas las librerías y dependencias
import cv2
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# Importamos el archivo de vídeo y lo guardamos dentro de una variable
cap = cv2.VideoCapture('VideoAnalytics/LacallePou.mp4')

# Creamos un dataframe en el que vamos a guardar la emotividad de cada frame y el time en del mismo
df = pd.DataFrame(columns=["time", "emotion"])

# Este es un bucle que se mantendrá activo hasta que no encuentre más frames por analizar, cuando ya no encuentre más,
# se detendrá
while True:
    # Aquí "cap.read" nos devuelve dos valores, "img" es el frame y "ret" es un booleano que indica true si quedan
    # frames por analizar y false en caso de que ya no quede ninguno.
    ret, img = cap.read()
    if not ret:
        break
    # Esta línea nos devuelve la emoción detectada en el frame actual
    result = DeepFace.analyze(img, actions=['emotion'])

    # Verificamos si result es una lista o un diccionario
    if type(result) is list:
        emotion = result[0]['dominant_emotion']
    else:
        emotion = result['dominant_emotion']

    # Aquí agregamos la emoción detectada y la hora actual al dataframe
    now = datetime.now()
    df.loc[len(df)] = [now, emotion]
    
    # Agregamos la etiqueta de emoción al frame actual
    cv2.putText(img, emotion, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    # Esto nos permite mostrar en una ventana el frame que se está analizando en el momento
    cv2.imshow('img', img)
    # Aquí le indicamos que debe esperar un milisegundo entre frame y frame y que si se aprieta la tecla "q" se rompe el bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Esto indica que una vez que el bucle finalice se debe de terminar el proceso de captura
cap.release()
# Aquí le decimos que cierre la ventana en la que se está mostrando el vídeo
cv2.destroyAllWindows()

# Exportamos el dataframe resultante con todas las emociones detectadas y su time correspondiente
df.to_csv("uy_timestamps.csv", index=False)