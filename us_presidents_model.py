import cv2
import pandas as pd
from datetime import datetime
from fastai.vision.all import *

def main():
    # Carga el modelo entrenado
    learn = load_learner('trained_model.pkl')

    # Importamos el archivo de vídeo y lo guardamos dentro de una variable
    cap = cv2.VideoCapture('VideoAnalytics/PresidentsSpeeches.mp4')

    # Creamos un dataframe en el que vamos a guardar la categoría detectada de cada frame y el time en del mismo
    df = pd.DataFrame(columns=["time", "category"])

    # Este es un bucle que se mantendrá activo hasta que no encuentre más frames por analizar
    while True:
        # Aquí "cap.read" nos devuelve dos valores, "img" es el frame y "ret" es un booleano que indica true si quedan
        # frames por analizar y false en caso de que ya no quede ninguno.
        ret, img = cap.read()
        if not ret:
            break

        # Convertimos el frame a imagen PIL para usar con el modelo de fastai
        img_pil = PILImage.create(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Esta línea nos devuelve la categoría detectada en el frame actual
        pred, pred_idx, probs = learn.predict(img_pil)
        
        # Aquí agregamos la categoría detectada y la hora actual al dataframe
        now = datetime.now()
        df.loc[len(df)] = [now, str(pred)]
        
        # Agregamos la etiqueta de categoría al frame actual
        cv2.putText(img, str(pred), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        # Esto nos permite mostrar en una ventana el frame que se está analizando en el momento
        cv2.imshow('img', img)
        # Aquí le indicamos que debe esperar un milisegundo entre frame y frame y que si se aprieta la tecla "q" se rompe el bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Esto indica que una vez que el bucle finalice se debe de terminar el proceso de captura
    cap.release()
    # Aquí le decimos que cierre la ventana en la que se está mostrando el vídeo
    cv2.destroyAllWindows()

    # Exportamos el dataframe resultante con todas las categorías detectadas y su time correspondiente
    df.to_csv("us_timestamps.csv", index=False)

if __name__ == '__main__':
    main()