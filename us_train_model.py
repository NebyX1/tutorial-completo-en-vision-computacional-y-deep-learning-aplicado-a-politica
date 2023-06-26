# Importamos todas las dependencias del módulo de visión computacional de fastai
from fastai.vision.all import *

# Declaramos la función de ejecución que servirá para correr el script 
def main():
    # Creamos un objeto/variable de ImageDataLoaders, que se encarga de cargar las imágenes para el entrenamiento y validación(prueba).
    # '/images' es la ubicación del directorio que contiene las imágenes.
    # train="." indica que todas las imágenes del directorio serán utilizadas para entrenamiento.
    # valid_pct=0.2 significa que el 20% del total de las imágenes será utilizado para la validación.
    # seed=42 se utiliza para garantizar la reproducibilidad del conjunto de validación, es decir en palabras más claras, quiere decir que los datos
    # que si bien serán aleatorios, empezarán a partir del número 42(puede ser cualquier otro) y eso nos garantiza que cuando ejecutemos otra vez
    # el escript, podremos reproducir el mismo resultado de aleatoriedad.
    # item_tfms=Resize(460) redimensiona todas las imágenes a 460x460 píxeles antes de la creación del batch.
    # batch_tfms=aug_transforms(size=224) aplica transformaciones de aumentación de datos a los batches y los redimensiona a 224x224, es decir
    # crea a partir del conjunto de datos que tenemos, nuevas variantes de los mismos a partir de pequeños cambios en las imágenes, cambios de color,
    # rotación, cambio de ejes, etc.
    data = ImageDataLoaders.from_folder(
        'images', train=".", valid_pct=0.2, seed=42,
        item_tfms=Resize(460), batch_tfms=aug_transforms(size=224)
    )

    # Creamos un modelo de aprendizaje profundo(Deep Learning) utilizando la arquitectura ResNet34 pre-entrenada.
    # 'data' es el objeto ImageDataLoaders que hemos creado.
    # 'resnet34' es el modelo pre-entrenado que utilizaremos.
    # 'metrics=accuracy' indica que queremos rastrear la precisión del modelo durante el entrenamiento.
    learn = cnn_learner(data, resnet34, metrics=accuracy)
    
    # Entrenamos el modelo durante 4 ciclos (epochs). En cada ciclo, el modelo ve todas las imágenes de entrenamiento una vez.
    learn.fit_one_cycle(4)

    # Exportamos nuestro modelo entrenado en un archivo .pkl para su posterior uso.
    # 'trained_model.pkl' es el nombre del archivo donde se guardará el modelo.
    learn.export('../trained_model.pkl')

# La siguiente condición verifica si este script se está ejecutando directamente.
# Si es así, se llama a la función main(), que comienza a ejecutar el script.
# Si el script se está importando desde otro script, la condición es False y main() no se llama automáticamente.
if __name__ == '__main__':
    main()