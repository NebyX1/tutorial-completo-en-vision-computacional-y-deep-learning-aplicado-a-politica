# Tutorial Completo en Visión Computacional Aplicado a Política

Este repositorio contiene un tutorial completo de visión computacional y aprendizaje profundo aplicado a contextos políticos. El tutorial se desarrolla a través de varios scripts de Python, cada uno diseñado para cumplir una función específica en el flujo de trabajo de análisis de imágenes y vídeos políticos. El tutorial está diseñado para ayudar a los aprendices a entender cómo se pueden aplicar las técnicas de visión computacional y aprendizaje profundo para analizar y entender el contenido visual en el contexto de la política.
Sin embargo, naturalmente su contenido y aplicaciones, puedes ser aprovechados por un público mucho más amplio, pensemos en que este tipo de tecnologías, pueden ser fácilmente adapatadas con leves modificaciones a cualquier área del conocimiento.
Algunos ejemplos claros, pueden ser, adapaciones para aplicaciones de marketing, psicología y psiquiatría, biotecnología y ciencias médicas/ciencias de la vida. No se requeriría muchas modificaciones si se quisiera usar el script "us_train_model.py" para detección de por ejemplo, células tumorales en placas.

### Descripción de los Scripts
us_train_model.py: Este script entrena un modelo de clasificación de imágenes usando la biblioteca Fastai. El modelo se entrena para distinguir entre imágenes de los expresidentes estadounidenses Barack Obama y Donald Trump.

us_test.py: Este script prueba el modelo entrenado en el script anterior en un nuevo conjunto de imágenes para evaluar su rendimiento.

us_presidents_model.py: Este script utiliza el modelo entrenado para analizar un vídeo y determinar qué expresidente aparece en cada fotograma. El resultado es un archivo CSV que contiene la clasificación de cada fotograma y su timestamp.

uy_emotion_analyzer.py: Este script utiliza la biblioteca DeepFace para analizar las emociones expresadas por el presidente de Uruguay en un vídeo.

us_average.py y uy_average.py: estos scripts crean un gráfico de "torta" con las distintas categorías que aparecen en los archivos csv que están destinados a leer.

### Casos de Uso
El tutorial ha sido diseñado para ilustrar dos problemas de investigación hipotéticos.

En el primer caso, una firma de consultores en análisis político internacional, busca hacer un estudio para un Think Tank financiado por el partido X llamado "Conservatives Freedom Group" que quiere un modelo de análisis de visión computacional, que les permita analizar de forma masiva, cientos de horas sacadas de canales de TV de todo EEUU para saber si le dan más minutos en pantalla a Trump o a Obama.

En el segundo caso, la misma firma, trabajará, para otra firma Europea que los subcontrata, y la misma está relacionada a una fundación alemana llamada "Die Rheinkonservativen", y que busca analizar en Uruguay, las emociones de los políticos de primera línea de ese país en sus discursos recientes, para lograr obtener un modelado mental de los mismos, para faiclitar la creación de una estrategia de negociación efectiva, ante intereses europeos en Uruguay que pudieran surgir.

### Metodología
El análisis de los vídeos se realiza mediante un modelo de clasificación de imágenes entrenado para reconocer a Trump y a Obama. El modelo es probado en un conjunto de imágenes que no ha visto antes, y luego se utiliza para analizar un vídeo de 40 segundos que contiene imágenes de ambos expresidentes.

Para el análisis de emociones, se utiliza la biblioteca DeepFace. Esta biblioteca aplica un modelo preentrenado para reconocer siete emociones básicas en imágenes de rostros. Se aplica a un vídeo del presidente de Uruguay, produciendo una tabla de tiempo de las emociones detectadas.

### Instalación y Uso
Los scripts requieren la instalación de varias bibliotecas de Python, incluyendo fastai, cv2, pandas, datetime, y deepface. Estas pueden ser instaladas utilizando pip:

>pip install -r requirements.txt

Una vez instaladas las dependencias, los scripts pueden ser ejecutados directamente desde la línea de comandos.

### Contribución
Las contribuciones a este proyecto son bienvenidas. Por favor, abra un problema para discutir la contribución antes de hacer un pull request.