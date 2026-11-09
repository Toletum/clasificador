# clasificador

Experimentos de clasificación de imágenes con una CNN en Keras/TensorFlow: entrenamiento (dos variantes) y predicción sobre una imagen.

## Visibilidad y rama
- Público (github.com/Toletum/clasificador). Rama por defecto: `master`.
- Descripción en GitHub: "clasificador keras".

## Stack
- Python 3, TensorFlow/Keras.
- Dependencias (`requirements.txt`): `keras`, `tensorflow`, `numpy`, `Pillow`, `scipy`.
- `prediccion.py` importa `keras_preprocessing`, que no está en `requirements.txt` (no verificado si lo trae TensorFlow).

## Estructura
- `gen2.py`: entrena con `image_dataset_from_directory` (validación 20%), guarda `class_names.obj` y `model_saved2.h5`. Usa rutas absolutas (`/home/toletum/clasificador/t_data/train`).
- `generator.py`: entrena una CNN binaria con `ImageDataGenerator` y `fit_generator`, rutas relativas `t_data/train` y `t_data/test`. Guarda `model_saved2.h5`.
- `pre2.py`: carga `class_names.obj` y `model_saved2.h5`, y clasifica una imagen (ruta absoluta de ejemplo).
- `prediccion.py`: predicción binaria sobre `t_data/test/location/001.jpg`. Imprime "0 - Cars, 1 - Planes".
- `README.md`: instrucciones del entorno virtual.
- `.gitignore`: plantilla estándar de Python.

## Comandos
```bash
python3 -m pip install --upgrade pip
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
- Entrenar: `python gen2.py` o `python generator.py` (no documentado en el README; inferido del código).
- Predecir: `python pre2.py` o `python prediccion.py`.
- Dataset `t_data/` (train/test por clase): no está en el repo.
- Tests, lint y build: no hay. No verificado en este entorno.

## Configuración
- Sin variables de entorno. Rutas fijas en el código.
- Artefactos generados que la predicción necesita: `model_saved2.h5` y `class_names.obj` en el CWD.

## Convenciones y reglas para agentes
- Tamaño de entrada: `gen2.py` y `pre2.py` usan `img_height=231, img_width=113`; `generator.py` y `prediccion.py` usan `(113, 231)`. Un modelo entrenado con uno no funciona con el otro. Unificar antes de comparar resultados.
- Ambos entrenamientos guardan en `model_saved2.h5`: el segundo sobrescribe al primero.
- No versionar datasets ni modelos (`.h5`, `.obj`) salvo petición explícita.

## Estado
- Sin tests.
- Código experimental, basado en tutoriales (URLs en los comentarios).
- `fit_generator` está obsoleto en TensorFlow 2; `model.fit` lo sustituye.
- Rutas absolutas de otro usuario en `gen2.py` y `pre2.py`.
- `prediccion.py` y `generator.py` clasifican solo dos clases (Cars/Planes), a diferencia de `gen2.py`, que genera N clases.
- `pickle.dump(..., open(...))` sin cerrar el fichero.
