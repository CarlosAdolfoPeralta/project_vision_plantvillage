🇬🇧 [Read in English](README.md)
# Clasificación de enfermedades en hojas: de 99 % en laboratorio a 9 % en campo

Proyecto de visión por computadora con PlantVillage: baseline con CNN propia, transfer learning con MobileNetV3-Small, interpretabilidad con Grad-CAM y una prueba de robustez con imágenes de campo (PlantDoc). Todo el entrenamiento se hizo **solo con CPU**.

**Autor:** Carlos Adolfo Peralta · [LinkedIn](https://www.linkedin.com/in/carlosperalta1993/) · [GitHub](https://github.com/CarlosAdolfoPeralta)

## Resumen

- Un modelo de transfer learning (MobileNetV3-Small) alcanza **accuracy 0,987 y F1 macro 0,982** en el test de PlantVillage (39 clases) entrenando con todo el train. Con el mismo 25 % de train que usó el baseline, obtiene 0,968 y 0,959, frente a **0,803 y 0,746** de la CNN entrenada desde cero.
- Sobre imágenes de campo (PlantDoc, 2.569 imágenes, 28 clases mapeadas) el mismo modelo cae a **accuracy top-1 de 0,088 y F1 macro de 0,095**. Más de la mitad de las imágenes se predicen como "fondo sin hojas".
- La **confianza del modelo no permite detectar esos errores**: es igual de alta en errores (0,91) que en aciertos (0,89).
- Grad-CAM muestra que, en imágenes de laboratorio, el modelo mira la hoja y no el fondo. El fallo en campo parece deberse a un **cambio de dominio** (hipótesis no probada) y no a un problema de atención al fondo.
- Conclusión: las métricas en datasets de laboratorio no garantizan el rendimiento en condiciones reales.

## Datos

| | |
|---|---|
| Dataset principal | PlantVillage (versión `DScomp380/plant_village` en Hugging Face) |
| Imágenes | 55.447, RGB, casi todas de 256x256 |
| Clases | 39 (38 de cultivo/condición + `Background_without_leaves`) |
| Composición | 39.221 enfermas, 15.083 sanas, 1.143 de fondo |
| Desbalance | 36,2x entre la clase más grande (5.507) y la más chica (152, `Potato___healthy`) |
| Split | estratificado 70/15/15, semilla 42: train 38.812, val 8.317, test 8.318 |
| Datos externos | PlantDoc (`Project-AgML/plant_doc_classification`), 2.569 imágenes recortadas de buscadores, mapeadas a las clases de PlantVillage |

**Aviso sobre fuga de datos:** esta versión del dataset no trae identificador de hoja, por lo que el split no pudo agruparse por hoja física y puede haber fotos casi idénticas entre train y test. Una estimación conservadora con hash perceptual detectó al menos 1,3 % de casi-duplicados entre test y train (19 de 1.500 imágenes de test; es un piso, no una medida del problema real). Las métricas en PlantVillage deben leerse como probablemente optimistas.

## Metodología

1. **EDA** (`01_eda`): distribución de clases, composición sana/enferma/fondo, tamaños de imagen y estimación de fuga.
2. **Baseline** (`02_baseline_cnn`): CNN de 4 bloques (103 mil parámetros) entrenada desde cero a 128x128 con el 25 % de train (9.703 imágenes) y data augmentation. Se comparó con y sin pesos de clase.
3. **Transfer learning** (`03_transfer_learning`): MobileNetV3-Small preentrenada en ImageNet y congelada. Cada imagen pasa una sola vez por la red a 224x224, se guardan los vectores de características y se entrena solo una cabeza densa (159 mil parámetros). Esto permite entrenar en segundos sin GPU. No usa data augmentation.
4. **Interpretabilidad y robustez** (`04_gradcam_robustez`): Grad-CAM calculado sobre el logit de la clase predicha, una medida aproximada de cuánta atención cae sobre la hoja, y evaluación sin reentrenar sobre PlantDoc.

La métrica principal es **F1 macro**, por el desbalance entre clases.

## Resultados

### Baseline vs. transfer learning (test de PlantVillage)

| Modelo | Train | Pesos de clase | Accuracy | F1 macro | Entrenamiento |
|---|---|---|---|---|---|
| CNN desde cero | 25 % | no | 0,803 | 0,746 | 18,3 min |
| CNN desde cero | 25 % | sí (balanceados) | 0,758 | 0,723 | 20,0 min |
| MobileNetV3-Small | 25 % | no | 0,968 | 0,959 | < 1 min |
| MobileNetV3-Small | 25 % | suaves (exp. 0,5) | 0,968 | 0,960 | < 1 min |
| MobileNetV3-Small | 100 % | no | 0,985 | 0,980 | < 1 min |
| **MobileNetV3-Small** | **100 %** | **suaves (exp. 0,5)** | **0,987** | **0,982** | **< 1 min** |

Las filas con 25 % de train usan el mismo subconjunto estratificado, por lo que son comparables entre sí. El tiempo de entrenamiento de los modelos con transfer learning corresponde solo a la cabeza; la extracción de features se hace una vez y queda en caché.

- Los datos pesan más que cualquier otro cambio: pasar de 25 % a 100 % de train sube el F1 macro unos 2 puntos.
- Los pesos de clase balanceados completos empeoraron el baseline (mejoran recall de clases chicas a costa de precision). Los pesos suaves aportan 0,1 a 0,2 puntos con transfer learning, diferencia que probablemente esté dentro del ruido entre corridas (una corrida por configuración).
- La clase de fondo no infla los resultados: el F1 macro sin ella es prácticamente igual.
- Los errores restantes son sobre todo entre enfermedades parecidas dentro de un mismo cultivo (por ejemplo, ácaros ↔ `Target_Spot` en tomate, `Cercospora` ↔ `Northern_Leaf_Blight` en maíz).

### Robustez en imágenes de campo (PlantDoc)

| Métrica | PlantVillage (test) | PlantDoc |
|---|---|---|
| Accuracy top-1 | 0,987 | **0,088** |
| Accuracy top-3 | n/d | 0,229 |
| F1 macro | 0,982 | 0,095 |

- 1.397 de 2.569 imágenes (54 %) se predicen como `Background_without_leaves`.
- Ignorar la clase fondo al predecir solo sube el top-1 a 0,123; usar un recorte cuadrado en lugar de estirar la imagen lo deja en 0,104. El problema no se explica por esos dos factores.
- **La confianza no sirve para rechazar errores:** confianza media de 0,894 en aciertos y 0,908 en errores. Aceptando solo predicciones con confianza ≥ 0,99 (53 % de las imágenes), la accuracy de lo aceptado sigue en 0,075.
- La mejor clase es `Tomato___Late_blight` (0,459); ninguna clase sana supera 0,06.

### Grad-CAM

- En el test de PlantVillage, el mapa cae sobre la hoja (a veces sobre lesiones, a veces sobre tejido sano cercano) y no sobre el fondo. Con una máscara aproximada de hoja por saturación de color, el **ratio de atención mediano es 1,40** (247 imágenes): la activación sobre la hoja es 1,4 veces la esperable por el área que ocupa.
- En PlantDoc, cuando el modelo acierta el mapa cae sobre las lesiones; cuando se equivoca, muestra una o dos manchas dispersas, sin relación con las lesiones.
- Grad-CAM se calculó sobre el logit y no sobre la probabilidad softmax: con predicciones muy seguras, el gradiente de la softmax se aplana y produce mapas vacíos.

## Limitaciones

- Split sin agrupar por hoja: las métricas en PlantVillage probablemente sean optimistas.
- Una sola corrida por configuración, sin repeticiones con distintas semillas: diferencias de pocas décimas no son concluyentes.
- Entrenamiento solo en CPU: baseline pequeño, con subconjunto y resolución reducida; el modelo de transfer learning no usa augmentation ni fine-tuning.
- El mapeo de clases PlantDoc → PlantVillage es manual y puede tener errores de criterio (por ejemplo, qué enfermedad concreta corresponde a una etiqueta genérica como "leaf spot"). PlantDoc proviene de buscadores de imágenes y puede tener ruido de etiquetas. Además, algunas clases tienen muy pocas imágenes (por ejemplo, 2 para ácaros en tomate), por lo que su accuracy por clase no es informativa.
- La máscara de hoja para medir la atención es una heurística por saturación de color, y los mapas de Grad-CAM tienen baja resolución (7x7). La revisión de las grillas fue visual y sobre una muestra pequeña; en la muestra solo hubo 4 errores en PlantVillage, por lo que no se pudieron comparar aciertos y errores.
- No se probó por qué falla el modelo en PlantDoc: la explicación más probable es un cambio de dominio general (iluminación, escala, fondos, calidad), pero es una hipótesis.

## Trabajo futuro

- Entrenar con data augmentation fuerte (color, fondo, recortes) y hacer fine-tuning de las últimas capas.
- Incorporar parte de las imágenes de campo al entrenamiento y evaluar en el resto (cuidando que recortes de una misma imagen original no queden en ambos lados).
- Agregar una clase o un método de detección de entradas fuera de distribución, ya que la confianza softmax no es útil para eso.
- Repetir los experimentos con varias semillas para estimar la variabilidad.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   └── 04_gradcam_robustez.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py          # carga de datos, splits y caché de arrays
│   └── features.py      # extracción de features con redes preentrenadas
├── reports/             # resultados de cada experimento (.json)
├── models/              # modelos entrenados (no se versionan)
└── data/                # datos y caché (no se versionan)
```

## Cómo reproducirlo

1. Crear un entorno e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. (Opcional) Descargar el dataset por consola, con reanudación si se corta:
   ```bash
   hf download DScomp380/plant_village --repo-type dataset --local-dir data/plant_village
   ```
   Si la carpeta `data/plant_village` no existe, `src/data.py` lo descarga automáticamente desde Hugging Face.
3. Ejecutar los notebooks en orden desde la carpeta `notebooks/`. La primera ejecución del 02 y del 03 tarda más porque genera cachés en `data/cache/`; a modo orientativo, cada corrida del baseline tarda entre 18 y 20 minutos en CPU. El notebook 04 usa el modelo que guarda el 03 con `USE_SUBSET=False` y `CW_POWER=0.5`, por lo que esa configuración debe haberse ejecutado antes.
4. Para la sección de robustez, el notebook 04 usa PlantDoc: lee la carpeta `data/plantdoc` si existe (se puede descargar con `hf download Project-AgML/plant_doc_classification --repo-type dataset --local-dir data/plantdoc`) o, si no, lo descarga desde Hugging Face. También acepta fotos propias en `data/external/<NombreDeClase>/`.

Desarrollado con Python 3.13, TensorFlow 2.21 y `datasets` 5.1.0. Las semillas están fijadas (42), pero los resultados pueden variar levemente entre ejecuciones y equipos.

## Referencias y licencias

- Mohanty, Hughes y Salathé (2016). *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science.
- Singh et al. (2019). *PlantDoc: A Dataset for Visual Plant Disease Detection*. arXiv:1911.10317. Licencia CC BY 4.0.
- La versión de PlantVillage usada figura en Hugging Face con licencia CC0 1.0, según su ficha.
- La versión de PlantDoc usada (Hugging Face) tiene 2.569 imágenes y 28 clases; el artículo original reporta 2.598 imágenes y 27 clases.
- Los datos no se incluyen en el repositorio: se descargan desde Hugging Face.
