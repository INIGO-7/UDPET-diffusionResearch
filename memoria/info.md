## Conociendo el dominio

_Qué es un escaneo PET?_

Positron Emision Technology, o PET, es un test de imágenes que utiliza material radiactivo para diagnosticar, monitorizar y tratar una variedad de condiciones y enfermedades. Los doctores pueden utilizar este tipo de escaneo para encontrar tumores, diagnosticar enfermedades de corazón, transtornos del cerebro, y otros. Este escanéo es preferido frente a otros (p. ej. Rayos X) ya que utiliza un trazado radioactivo que puede mostrar cómo funcionan los órganos en tiempo real.

_Qué es un problema inverso?_

El problema inverso consiste en reconstruir la imagen idealizada a partir de la imagen medida, en nuestro caso reconstruyendo el escaneo PET producido con la máxima dosis de radiación y tiempo en la máquina, a partir de una imagen realizada con mediciones parciales.

## Investigando el estado del arte

Aquí se listan métodos y propuestas actualizadas, relacionadas con el problema de la reconstrucción de imágenes médicas.

- [Dic 2025] [link](https://papers.miccai.org/miccai-2025/paper/1155_paper.pdf) Se propone un Task-Adaptive Transformer (TAT) para la restauración de imágenes médicas, reclamando que alcanza el **estado del arte** en las tareas de síntesis de PET, reducción de ruido en TAC, y "super-resolution" de RM. Código disponible en [github](https://github.com/Yaziwel/TAT)
- [Jun 2025] [link](https://www.sciencedirect.com/science/article/pii/S1361841524002597) Propone un modelo basado en el paradigma `MAMBA` para la reconstrucción de imágenes médicas y modelización de la incertidumbre es propuesto como el **estado del arte** para la tarea del **Low-dose PET**, además de la resonancia magnética (RM) rápida y la tomografía axial computarizada (TAC) "sparse-view"
- [Jun 2024] [link](https://jnm.snmjournals.org/content/65/supplement_2/241109.abstract) In this work, we proposed a novel denoising diffusion probabilistic model (DDPM) based low dose PET image reconstruction method, named DDPEM. The proposed DDPEM integrates the iterative process of Expectation Maximization (EM) with the reverse sampling process of DDPM.
- [Jun 2024] [link](https://academic.oup.com/bjrai/article/1/1/ubae013/7745314) Establece que los algoritmos de reconstrucción de imágenes que incorporan modelos de difusión no supervisados son el estado del arte para tareas como RM ultra rápido, TAC "super-sparse-view" y PET de dosis baja. Propone una introducción accesible a la reconstrucción de imágenes y los modelos de difusión, además de la metodología para reconstruir imágenes con modelos de difusión, retos específicos de la modalidad y los campos de investigación claves.
- [Ene 2024] [link](https://arxiv.org/abs/2308.14190) Score-Based Generative Models for PET Image Reconstruction.
- [Oct 2023] [link](https://link.springer.com/article/10.1007/s00259-023-06417-8) Propone un Denoising Diffusion Probabilistic Model (DDPM) como modelo de aprendizaje de una distribución, y evalúa distintos métodos basados en DDPM para la reducción de ruido en escaneos PET.

## Fundamentos de modelos de difusión

En el contexto de generación de datos, empezamos con una distribución de probabilidad `p(x)`, que representa nuestros datos de entrenamiento. En el caso de las imágenes, podemos pensar en esta distribución como una representación de todas las imágenes naturales; por simplicidad vamos a pensar en un subconjunto de imágenes conteniendo únicamente caras de personas. 
Esta distribución es tan compleja que no podemos encontrar una sola expresión que la describa por completo. Igualmente, sin una fórmula en específico, buscamos generar imágenes nuevas, que es equivalente a  muestrear puntos de esta distribución subyacente. El reto es encontrar una forma de crear nuevos ejemplos, sin tener una manera comprensiva de encontrar la distribución. 
Los modelos de difusión resolven este problema con un enfoque completamente diferente.

Se encuentra un proceso de difusión que transforma cualquier distribución en una distribución normal $N(0,1)$. La distribución condicional `q` es definida para ir de un paso al siguiente, se define la distribución condicional `p` para ir hacia atrás en la dirección opuesta. En el proceso hacia delante todos los parámetros están fijos, pero para el proceso hacia detrás el objetivo es de hecho encontrar los mejores parámetros $theta$ que ayuden a quitar el ruido de manera efectiva. Entrenaremos una red neuronal precisamente con el fin de encontrar estos parámetros.

Cómo entrenamos esta red neuronal? Para entrenarla minimizaremos la log-verosimilitud de la probabilidad de producir una muestra con nuestro modelo. De manera más sencilla, significa encontrar el conjunto de parámetros $theta$ que maximizan la probabilidad de generar muestras reales $x_0$ de la distribución de nuestros datos usando la red neuronal.

La probabilidad conjunta describiendo la distribución de todas las variables desde $x_1$ hasta $x_t$ dado $x_0$ representa el proceso hacia delante por completo:
$$q(x_1, \ldots, x_T \mid x_0) = q(x_1 \mid x_0)\, q(x_2 \mid x_1, x_0)\, \ldots\, q(x_T \mid x_{T-1}, \ldots, x_0)$$

El proceso hacia delante (forward process) del modelo de difusión es una cadena de markov, lo cuál significa que cada paso del proceso de adición de ruido depende exclusivamente del paso anterior, lo cuál simplifica las probabilidades condicionales.

$$q(x_1, \ldots, x_T \mid x_0) = q(x_1 \mid x_0)\, q(x_2 \mid x_1)\, \ldots\, q(x_T \mid x_{T-1})$$

Podemos establecer los procesos hacia delante y hacia atrás como:
- Proceso hacia delante completo: $q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})$
- Proceso hacia detrás completo: $p_\theta(x_{0:T}) = p_\theta(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)$


### Referencias

- [Jun 2020] [link](https://arxiv.org/abs/2006.11239) Paper que popularizó los modelos de difusión en 2020, llamado "Denoising Diffusion Probabilistic Models" hecho por Ho. et al.
- [Mar 2015] [link](https://arxiv.org/abs/1503.03585) La referencia más temprana conocida a los modelos de difusión, paper titulado "Deep Unsupervised Learning using Nonequilibrium Thermodynamics", Sohl-Dickstein et al.

## Generando mariposas - modelos de difusión en acción

Como primer experimento práctico, se entrena un modelo DDPM (*Denoising Diffusion Probabilistic Model*) sobre un conjunto de imágenes de mariposas, con el objetivo de generar muestras sintéticas nuevas. Este ejercicio sirve como banco de pruebas para comprender el proceso de entrenamiento e inferencia antes de abordar el problema de reconstrucción de imágenes PET.

### Datos y preprocesamiento

Se utiliza el subconjunto `huggan/smithsonian_butterflies_subset` del Instituto Smithsoniano. Cada imagen se redimensiona a $128 \times 128$ píxeles, se aplica volteo horizontal aleatorio como aumento de datos, y se normaliza al rango $[-1, 1]$ mediante la transformación:

$$x' = \frac{x - 0.5}{0.5}$$

El conjunto se carga en lotes de 16 imágenes con orden aleatorio en cada época.

### Proceso de difusión hacia delante

El proceso de difusión hacia delante $q$ corrompe progresivamente una imagen limpia $\mathbf{x}_0$ añadiendo ruido gaussiano a lo largo de $T = 1000$ pasos discretos. En cada paso $t$, la distribución de la imagen ruidosa condicionada a la imagen anterior es:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I}\right)$$

donde $\{\beta_t\}_{t=1}^{T}$ es un calendario de varianza predefinido. Gracias a la propiedad de reproducibilidad de la distribución gaussiana, este proceso admite una forma cerrada que permite muestrear $\mathbf{x}_t$ directamente desde $\mathbf{x}_0$ en un único paso:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1 - \bar{\alpha}_t)\mathbf{I}\right)$$

donde $\alpha_t = 1 - \beta_t$ y $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. En la práctica, esto equivale al muestreo por reparametrización:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### Arquitectura del modelo

La red neuronal $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)$, encargada de predecir el ruido, es una UNet2D con bloques ResNet y atención espacial. La arquitectura presenta seis niveles de resolución con canales $(128, 128, 256, 256, 512, 512)$ y dos capas ResNet por bloque. La atención multi-cabeza se incorpora únicamente en el quinto nivel, tanto en el camino descendente (`AttnDownBlock2D`) como en el ascendente (`AttnUpBlock2D`), donde la resolución espacial es suficientemente reducida para que el coste cuadrático de la atención sea asumible.

El paso temporal $t$ se inyecta en cada bloque ResNet mediante *embeddings* sinusoidales, análogos a los empleados en el Transformer original, de modo que el modelo aprende una política de denoising condicional en el nivel de ruido.

### Objetivo de entrenamiento

En lugar de predecir directamente $\mathbf{x}_0$, el modelo sigue la formulación simplificada de Ho et al. (2020) y aprende a predecir el ruido $\boldsymbol{\varepsilon}$ a partir de la imagen ruidosa. La función de pérdida es el error cuadrático medio entre el ruido real y el predicho:

$$\mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}[1,T],\, \mathbf{x}_0,\, \boldsymbol{\varepsilon}}\!\left[\left\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta\!\left(\mathbf{x}_t, t\right)\right\|^2\right]$$

En cada iteración se muestrea $t$ uniformemente en $\{1, \ldots, 1000\}$, se construye $\mathbf{x}_t$ mediante la ecuación de muestreo directo, y el gradiente se propaga únicamente a través de la predicción del modelo.

### Configuración del entrenamiento

El modelo se optimiza con AdamW con tasa de aprendizaje inicial $\eta = 10^{-4}$ y un planificador cosenoidal con 500 pasos de calentamiento (*warmup*). El entrenamiento se lleva a cabo durante 50 épocas con precisión mixta `fp16` para reducir el consumo de memoria, y se recortan los gradientes a norma máxima 1.0 para garantizar la estabilidad numérica.

### Proceso de generación (inferencia)

Para generar una nueva imagen, se parte de ruido puro $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ y se desnuida iterativamente aplicando el paso inverso aprendido durante $T$ iteraciones. En cada paso $t$, el scheduler DDPM recupera $\mathbf{x}_{t-1}$ como:

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t\,\mathbf{z}, \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Tras los $T = 1000$ pasos de denoising, $\mathbf{x}_0$ constituye la imagen sintética final. La implementación permite fijar una semilla aleatoria para garantizar reproducibilidad o variarla libremente para explorar el espacio generativo del modelo entrenado.

### Aprendizajes adquiridos y conclusiones

Los modelos de difusión son buenos generalizando ya que es más complicado que puedan memorizar los ejemplos de entrenamiento, dado que van a ver las mismas imágenes pero con una cantidad de ruido completamente diferente entre un epoch y otro.
Se ven forzados a aprender la distribución subyacente de las imágenes que analizan, en este caso, aprendiendo como es una mariposa.
Aprender a predecir la cantidad de ruido que hay en una imagen es matemáticamente equivalente a predecir la imagen de una mariposa a partir de una imagen ruidosa.

Se ha observado que la pérdida durante el entrenamiento decrece muy rápidamente en los primeros epoch, y después se estabiliza a lo largo de todo el proceso de entrenamiento. Esto no significa que el modelo deje de aprender una vez se estabiliza la pérdida, sino que aprende a predecir la cantidad de ruido que hay en las imágenes más rápidamente; pero los parámetros se siguen actualizando y aprendiendo con más certeza la distribución subyacente de imágenes de mariposas. 

(TODO - poner imágenes de mariposas comparando los resultados en los primeros epoch vs en los últimos)

A pesar de que el loss es el mismo que en epoch más tempranos, observamos que las imágenes empiezan a ser buenas en el epoch 90, parando en el epoch 100, en el cuál podemos obtener resultados como los siguientes:

(TODO - poner los mejores resultados de inferencia de mariposas, escogiéndolos a mano)


## Desarrollo - modelos de difusión para reconstrucción de escaneos PET

### Ruido

Una idea clave para desarrollar esto es utilizar los escaneos de máxima calidad, y entrenar un modelo de difusión de tal forma que las funciones de adición de ruido imiten el tipo de ruido presente en los escaneos de dosis reducida. Típicamente, los datos PET están muestreados con ruido de Poisson de alta varianza.

A common misconception is that PET’s Poisson noise model is incompatible with the artificial Gaussian noise used in the diffusion process. This is not an issue. In general, the diffusion process maps between a useful distribution of medical images (with some inherent noise) and a known distribution. This known distribution is chosen to be Gaussian for its nice mathematical properties, though other options are possible.

### Planteamiento del MVP

A partir de los fundamentos repasados anteriormente y de las propuestas recogidas en el estado del arte, se fija un primer producto mínimo viable (MVP) para la tarea de reconstrucción de PET de dosis baja con modelos de difusión. El diseño persigue dos objetivos: ser **defendible científicamente** —cada decisión está justificada por la literatura recogida en la sección 2 o por los fundamentos teóricos repasados— y **ejecutable sobre hardware local** —un MacBook Pro M4 Pro con 24 GB de memoria unificada y aceleración MPS, sin acceso a CUDA—.

La reconstrucción de PET de dosis baja se formula como un problema inverso: dada una observación $\mathbf{y}$ (escaneo de dosis reducida), se busca recuperar $\mathbf{x}$ (escaneo equivalente a dosis completa). En la literatura coexisten dos familias de soluciones basadas en difusión, ambas representadas en la sección 2:

1. **Difusión condicional supervisada.** Se entrena directamente la distribución posterior $p(\mathbf{x} \mid \mathbf{y})$, incorporando $\mathbf{y}$ como entrada de condicionamiento de la red de denoising. Es la formulación más simple y aprovecha íntegramente los 371 pares registrados disponibles.

2. **Prior no condicional con consistencia de medición.** Se entrena un modelo $p(\mathbf{x})$ sobre los escaneos de dosis completa exclusivamente; en inferencia, la trayectoria inversa del muestreo se guía hacia consistencia con la observación $\mathbf{y}$ mediante un término de verosimilitud (Chung et al., 2023). Esta es la línea identificada como estado del arte para el PET de dosis baja por la revisión de [Jun 2024] del *British Journal of Radiology, Artificial Intelligence*.

Se decide construir **ambos enfoques en paralelo** y compararlos sobre el mismo conjunto de test. La comparación entre el supervisado y el no condicional con guiado constituye uno de los resultados centrales que la memoria persigue reportar.

### Datos: caracterización y preprocesamiento

El conjunto disponible consta de **371 pares** de volúmenes en formato NIfTI: para cada paciente se dispone de la versión adquirida con dosis completa y la versión con dosis reducida a un veinteavo. Cada volumen tiene forma $(440, 440, 644)$ con vóxeles isotrópicos de $1{,}65$ mm. Las matrices afines (`affine`) coinciden entre los volúmenes pareados, lo cual confirma que la reducción de dosis se aplica sobre el mismo grid espacial sin re-muestreo intermedio.

Las estadísticas relevantes para el preprocesamiento, calculadas sobre el volumen de referencia `01122021_1`, son:

- La distribución de intensidades en *foreground* es muy asimétrica: mediana de 251 cuentas, percentil 99 en torno a 9.600, percentil 99,9 en torno a 19.000 y máximo cercano a $1{,}3\times10^{5}$.
- Aproximadamente el 18 % de los vóxeles del volumen son *foreground* (valor estrictamente positivo).
- En la región de *foreground*, la transformación $\log(1+x)$ aproxima una distribución $\mathcal{N}(4{,}7,\, 2{,}6)$, lo cual sugiere que una compresión logarítmica produce un rango bien centrado para el modelo.
- La desviación típica del residuo `low − full` restringido al *foreground* es del orden del 31 % de la media de señal, lo cual cuantifica el nivel de degradación introducido por la reducción de dosis.

Las decisiones de preprocesamiento son las siguientes.

**Unidad espacial.** Se trabaja con slices axiales en 2D puro. Un U-Net 3D sobre volúmenes de $440^3$ es inviable con 24 GB de memoria unificada; las variantes 2.5D (canales adicionales con slices contiguos) se reservan como extensión natural si la formulación 2D muestra inconsistencias inter-slice.

**Recorte y reescalado.** Para cada volumen se calcula una *bounding box* única a partir del *foreground* de la versión de dosis completa, y se aplica idéntica al volumen pareado de dosis baja, garantizando que ambos comparten exactamente el mismo recorte. Cada slice axial recortado se redimensiona a $256 \times 256$ píxeles mediante interpolación bilineal. Esta resolución concentra la capacidad del modelo en la anatomía relevante y reduce la carga computacional respecto al $440 \times 440$ nativo.

**Normalización per-volumen estabilizadora de varianza.** Las cuentas se mapean al rango $[-1, 1]$ mediante la transformación:

$$\mathbf{x}' = 2 \cdot \frac{\operatorname{arcsinh}(\mathbf{x}/k)}{\operatorname{arcsinh}(M/k)} - 1$$

donde $k = 10$ actúa como codo suave entre el régimen lineal y el logarítmico, y $M$ es el percentil 99,5 del **volumen de dosis completa**. El mismo $M$ se aplica al volumen pareado de dosis baja, de modo que ambos comparten escala y siguen siendo directamente comparables tras la normalización. La función $\operatorname{arcsinh}$ es la transformación canónica estabilizadora de varianza para señales positivas de cola pesada con estructura aproximadamente Poisson: se comporta linealmente cerca de cero y logarítmicamente para valores grandes, sin la patología de $\log(0)$. Los parámetros $(M, k)$ se almacenan por volumen para invertir la transformación durante la evaluación.

**Filtrado de slices.** Tras el recorte y la normalización, se conservan únicamente aquellos slices axiales cuyo *foreground* supera un umbral mínimo de ocupación (típicamente $1\%$ de vóxeles por encima de un valor bajo). Los extremos vacíos del FOV se descartan para no penalizar al modelo con cientos de slices triviales por volumen.

**Ausencia de aumento de datos.** El PET presenta asimetría clínica relevante —corazón en el lado izquierdo, hígado en el derecho, etcétera—; un volteo horizontal aleatorio induciría una lateralidad incorrecta en el modelo aprendido. El tamaño efectivo del conjunto de entrenamiento (aproximadamente $1{,}5 \times 10^{5}$ slices con *foreground*) hace innecesario recurrir a la augmentación para evitar sobreajuste.

**Particiones.** Los 371 volúmenes se dividen al nivel de fichero en proporción $80 / 10 / 10$ (~296 / 37 / 38 volúmenes) con semilla fija, y la asignación se persiste en un fichero JSON compartido por ambos pipelines. Se asume que cada fichero corresponde a un paciente independiente; el patrón de los *timestamps* en los nombres es consistente con un proceso de reconstrucción por lotes en el operador, no con frames dinámicos del mismo paciente.

**Caché en disco.** El preprocesamiento se ejecuta una única vez en modo *offline*; cada slice procesado se almacena como tensor PyTorch en formato `.pt` con precisión `float16` para evitar la decodificación repetida de NIfTI en cada época. El tamaño total estimado del caché es de aproximadamente 60 GB.

### Arquitectura y paradigma de entrenamiento

La arquitectura del U-Net es idéntica a la empleada en el modelo de mariposas, modificando exclusivamente los canales de entrada y salida. Las dimensiones y la topología se conservan:

- `block_out_channels = (128, 128, 256, 256, 512, 512)`, lo cual produce un modelo de aproximadamente 114 millones de parámetros (idéntico recuento al modelo de mariposas, dado que el primer canal convolucional contribuye una fracción despreciable al total).
- `layers_per_block = 2`.
- Bloque de atención multi-cabeza en el quinto nivel del camino descendente y ascendente (resolución espacial $16 \times 16$ tras cinco *downsamplings* del input de $256 \times 256$).

Los canales de entrada y salida dependen del pipeline:

- **Pipeline A (supervisado):** `in_channels = 2` —canal de difusión $\mathbf{x}_t$ y canal de condicionamiento $\mathbf{y}$—, `out_channels = 1`.
- **Pipeline B (prior no condicional):** `in_channels = 1`, `out_channels = 1`.

El paradigma de difusión combina tres elecciones que mejoran sobre la línea base ($\varepsilon$-prediction + schedule lineal) del modelo de mariposas:

**Predicción de velocidad (*v-prediction*).** Salimans y Ho (2022, arXiv:2202.00512) proponen reparametrizar la red para que prediga $\mathbf{v}_t = \alpha_t\,\boldsymbol{\varepsilon} - \sigma_t\,\mathbf{x}_0$ en lugar de $\boldsymbol{\varepsilon}$. Esta parametrización tiene un comportamiento numérico estable en los dos límites $t \to 0$ y $t \to T$, lo cual evita inestabilidades de entrenamiento en los timesteps de muy bajo o muy alto ruido. Su activación en `diffusers` consiste únicamente en fijar `prediction_type="v_prediction"` en el scheduler.

**Schedule cosenoidal.** Nichol y Dhariwal (2021, arXiv:2102.09672) demuestran empíricamente que un schedule de varianza $\beta_t$ con perfil cosenoidal produce una distribución de SNR a lo largo de los timesteps mejor adaptada a resoluciones medias que el schedule lineal original de Ho et al. (2020). Se mantienen $T = 1000$ timesteps de entrenamiento.

**Pérdida MSE uniforme sobre $\mathbf{v}$.** La función de pérdida es:

$$\mathcal{L} = \mathbb{E}_{t,\, \mathbf{x}_0,\, \boldsymbol{\varepsilon}}\!\left[\left\|\mathbf{v}_t - \mathbf{v}_\theta(\mathbf{x}_t,\, \mathbf{c},\, t)\right\|^2\right]$$

donde $\mathbf{c}$ representa el canal de condicionamiento $\mathbf{y}$ en Pipeline A y se omite en Pipeline B. Se opta por una ponderación uniforme sobre los timesteps, dejando el reponderado por SNR mínimo (*Min-SNR-γ*, Hang et al., 2023, arXiv:2303.09556) como mejora candidata si la convergencia resulta insuficiente.

Los hiperparámetros de optimización se ajustan a la realidad del hardware:

- Optimizador `AdamW` con tasa de aprendizaje $\eta = 10^{-4}$ y *cosine warmup* de 500 pasos, idénticos a los empleados con el modelo de mariposas.
- *Exponential moving average* (EMA) sobre los pesos del U-Net con factor de decaimiento $0{,}9999$. Los pesos EMA son los que se evalúan y se guardan como checkpoint definitivo; aportan típicamente $0{,}5$–$1$ dB adicional de PSNR sin coste de entrenamiento.
- Precisión `float32` en todo el cómputo. La implementación de MPS para PyTorch 2.10 presenta inestabilidades ocasionales con `float16`, y se prefiere renunciar a aproximadamente un 30 % de velocidad antes que arriesgar fallos catastróficos durante un entrenamiento de varios días.
- *Micro-batch* de 4 slices y acumulación de gradiente de 4 pasos, lo cual produce un batch efectivo de 16 slices —idéntico al del modelo de mariposas—.
- Se entrena durante 30 épocas, lo cual equivale a aproximadamente $1{,}1 \times 10^{6}$ pasos de gradiente sobre las $\sim 1{,}5 \times 10^{5}$ slices con *foreground*. El tiempo estimado de pared es de aproximadamente 75 horas por pipeline en el hardware objetivo.

### Pipeline A — supervisado condicional

El condicionamiento se realiza mediante **concatenación por canales**, el mecanismo dominante en la literatura supervisada de PET de dosis baja con DDPMs. La entrada del U-Net es el tensor de forma $2 \times 256 \times 256$ obtenido concatenando $\mathbf{x}_t$ —el slice de dosis completa con ruido inyectado en el timestep $t$— y $\mathbf{y}$ —el slice pareado de dosis baja, normalizado a la misma escala que $\mathbf{x}_0$—. La salida es la predicción $\hat{\mathbf{v}}_t$ de tamaño $1 \times 256 \times 256$.

En inferencia, para cada slice axial del volumen de test se parte de $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ concatenado con el correspondiente $\mathbf{y}$, y se aplica el muestreador **DDIM** (Song et al., 2021, arXiv:2010.02502) con 50 pasos de denoising y $\eta = 0$, lo cual hace el proceso determinista para una semilla dada. Tras procesar todos los slices del volumen, éstos se ensamblan en un único array tridimensional, se aplica la transformación inversa de la normalización per-volumen, y el resultado se escribe en formato NIfTI con la misma matriz afín que el volumen de entrada.

### Pipeline B — prior no condicional con guiado de medición (DPS)

El entrenamiento del Pipeline B es idéntico al del Pipeline A pero **sin el canal de condicionamiento**: el U-Net aprende $p(\mathbf{x})$ sobre los slices de dosis completa exclusivamente. La observación $\mathbf{y}$ no interviene en el entrenamiento.

En inferencia se introduce la consistencia con la medición mediante **Diffusion Posterior Sampling** (Chung et al., 2023, arXiv:2209.14687) con operador directo identidad. En cada paso DDIM se ejecuta el siguiente procedimiento:

1. La red predice $\hat{\mathbf{v}}_t$ a partir del estado actual $\mathbf{x}_t$.
2. Se reconstruye la estimación del estado limpio mediante la inversa de la parametrización $\mathbf{v}$:

$$\hat{\mathbf{x}}_0 = \alpha_t\,\mathbf{x}_t - \sigma_t\,\hat{\mathbf{v}}_t$$

3. Se aplica un paso de gradiente sobre $\mathbf{x}_t$ que penaliza el error de medición:

$$\mathbf{x}_t \leftarrow \mathbf{x}_t - \omega \cdot \nabla_{\mathbf{x}_t}\!\left[\frac{1}{2}\left\|\hat{\mathbf{x}}_0 - \mathbf{y}\right\|_2^2\right]$$

   donde $\omega$ es el escalar de guiado, único hiperparámetro tunable del procedimiento.

4. Se ejecuta el paso DDIM estándar para obtener $\mathbf{x}_{t-1}$ a partir del $\mathbf{x}_t$ ya corregido.

La elección del operador directo identidad ($A = I$) y de una varianza $\sigma$ uniforme es la aproximación más simple y defendible en este contexto, ya que el PET de dosis baja carece de una formulación limpia $\mathbf{y} = A\mathbf{x}$ —la reducción de dosis combina re-muestreo de Poisson, reducción del tiempo de escaneo y la posterior etapa de reconstrucción algorítmica, ninguna de las cuales se modela explícitamente—. La pérdida cuadrática en el espacio normalizado por $\operatorname{arcsinh}$ equivale a una verosimilitud gaussiana isotrópica sobre la representación estabilizada de varianza, suficiente como *baseline*. La sustitución de $\sigma$ uniforme por $\sigma(\mathbf{y})$ derivada de la estadística de Poisson es una de las extensiones identificadas como prioritarias en el *roadmap* posterior al MVP.

### Evaluación

Sobre el conjunto de test (~38 volúmenes) se reportan tres familias de métricas.

**Métricas de calidad de imagen.** PSNR, SSIM y NRMSE se calculan por slice y se agregan por volumen mediante la media a lo largo de los slices. Se reportan dos variantes: (i) sobre el slice completo y (ii) restringidas a la máscara de *foreground* del slice. La variante restringida es la clínicamente relevante porque ignora el fondo vacío; la variante completa es la convención estándar y permite la comparación con la literatura.

**Preservación de intensidad.** Como métrica proxy de la preservación de SUV —no se dispone de metadatos de dosis inyectada ni peso del paciente para computar SUV reales—, se reporta el error relativo entre la **media** y el **máximo** de intensidad de *foreground* por volumen, calculado en **espacio de cuentas original** (tras invertir la transformación $\operatorname{arcsinh}$).

**Evaluación cualitativa.** Para $N = 5$ volúmenes representativos del conjunto de test se generan figuras de cuatro paneles (dosis baja, dosis completa, reconstrucción, residuo absoluto) que permiten un análisis visual del comportamiento del modelo.

La comparación entre Pipeline A y Pipeline B se realiza sobre exactamente los mismos volúmenes y los mismos slices, lo cual garantiza que las métricas reportadas son directamente comparables.

### Hardware y presupuesto

Todo el entrenamiento y la inferencia se ejecutan localmente en un MacBook Pro M4 Pro con 24 GB de memoria unificada, PyTorch 2.10 con *backend* MPS. El presupuesto estimado es:

- Aproximadamente 75 horas por pipeline para 30 épocas en precisión `float32` y batch efectivo 16.
- Aproximadamente 150 horas en total para ambos pipelines completos.
- Aproximadamente 2 horas por evaluación completa del conjunto de test (38 volúmenes $\times$ ~500 slices $\times$ 50 pasos DDIM).

En términos prácticos, esto corresponde a aproximadamente una semana de cómputo principalmente nocturno para cubrir ambos pipelines.

### Estructura del código

El código fuente se organiza como un único paquete `pet_reconstruction/src/` con infraestructura compartida y módulos específicos por pipeline:

```
pet_reconstruction/src/
  config.py                       # dataclasses, sub-configs por pipeline
  volume_io.py                    # NIfTI I/O, bbox, arcsinh + inverso
  splits.py                       # split 80/10/10 a nivel paciente → JSON
  preprocess.py                   # offline: NIfTI → caché .pt
  data.py                         # PairedSliceDataset + FullDoseOnlyDataset
  metrics.py                      # PSNR/SSIM/NRMSE + preservación de intensidad
  visualize.py                    # figuras 4-panel
  model_supervised.py             # UNet builder, in_channels=2
  train_supervised.py             # bucle de entrenamiento Pipeline A
  reconstruct_supervised.py       # DDIM-50, salida NIfTI
  model_unconditional.py          # UNet builder, in_channels=1
  train_unconditional.py          # bucle de entrenamiento Pipeline B
  reconstruct_unconditional.py    # DDIM-50 + guiado DPS, salida NIfTI
  evaluate.py                     # inferencia + métricas sobre un split (test/val) → CSV + JSON + figuras
  main.py                         # dispatcher: preprocess / train / reconstruct / evaluate

```

Un preset `--smoke` (50 pacientes, 5 épocas, resolución $128 \times 128$) permite verificar el flujo completo de extremo a extremo en aproximadamente una hora antes de comprometerse a los entrenamientos largos.

### *Roadmap* posterior al MVP

Se identifican cinco extensiones, ordenadas por proximidad a la implementación del MVP.

1. **DPS con $\sigma(\mathbf{y})$ dependiente de la intensidad.** Modelar la estadística de Poisson del ruido PET propagada a través de la normalización $\operatorname{arcsinh}$ y sustituir la verosimilitud isotrópica del Pipeline B por una verosimilitud heteroscedástica. Es la extensión natural más cercana al planteamiento sobre el ruido descrito en la sección **Ruido** anterior.

2. **Contexto 2.5D o 3D.** Añadir slices vecinos como canales adicionales de entrada (2.5D) para introducir coherencia inter-slice, o pasar a un U-Net 3D sobre parches si el hardware se amplía.

3. **Reponderado de pérdida por SNR mínimo.** Activación del *Min-SNR-γ* de Hang et al. (2023) si la convergencia del *v-prediction* con 30 épocas resulta insuficiente.

4. **Modelos puente (I²SB).** Liu et al. (2023, arXiv:2302.05872) proponen entrenar un modelo cuyo proceso directo mapee directamente la distribución de dosis completa a la de dosis baja mediante un puente de Schrödinger, sustituyendo el ruido gaussiano del schedule por una transición entre las dos distribuciones de interés. Es la línea más cercana a la observación recogida en la sección **Ruido**: el proceso directo del modelo imita explícitamente la reducción de dosis. Constituye una línea prometedora para un capítulo de "contribución" en la memoria.

5. **Métricas clínicas.** En el caso de obtener metadatos de dosis inyectada y peso del paciente, computar SUV reales y reportar preservación de SUV en regiones de interés segmentadas (manualmente o con un modelo auxiliar).
