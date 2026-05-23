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

## Desarrollo - modelos de difusión para reconstrucción de escaneos PET

### Ruido

Una idea clave para desarrollar esto es utilizar los escaneos de máxima calidad, y entrenar un modelo de difusión de tal forma que las funciones de adición de ruido imiten el tipo de ruido presente en los escaneos de dosis reducida. Típicamente, los datos PET están muestreados con ruido de Poisson de alta varianza.

A common misconception is that PET’s Poisson noise model is incompatible with the artificial Gaussian noise used in the diffusion process. This is not an issue. In general, the diffusion process maps between a useful distribution of medical images (with some inherent noise) and a known distribution. This known distribution is chosen to be Gaussian for its nice mathematical properties, though other options are possible