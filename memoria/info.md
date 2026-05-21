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
- [Ene 2024] [link](https://arxiv.org/abs/2308.14190) 
- [Oct 2023] [link](https://link.springer.com/article/10.1007/s00259-023-06417-8) Propone un Denoising Diffusion Probabilistic Model (DDPM) como modelo de aprendizaje de una distribución, y evalúa distintos métodos basados en DDPM para la reducción de ruido en escaneos PET.

## Fundamentos de modelos de difusión

### `"Diffusion models for medical image reconstruction"` - Paper insights


## Generando mariposas - modelos de difusión en acción


## Desarrollo - modelos de difusión para reconstrucción de escaneos PET

### Ruido

Una idea clave para desarrollar esto es utilizar los escaneos de máxima calidad, y entrenar un modelo de difusión de tal forma que las funciones de adición de ruido imiten el tipo de ruido presente en los escaneos de dosis reducida. Típicamente, los datos PET están muestreados con ruido de Poisson de alta varianza.

A common misconception is that PET’s Poisson noise model is incompatible with the artificial Gaussian noise used in the diffusion process. This is not an issue. In general, the diffusion process maps between a useful distribution of medical images (with some inherent noise) and a known distribution. This known distribution is chosen to be Gaussian for its nice mathematical properties, though other options are possible