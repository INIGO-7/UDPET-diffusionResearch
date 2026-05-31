## Parámetros interesantes

### Transformación asinh

Se utiliza una transformación asinh para que los datos estén en el rango [-1, +1]. Esto sirve a modo de normalización de imagen, relacionado con tareas de preprocesado de los datos, en cuyo dominio es conocido que los modelos de inteligencia artificial aprenden mucho mejor los patrones cuando los datos están normalizados en un dominio definido, sin valores altos arbitrarios (e.g. un píxel puede tener intensidad 800 y otro 3; esto no conviene al entrenar modelos de IA).

Visto esto, nuestras imágenes tienen una intensidad variable, donde utilizamos el percentil 99.5 de los datos para definir lo que es +1, siendo 0 = -1. Los valores por encima del percentil no se "clippean", dejamos que sobresalgan por encima de +1. Gracias a esta normalización, podemos entregar valores en un formato correcto al modelo, además de que la transformación inversa es exacta: si "clippeamos" los valores estaríamos aplanando todo lo que esté por encima de 99.5. Más aún, así preservamos mejor el detalle de las imágenes, ya que si establecemos M = percentil 100 ergo el valor más alto, arriesgamos que el valor más alto se desvíe mucho de la distribución y que perdamos el detalle en el resto de píxeles de nuestras imágenes.

Por qué esa banda importa: los modelos de difusión asumen que los datos viven en [-1, +1] — para eso está ajustado el calendario de ruido. Los vóxeles que se encuentran bastante por encima de +1 están efectivamente fuera de distribución para el proceso de ruido hacia adelante, y el modelo tiende a reconstruirlos mal (una saturación suave / subestimación de la intensidad máxima). Por eso, al preguntarnos qué pasaría si bajáramos el percentil, podríamos observar lo siguiente:

Probablemente peor: la fidelidad de intensidad máxima en regiones calientes, las métricas de preservación de intensidad (las del espacio de conteos tras la inversión asinh), posiblemente el PSNR.
Posiblemente mejor: las métricas estructurales en tejido blando (SSIM, contraste de intensidad media), porque ahora se dedica más del rango [-1, +1] a la masa de vóxeles en lugar de "reservarse" para una cola brillante larga.

La duda es: qué pasaría si mapeamos al percentil 95? Mejoraría el rendimiento? Podemos analizar esto con un examen visual de varias imágenes mapeadas a distintos rangos para escoger cuál es mejor? Para esto hacemos un análisis de cuántos voxels se quedan fuera del rango [-1, +1] escogiendo distintos percentiles:

Aggregate %foreground mapped above +1 (mean +/- std across volumes):
  p= 90.0:  10.00% +/- 0.00%
  p= 95.0:   5.00% +/- 0.00%
  p= 99.0:   1.00% +/- 0.00%
  p= 99.5:   0.50% +/- 0.00%  <- default
  p= 99.9:   0.10% +/- 0.00%

Visto esto, se aprecia que el percentil escogido como por defecto deja pocos valores fuera, los cuáles serán outliers muy brillantes; de esta forma no pierde valores que se queden fuera del rango y preserva mejor la distribución original de los datos a la par que consigue normalizarlos.

### Tamaño de imagen

El tamaño de imagen con el que se trabaja es $256^2$, lo cuál hace que la imagen resultante sea más pequeña, dado que la original es de 440x440 píxeles. La duda surge: que pasaría si entrenamos el modelo buscando la máxima calidad, utilizando el tamaño original de las imágenes? Para empezar, no podemos usar $440^2$ dado que nuestra arquitectura U-net tiene 6 bloques, es decir hace down/upsample 5 veces: $2^5 = 32$; por lo tanto el tamaño de imagen entrante tiene que ser divisible entre 32. Aún así, se podría escoger un tamaño como $384$ o $416$, tamaños más cercanos a la calidad original que sí que cumplen con este requisito de ser divisible entre 32. No se ha podido probar con estas configuraciones dado el alcance del proyecto y, mayormente, el coste que viene con ejecutar el entrenamiento utilizando imágenes más grandes: ya es costoso para $256^2$, si subimos la calidad el coste de cualquier parte del entrenamiento crece aproximadamente de forma cuadrática.

## Workflow completo de entrenamiento

[TODO - mirar cuánto de esto ya está escrito en la memoria]