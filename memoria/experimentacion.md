
## Experimentos

### Supervisado



## Métricas

PSNR/SSIM/NRMSE evalúan la fidelidad estructural, y se miden sobre el espacio normalizado de las imágenes: esto se debe a que PSNR colapsa al voxel más brillante, las muestras de PET tienen un altísimo rango dinámico, entonces el PSNR es prácticamente una evaluación de cómo de bien se ha podido reproducir el spot más brillante, y es numéricamente insensible a los errores en el resto de la imagen. En el SSIM, los datos atípicos en la imagen harían que nos diera un SSIM más alto de lo debido. Además, la media a lo largo de todos los pacientes necesita una escala común, para que los cálculos intra-volumen sean justos y precisos. 

La reformulación crucial: asinh es monótonica e invertible — la misma transformación aplicada tanto a recon como a GT. No oculta el error; lo repondera para que los errores relativos en todo el rango dinámico cuenten en lugar de los errores absolutos en el pico. Entonces PSNR/SSIM en espacio normalizado es una comparación fiel del resultado final sin transformaciones — simplemente bajo una ponderación de intensidad donde la métrica no está rota.

Utilizamos el count-space para evaluar intensidad cuantitativa.

## Limitaciones

La mayor limitación del proyecto es la demanda de máquinas de alto rendimiento para la experimentación, y el amplio espacio de experimentos que se pueden llevar a cabo. Desde cambiar los parámetros del paradigma de entrenamiento (tamaño de imagen, learning rate, estabilización), hasta la configuración de tratamiento de las imágenes médicas () y las posibles elecciones de arquitectura (DDIM vs DDPM, U-net + attention, DPS, tratar de imitar el ruido en imágenes low-dose, etc); todo combinado con los límites de hardware que solo permiten realizar una cantidad limitada de experimentos, ha sido el mayor reto del proyecto. Por suerte esto se ha mitigado haciendo una investigación de campo potente en la cuál se han escogido las ideas más interesantes de los últimos papers del estado del arte en este dominio: sin esto, hubiera sido muy complicado poder llegar a la fecha ya que se hubiera necesitado una experimentación mucho mayor, comprobando qué funciona y qué no, y donde rompe la arquitectura.