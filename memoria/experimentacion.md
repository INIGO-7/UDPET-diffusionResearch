
## Experimentos

### Supervisado



## Métricas

PSNR/SSIM/NRMSE evalúan la fidelidad estructural, y se miden sobre el espacio normalizado de las imágenes: esto se debe a que PSNR colapsa al voxel más brillante, las muestras de PET tienen un altísimo rango dinámico, entonces el PSNR es prácticamente una evaluación de cómo de bien se ha podido reproducir el spot más brillante, y es numéricamente insensible a los errores en el resto de la imagen. En el SSIM, los datos atípicos en la imagen harían que nos diera un SSIM más alto de lo debido. Además, la media a lo largo de todos los pacientes necesita una escala común, para que los cálculos intra-volumen sean justos y precisos. 

La reformulación crucial: asinh es monótonica e invertible — la misma transformación aplicada tanto a recon como a GT. No oculta el error; lo repondera para que los errores relativos en todo el rango dinámico cuenten en lugar de los errores absolutos en el pico. Entonces PSNR/SSIM en espacio normalizado es una comparación fiel del resultado final sin transformaciones — simplemente bajo una ponderación de intensidad donde la métrica no está rota.

Utilizamos el count-space para evaluar intensidad cuantitativa.