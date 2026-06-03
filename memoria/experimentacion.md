
## Experimentos

### Supervisado

***epoch 50***
- "psnr_fg": 29.941329540122112,
- "psnr_whole": 34.524634523543476,
- "ssim_fg": 0.9638390274204348,
- "ssim_whole": 0.9206913763098331,
- "nrmse_fg": 0.03750145999946974,
- "nrmse_whole": 0.02126243088470933,
- "mean_pct_err": -4.177727755669889,
- "max_pct_err": -14.644118229392996

***epoch 100***
- "psnr_fg": 30.204600212528327,
- "psnr_whole": 35.31630623781877,
- "ssim_fg": 0.9651728881635684
- "ssim_whole": 0.9251425834803598,
- "nrmse_fg": 0.03639723158958674,
- "nrmse_whole": 0.019435456209188604,
- "mean_pct_err": -4.749601186820047,
- "max_pct_err": -18.473233750199388,

## Limitaciones

La mayor limitación del proyecto es la demanda de máquinas de alto rendimiento para la experimentación, y el amplio espacio de experimentos que se pueden llevar a cabo. Desde cambiar los parámetros del paradigma de entrenamiento (tamaño de imagen, learning rate, estabilización), hasta la configuración de tratamiento de las imágenes médicas () y las posibles elecciones de arquitectura (DDIM vs DDPM, U-net + attention, DPS, tratar de imitar el ruido en imágenes low-dose, etc); todo combinado con los límites de hardware que solo permiten realizar una cantidad limitada de experimentos, ha sido el mayor reto del proyecto. Por suerte esto se ha mitigado haciendo una investigación de campo potente en la cuál se han escogido las ideas más interesantes de los últimos papers del estado del arte en este dominio: sin esto, hubiera sido muy complicado poder llegar a la fecha ya que se hubiera necesitado una experimentación mucho mayor, comprobando qué funciona y qué no, y donde rompe la arquitectura.