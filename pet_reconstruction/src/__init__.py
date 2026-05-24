"""Diffusion-based reconstruction of low-dose PET scans.

Two pipelines:
    - supervised conditional v-DDPM (channel concatenation of low-dose)
    - unconditional v-DDPM prior + DPS measurement-consistency guidance

See memoria/info.md, sections under "Desarrollo - modelos de difusión para
reconstrucción de escaneos PET" for the full design rationale.
"""
