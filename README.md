# AFNO Air Quality Forecasting


- Step 1: interpolate_obs_to_grid.py - Interpolo i dati EEA_DATA in modo da ottenere una griglia uguale a CAMx
- Step 2: build_camx_training_dataset_xx.py - Genero un unico file riadattato con le finestre temporali precedenti e successive, definite nel config file.