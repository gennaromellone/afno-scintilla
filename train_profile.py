import torch.multiprocessing as mp
import time

def main():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    import os
    import csv
    import yaml
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error
    from torch.profiler import profile, record_function, ProfilerActivity, schedule

    from torch.amp import autocast, GradScaler

    import numpy as np
    import logging, time, itertools
    from pathlib import Path
    from my_dataset.afno_dataset import AFNOZarrObsDatasetOptimized
    from models.afno_modulus import AFNOModel

    # ------------------------------------------------------------------------
    #  GLOBAL SETTINGS
    # ------------------------------------------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = True         # ⇒ Tensor‑Core in TF32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #NUM_WORKERS = max(4, os.cpu_count() // 2)            # es. 8‑12 su CPU 16c
    NUM_WORKERS=8
    PREFETCH = 8

    #NUM_WORKERS = 4
    base_path = "/home/gmellone/afno-scintilla/configs"
    train_config_path = os.path.join(base_path, "train.yaml")
    model_config_path = os.path.join(base_path, "model.yaml")
    data_config_path  = os.path.join(base_path, "data.yaml")

    # === LOGGER ===
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("train_afno")

    # === CONFIG ===
    with open(train_config_path) as f:
        train_cfg = yaml.safe_load(f)

    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)

    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)

    # === PARAMETRI ===
    sim_zarr_path = data_cfg["training_simulated_path"]
    obs_npz_path = data_cfg["interpolated_observation_path"]
    input_vars = data_cfg["input_vars"]
    output_vars = data_cfg["output_vars"]
    species_vars = data_cfg["species_vars"]

    img_shape = tuple(model_cfg["img_shape"])
    time_window = model_cfg["time_window"]
    forecast_horizon = model_cfg["forecast_horizon"]

    def expand_vars(var_list, species_vars):
        expanded = []
        for var in var_list:
            if var == "PM10":
                expanded += species_vars["PM25"] + species_vars["PM10_extra"]
            elif var == "PM25":
                expanded += species_vars["PM25"]
            else:
                expanded.append(var)
        return expanded

    # === HYPERPARAMS ===
    BATCH_SIZE = train_cfg["batch_size"]
    EPOCHS = train_cfg["epochs"]
    LR = train_cfg["learning_rate"]
    ALPHA = train_cfg["alpha"]
    BETA = train_cfg["beta"]
    VAL_SPLIT = train_cfg.get("val_split", 0.2)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    step_log_every = 100 

    logger.info(f"CUDA visible devices: {torch.cuda.device_count()}")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Input vars: {input_vars}, Output vars: {output_vars}")
    logger.info(f"Time window: {time_window}, Forecast horizon: {forecast_horizon}")

    # === DATASET ===
    dataset = AFNOZarrObsDatasetOptimized(
        sim_zarr_path,
        obs_npz_path,
        input_vars=input_vars,
        output_vars=output_vars,
        species_vars=species_vars
    )
    logger.info(f"Loading Dataset ...")
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    logger.info(f"Loading Dataloader ...")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, prefetch_factor=None, timeout=0)
    #train_loader = DataLoader(
    #    train_set,
    #    batch_size=BATCH_SIZE,
    #    shuffle=False,
    #    num_workers=NUM_WORKERS,
    #    pin_memory=True,
    #    prefetch_factor=PREFETCH,
    #    persistent_workers=True,
    #    timeout=120,
    #    drop_last=True
    #)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH, timeout=0)
    #logger.info(f"Start Warmup ...")
    #t0 = time.time()
    #for i, _ in zip(range(200), train_loader):
    #    pass
    # logger.info(f"Mediana load time: {(time.time()-t0)/200} s")
    # === MODEL ===
    expanded_input_vars = expand_vars(input_vars, species_vars)
    in_channels = time_window * len(expanded_input_vars)
    out_channels = len(output_vars)

    logger.info(f"Loading Model ...")
    model = AFNOModel(
        img_shape=img_shape,
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=model_cfg.get("patch_size", [2, 4]),
        embed_dim=model_cfg.get("embed_dim", 256),
        depth=model_cfg.get("depth", 8),
        num_blocks=model_cfg.get("num_blocks", 8)
    ).to(DEVICE)

    # === SETUP CSV ===
    CSV_LOG = "training_log.csv"
    if not Path(CSV_LOG).exists():
        with open(CSV_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "mae", "rmse", "r2", "explained_variance", "max_error"])

    # === TRAINING ===
    model = torch.compile(model, mode="reduce-overhead")
    scaler = GradScaler()

    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.MSELoss()

    logger.info("\n🚀 Inizio training AFNO con dual loss (CAMx + osservazioni)...\n")

    # ------------------------------------------------------------------
    # PROFILER set‑up

    prof_sched = schedule(wait=1, warmup=1, active=3, repeat=1)  
    prof_kwargs = dict(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./profiler_log", worker_name="main")
    )

    
    # ------------------------------------------------------------------
    last_log = time.time()
    with profile(**prof_kwargs) as prof:
        for epoch in range(train_cfg["epochs"]):
            model.train()
            total_loss = 0

            for step, (x, y) in enumerate(train_loader):
                t0 = time.time()

                x = x.to(DEVICE, non_blocking=True)
                logging.info(f"X-SHAPE= {x.shape}")
                y = y.to(DEVICE, non_blocking=True)

                with autocast(device_type="cuda"):                     # ── Mixed Precision
                    pred      = model(x)
                    loss_sim  = criterion(pred, y)
                    loss_obs  = torch.tensor(1., device=DEVICE)  # TODO
                    loss      = train_cfg["alpha"] * loss_sim + \
                                train_cfg["beta"]  * loss_obs

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                prof.step()                        
                if step % step_log_every == 0:
                    logging.info(f"ep {epoch} step {step}/{len(train_loader)} "
                                f"loss {loss_sim.item():.3f} "
                                f"elapsed {(time.time()-last_log):.1f}s")
                    last_log = time.time()
                '''
                if step % 20 == 0:
                    torch.cuda.synchronize()
                    logger.info(f"epoch {epoch} | step {step:4d} "
                                f"| {(time.time()-t0)*1e3:5.1f} ms | "
                                f"loss {loss_sim.item():.4f}")
                '''
                

                #if step == 59:                      # trace corto (facolt.)
                #    break

            # ------------------------------------------------------------
            #  Riepilogo testuale: 20 operatori ordinati per tempo CUDA
            # ------------------------------------------------------------
            print("\n=== Torch profiler – top 20 op per self_cuda_time_total ===")
            print(prof.key_averages()
                    .table(sort_by="self_cuda_time_total", row_limit=20))
            
            avg_loss = total_loss / len(train_loader)
            # === VALIDAZIONE ===
            model.eval()
            y_true = []
            y_pred = []

            with torch.no_grad():
                for batch in val_loader:
                    x, y_obs = batch
                    x, y_obs = x.to(DEVICE), y_obs.to(DEVICE)
                    pred = model(x)

                    y_true.append(y_obs.cpu().numpy())
                    y_pred.append(pred.cpu().numpy())

            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

            mse = mean_squared_error(y_true.ravel(), y_pred.ravel())
            mae = mean_absolute_error(y_true.ravel(), y_pred.ravel())
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true.ravel(), y_pred.ravel())
            explained_var = explained_variance_score(y_true.ravel(), y_pred.ravel())
            max_err = max_error(y_true.ravel(), y_pred.ravel())

            logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.6f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

            with open(CSV_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_loss, mae, rmse, r2, explained_var, max_err])

    # === SALVATAGGIO ===
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/afno_model.pt")
    logger.info("\n✅ Modello salvato in checkpoints/afno_model.pt")

    config_text = f"\n Training configuration: \n- EPOCHS: {EPOCHS}\n- BATCH_SIZE: {BATCH_SIZE}\n- LEARNING_RATE: {LR}\n- ALPHA: {ALPHA}\n- BETA: {BETA}\n- TIME_WINDOW: {time_window}\n- FORECAST_HORIZON: {forecast_horizon}\n- INPUT_VARS: {input_vars}\n- OUTPUT_VARS: {output_vars}\n- IMG_SHAPE: {img_shape}"

    with open("checkpoints/afno_model_info.txt", "w") as f:
        f.write(config_text)

    logger.info("\n✅ Configurazione di training salvata in checkpoints/afno_model_info.txt")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
