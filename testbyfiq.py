import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import wandb # Make sure wandb is imported
import multiprocessing
import torch
from torch.utils.data import DataLoader
from torch import nn
import random
import numpy as np
from trainer import Trainer
from config import Config
import datetime
from utils import get_model, set_grad, get_preprocess, get_laion_cirr_dataset, get_laion_fiq_dataset, extract_index_features, collate_fn, get_optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def main(cfg):
    # setup_seed(0)
    # get the corresponding model
    model = get_model(cfg)
    set_grad(cfg, model)
    model.pretrained_model.eval().float()


    # input_dim = combiner.blip_model.visual.input_resolution
    if cfg.model_name.startswith('blip'):
        input_dim = 384
    elif cfg.model_name.startswith('clip'):
        input_dim = model.pretrained_model.visual.input_resolution
    preprocess = get_preprocess(cfg, model, input_dim)

    # --- Dataset Loading Logic ---
    # Need to load validation data even for eval-only
    if cfg.dataset == 'fiq':
        val_dress_types = ['dress', 'toptee', 'shirt']
        # We don't need relative_train_dataset for evaluation, but the function returns it.
        # Use dummy variable names or handle appropriately if needed elsewhere.
        _, relative_val_dataset, classic_val_dataset, idx_to_dress_mapping = get_laion_fiq_dataset(preprocess, val_dress_types, cfg.laion_type)
        # We don't need the train loader for evaluation
        # relative_train_loader = DataLoader(...)
    elif cfg.dataset == 'cirr':
        # Even if evaluating FIQ, keep this structure in case cfg.dataset is accidentally set to 'cirr'
        _, relative_val_dataset, classic_val_dataset = get_laion_cirr_dataset(preprocess, cfg.laion_type)
        # We don't need the train loader for evaluation
        # relative_train_loader = DataLoader(...)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset}")

    # --- Precomputing Index Features (Still relevant for eval if encoder is frozen) ---
    kwargs = {}
    if cfg.dataset == 'fiq':
        kwargs['val_index_features'] = []
        kwargs['val_index_names'] = []
        kwargs['val_total_index_features'] = []
        kwargs['idx_to_dress_mapping'] = idx_to_dress_mapping
        # Precompute features if visual encoder is frozen during training (and thus for eval)
        if cfg.encoder == 'text' or cfg.encoder == 'neither':
            print("Precomputing FashionIQ validation index features...")
            for classic_val_dataset_ in classic_val_dataset:
                val_index_features, val_index_names, _ = extract_index_features(classic_val_dataset_, model, return_local=False)
                kwargs['val_index_features'].append(val_index_features)
                kwargs['val_index_names'].append(val_index_names)
                kwargs['val_total_index_features'].append(_)
            print("Finished precomputing features.")
    elif cfg.dataset == 'cirr': # Keep this logic for completeness
         if cfg.encoder == 'text' or cfg.encoder == 'neither':
            val_index_features, val_index_names, val_total_index_features = extract_index_features(classic_val_dataset, model, return_local=False)
            kwargs['val_index_features'], kwargs['val_index_names'], kwargs['val_total_index_features'] = val_index_features, val_index_names, val_total_index_features

    # --- Optimizer, Scheduler, Criterion (Not needed for eval, but keep for Trainer init) ---
    optimizer = get_optimizer(model, cfg) # Dummy optimizer for init
    # Dummy scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, eta_min=1e-8)
    crossentropy_criterion = nn.CrossEntropyLoss(ignore_index=-100) # Dummy criterion

    # --- Initialize Trainer ---
    # Need a dummy train_loader if the Trainer class requires it
    dummy_train_loader = None # Or DataLoader with an empty list if necessary
    trainer = Trainer(cfg, model, dummy_train_loader, optimizer, lr_scheduler, crossentropy_criterion, classic_val_dataset, relative_val_dataset, **kwargs)

    # --- Original Training Call (Commented Out) ---
    # trainer.train()

    # --- ADDED CODE FOR FIQ EVALUATION ---
    print("\n--- Starting FashionIQ Evaluation-Only Mode ---")

    # (1) Define path and load your FIQ model checkpoint
    # !!! IMPORTANT: Replace this path with the actual path to your FIQ model !!!
   # In testbyfig.py inside the main() function
    model_path = "D:\Documents 2.0\5th semester\computer vision\Vision Project\epoch_05_laion_template.pth"
    print(f"Loading model checkpoint from: {model_path}")
    # Load the state dict. Ensure the model architecture matches the checkpoint.
    # Use map_location='cpu' if you are loading a GPU-trained model onto a CPU environment
    # Use map_location=cfg.device if loading onto the specified device (GPU or CPU)
    try:
        #model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        model.load_state_dict(torch.load(model_path, map_location=cfg.device), strict=False)
        print("Model checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        print("Please ensure the model_path is correct and the checkpoint matches the model architecture.")
        return # Exit if loading fails

    model.eval() # Set model to evaluation mode

    # (2) Run the FIQ evaluation function from the trainer
    print("Running trainer.eval_fiq()...")
    # Make sure eval_fiq handles the case where features weren't precomputed (if encoder='both')
    # Also ensure wandb.log is NOT commented out in trainer.py if using wandb here
    results10, results50 = trainer.eval_fiq()

    print("\n--- FashionIQ Evaluation Complete ---")
    print("Recall@10 Results (Order depends on val_dress_types: ['dress', 'toptee', 'shirt']):")
    print(results10)
    print("Recall@50 Results (Order depends on val_dress_types: ['dress', 'toptee', 'shirt']):")
    print(results50)

    # Calculate and print averages
    if results10 and results50: # Check if results are not empty
        avg_recall10 = sum(results10) / len(results10)
        avg_recall50 = sum(results50) / len(results50)
        print(f"\nAverage Recall@10: {avg_recall10:.2f}")
        print(f"Average Recall@50: {avg_recall50:.2f}")
    else:
        print("\nEvaluation produced no results.")
    # --- END OF ADDED CODE ---


if __name__ == '__main__':
    cfg = Config()
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    # This path might not be needed if not saving anything during eval, but keep for structure
    cfg.save_path = f"{cfg.save_path_prefix}/{current_time}_{cfg.comment}_eval_run.pth"

    wandb_config = vars(cfg)

    # --- UNCOMMENTED wandb init ---
    print("Initializing wandb...")
    # Using a different project name for clarity, or keep the original if preferred
    wandb.init(project='ZeroShot-CIR-Eval', notes=cfg.comment + " (Evaluation Only)", config=wandb_config, name=cfg.comment + "_eval")

    print(f"Running evaluation for dataset: {cfg.dataset}")
    if cfg.dataset != 'fiq':
        print("WARNING: config.dataset is not set to 'fiq'. This script is modified for FIQ evaluation.")
        # Optionally, force cfg.dataset to 'fiq' here if desired:
        # cfg.dataset = 'fiq'
        # print("Forcing dataset to 'fiq'.")

    main(cfg) # Calls the modified main function

    # --- UNCOMMENTED wandb finish ---
    print("Finishing wandb run...")
    wandb.finish()

    print("\nScript finished.")