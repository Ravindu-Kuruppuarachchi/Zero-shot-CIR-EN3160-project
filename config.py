import torch 
from dataclasses import dataclass

@dataclass
class Config:
    dropout: float = 0.5 
    num_layers: int = 2
    # In config.py
    model_name: str = "clip-Vit-B/32" # Changed from "blip" 
    #model_name: str = "blip" # [blip, clip-Vit-B/32, clip-Vit-L/14]
    device: torch.device = torch.device('cuda')
    batch_size: int = 8  # you can adjust it according to your GPU memory
    encoder: str = 'text' # ['neither', 'text', 'both']
    laion_type: str = 'laion_template' # ['laion_combined', 'laion_template', 'laion_llm'] choose different dataset
    transform: str = 'targetpad'
    target_ratio: float = 1.25
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_epsilon: float = 1e-8
    num_epochs: int = 20
    save_best: bool = True 
    use_amp: bool = True 
    validation_frequency: int = 1
    comment: str = "fiq_test_template"
    dataset: str='fiq' # ['fiq', 'cirr']
    save_path_prefix = "/wandb"
    # eval related
    eval_load_path: str= "D:\Documents 2.0\5th semester\computer vision\Vision Project\epoch_05_laion_template.pth"
    submission_name: str='fiq_test_template'

    # Set this to the folder where you want checkpoints to be saved
    #save_path_prefix: str = "D:/Documents 2.0/5th semester/computer vision/Vision Project/ZS-CIR/runs"
    # A descriptive name for this training run
    #comment: str = "train_blip_text_fashion_subset_2k"

    # --- Paths (Not strictly needed for this eval method, but good practice) ---
    #save_path_prefix = "D:/Documents 2.0/5th semester/computer vision/Vision Project/ZS-CIR/runs" # Or any valid path
    #comment: str = "fiq_evaluation_run"

