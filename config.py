import torch 
from dataclasses import dataclass

@dataclass
class Config:
    dropout: float = 0.5 
    num_layers: int = 2 
    model_name: str = "blip" # [blip, clip-Vit-B/32, clip-Vit-L/14]
    device: torch.device = torch.device('cuda')
    batch_size: int = 64  # you can adjust it according to your GPU memory
    encoder: str = 'text' # ['neither', 'text', 'both']
    laion_type: str = 'laion_combined' # ['laion_combined', 'laion_template', 'laion_llm'] choose different dataset
    transform: str = 'targetpad'
    target_ratio: float = 1.25
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_epsilon: float = 1e-8
    num_epochs: int = 100
    save_best: bool = True 
    use_amp: bool = True 
    validation_frequency: int = 1
    comment: str = "fiq_test_finetune_blip_text_combined"
    dataset: str='fiq' # ['fiq', 'cirr']
    save_path_prefix = ""
    # eval related
    eval_load_path: str="D:/Documents 2.0/5th semester/computer vision/Vision Project/epoch_10_laion_combined.pth"
    submission_name: str='fiq_test_finetune_blip_text_combined'
    
    