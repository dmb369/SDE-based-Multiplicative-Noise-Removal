import torch
import os

def save_model(filename: str, model: torch.nn.Module, verbose: bool = False):
    # Create directories if they don't exist
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Handle DataParallel or normal model
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), filename)
    else:
        torch.save(model.state_dict(), filename)
    
    if verbose:
        print(f"Model saved at {filename}")


def load_model(filename: str, model: torch.nn.Module, device: str = 'cpu') -> torch.nn.Module:
    """
    Load a model saved as regular or DataParallel, automatically map to the specified device.
    Args:
        filename: Path to the saved state dict.
        model: The model instance to load weights into.
        device: 'cpu' or 'cuda'.
    Returns:
        Model with loaded weights in eval mode.
    """
    # Load state dict to the specified device
    state_dict = torch.load(filename, map_location=torch.device(device))
    
    # If the keys were saved from a DataParallel model, remove 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model
