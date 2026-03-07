import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers


def load_esm_c_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.device(device):
        esm_model = ESMC(
            d_model=960,
            n_heads=15,
            n_layers=30,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=True,
        ).eval()
    state_dict = torch.load(
        model_path,
        map_location=device,
    )
    esm_model.load_state_dict(state_dict)
    if device.type != "cpu":
        esm_model = esm_model.to(torch.bfloat16)
    
    return esm_model


def run_esmc_embed(sequence: str, model: ESMC) -> torch.Tensor:
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    logits_output = model.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    seq_embed = logits_output.embeddings[:, 1:-1, :]
    return seq_embed  # (1, L, 960)