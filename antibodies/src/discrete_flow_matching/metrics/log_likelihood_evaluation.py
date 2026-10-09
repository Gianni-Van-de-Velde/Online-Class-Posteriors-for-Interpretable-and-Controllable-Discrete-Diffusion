import torch


class IgLMEvaluator:
    """Antibody-specific autoregressive log-likelihood via IgLM (Shuai et al., 2021),
    trained on paired+unpaired OAS heavy/light chains -- unlike a general protein AR model,
    it conditions on chain/species tokens, so its likelihood directly measures "how
    antibody-like" a sequence is rather than "how protein-like"."""

    def __init__(self, model_name="IgLM"):
        from iglm import IgLM
        self.model = IgLM(model_name=model_name)  # picks cuda internally if available

    @torch.no_grad()
    def get_iglm_log_likelihood(self, sequence, chain_token="[HEAVY]", species_token="[HUMAN]"):
        return self.model.log_likelihood(sequence, chain_token, species_token)
