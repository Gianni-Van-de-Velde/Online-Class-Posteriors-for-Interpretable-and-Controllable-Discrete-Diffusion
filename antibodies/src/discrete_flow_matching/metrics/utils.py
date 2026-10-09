import torch
from transformers import AutoModel, AutoTokenizer, EsmModel


@torch.no_grad()
def get_embeddings_in_batches(detector, seqs, batch_size=64):
    feats = []
    for i in range(0, len(seqs), batch_size):
        feats.append(detector(seqs[i:i + batch_size]))
    return torch.cat(feats, dim=0)


class ESM2Detector:
    """Mean-pooled ESM2 embeddings: the general-protein feature space for FPD / precision-recall."""

    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading ESM2 model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.feature_dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, sequences):
        """List of sequence strings -> [B, feature_dim] mean-pooled embeddings."""
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True,
                                max_length=1024).to(self.device)
        outputs = self.model(**inputs)

        # Mean pooling over the sequence length, respecting the attention mask
        mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = outputs.last_hidden_state * mask
        return (embeddings.sum(dim=1) / mask.sum(dim=1)).cpu()


class IgBertDetector:
    """Antibody-specific counterpart of ESM2Detector (Exscientia/IgBert, BERT trained on paired
    OAS heavy/light chains). Same mean-pooled-embedding interface, so it is a drop-in swap for
    the FPD / precision-recall / C2ST metrics (--use_igbert).

    NB: IgBert's tokenizer expects space-separated residues (protBERT-style), where ESM2's
    takes the raw sequence string."""

    def __init__(self, model_name="Exscientia/IgBert", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading IgBert model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.feature_dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, sequences):
        """List of sequence strings -> [B, feature_dim] mean-pooled embeddings."""
        spaced = [" ".join(list(seq)) for seq in sequences]
        inputs = self.tokenizer(spaced, return_tensors="pt", padding=True, truncation=True,
                                max_length=1024).to(self.device)
        outputs = self.model(**inputs)

        # Mean pooling over the sequence length, respecting the attention mask
        mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = outputs.last_hidden_state * mask
        return (embeddings.sum(dim=1) / mask.sum(dim=1)).cpu()
