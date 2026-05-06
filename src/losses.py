import torch


def center_loss(embeddings: torch.Tensor, center: torch.Tensor | None = None):
    if center is None:
        center = embeddings.detach().mean(dim=0, keepdim=True)
    return ((embeddings - center) ** 2).sum(dim=1).mean()


@torch.no_grad()
def compute_center(model, loader, device):
    model.eval()
    vectors = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        vectors.append(out["embedding"].detach().cpu())

    if not vectors:
        raise ValueError("Cannot compute center: empty loader.")

    return torch.cat(vectors, dim=0).mean(dim=0)
