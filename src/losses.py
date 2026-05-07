from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans


def center_loss(
    embeddings: torch.Tensor,
    center: torch.Tensor | None = None,
):
    """
    Training-time compactness loss.

    During training, this keeps normal embeddings compact.
    If multiple centers are provided, distance is computed to nearest center.
    """

    if center is None:
        center = embeddings.detach().mean(dim=0, keepdim=True)

    if center.dim() == 1:
        center = center.unsqueeze(0)

    if center.size(0) > 1:
        dist = torch.cdist(embeddings, center, p=2) ** 2
        min_dist = dist.min(dim=1).values
        return min_dist.mean()

    return ((embeddings - center) ** 2).sum(dim=1).mean()


@torch.no_grad()
def collect_embeddings(
    model,
    loader,
    device,
    desc: str = "Collecting embeddings",
):
    model.eval()
    vectors = []

    progress = tqdm(loader, desc=desc, unit="batch")

    for batch in progress:
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
        raise ValueError("No embeddings collected. Loader is empty.")

    return torch.cat(vectors, dim=0)


@torch.no_grad()
def compute_center(
    model,
    loader,
    device,
    desc: str = "Computing normal center",
    num_prototypes: int = 1,
    prototype_method: str = "kmeans",
    seed: int = 42,
):
    """
    Compute normal center(s).

    If num_prototypes = 1:
        returns tensor shape: [projection_dim]

    If num_prototypes > 1:
        returns tensor shape: [num_prototypes, projection_dim]
    """

    embeddings = collect_embeddings(
        model=model,
        loader=loader,
        device=device,
        desc=desc,
    )

    n_samples = embeddings.size(0)
    dim = embeddings.size(1)

    num_prototypes = int(num_prototypes)

    if num_prototypes <= 1:
        center = embeddings.mean(dim=0)
        print(f"[Normal center computed] shape={tuple(center.shape)}")
        return center

    if num_prototypes > n_samples:
        print(
            f"[Warning] num_prototypes={num_prototypes} is larger than "
            f"number of samples={n_samples}. Using num_prototypes={n_samples}."
        )
        num_prototypes = n_samples

    print(
        f"[Multi-prototype center] method={prototype_method}, "
        f"k={num_prototypes}, samples={n_samples}, dim={dim}"
    )

    x = embeddings.numpy()

    if prototype_method == "kmeans":
        kmeans = KMeans(
            n_clusters=num_prototypes,
            random_state=seed,
            n_init=10,
        )
        kmeans.fit(x)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    elif prototype_method == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_samples, size=num_prototypes, replace=False)
        centers = embeddings[idx].clone()

    else:
        raise ValueError(
            f"Unknown prototype_method={prototype_method}. "
            "Use 'kmeans' or 'random'."
        )

    print(f"[Normal prototypes computed] shape={tuple(centers.shape)}")

    return centers