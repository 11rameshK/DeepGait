import torch
import torch.nn.functional as F

def batch_all_triplet_loss(embeddings, labels, margin=0.2):
    """
    Batch-all triplet loss over flattened embeddings.
    embeddings: [N, ...]
    labels: [N]
    """
    x = embeddings.flatten(1)
    d = torch.cdist(x, x, p=2)
    labels = labels.view(-1)
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos_mask = same & ~eye
    neg_mask = ~same
    ap = d[:, :, None]
    an = d[:, None, :]
    losses = F.relu(ap - an + margin)
    valid = pos_mask[:, :, None] & neg_mask[:, None, :]
    vals = losses[valid]
    if vals.numel() == 0:
        return d.new_tensor(0.0)
    return vals.mean()
