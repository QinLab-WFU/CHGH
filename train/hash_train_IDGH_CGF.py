import torch


class Trainer:
    """Training scaffold with a caller-supplied base objective and optimizer.

    The objective accepts image embeddings, text embeddings and multi-label
    targets, and returns a scalar loss. Full CHGH training is not in this preview.
    """

    def __init__(self, model, objective, optimizer):
        self.model = model
        self.objective = objective
        self.optimizer = optimizer

    def train_epoch(self, train_loader, *, base_only=False):
        if not base_only:
            raise NotImplementedError("Full CHGH training is not included in this preview.")

        self.model.train()
        device = next(self.model.parameters()).device
        loss_sum = 0.0
        batches = 0
        for image, text, labels, _ in train_loader:
            if train_loader.batch_size is not None and len(image) < train_loader.batch_size:
                continue
            image = image.to(device, dtype=torch.float32, non_blocking=True)
            text = text.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            image_embedding, text_embedding = self.model(image, text)
            loss = self.objective(image_embedding, text_embedding, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            self.optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        if batches == 0:
            raise ValueError("No full training batches")
        return loss_sum / batches
