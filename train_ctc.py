import argparse
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from data.real_dataset import MultiSourceSignLanguageDataset
from models.asl_ctc_transformer import ASLCTCModel
from nlp.ctc_beam_search import CTCBeamSearch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BLANK_ID = 0
CHAR_TO_ID = {" ": 1}
for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    CHAR_TO_ID[char] = i + 2
ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}
ID_TO_CHAR[BLANK_ID] = ""


class TrainingConfig:
    def __init__(self):
        self.num_classes = 28
        self.num_epochs = 100
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.gradient_clip = 1.0
        self.num_workers = 2
        self.use_amp = True
        self.use_landmarks = True
        self.dropout = 0.25
        self.num_frames = 30
        self.image_size = 64
        self.include_sources = ("images", "letters")
        self.preprocessing_roi = "mediapipe_tight_hand_crop"
        self.crop_padding = 0.35
        self.lr_scheduler_patience = 5
        self.lr_scheduler_factor = 0.5
        self.early_stopping_patience = 20
        self.early_stopping_delta = 1e-4
        self.beam_width = 5
        self.blank_penalty_weight = 0.35
        self.short_target_blank_penalty_weight = 0.80
        self.classification_weight = 1.0
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)


def stratified_train_val_split(dataset, val_fraction=0.2, seed=42, min_val_group_size=2):
    generator = torch.Generator().manual_seed(seed)
    groups = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        groups[(sample["source"], sample["label"])].append(idx)

    train_indices = []
    val_indices = []
    for indices in groups.values():
        perm = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in perm]
        if len(shuffled) < min_val_group_size:
            train_indices.extend(shuffled)
            continue
        val_count = max(1, int(round(len(shuffled) * val_fraction)))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    if not val_indices:
        all_indices = torch.randperm(len(dataset), generator=generator).tolist()
        split = int((1.0 - val_fraction) * len(dataset))
        train_indices = all_indices[:split]
        val_indices = all_indices[split:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def collate_fn(batch):
    frames_list = [item["frames"] for item in batch]
    landmarks_list = [item.get("landmarks") for item in batch]
    labels_list = [item["labels"] for item in batch]
    frame_lens = torch.stack([item["frame_len"] for item in batch])
    label_lens = torch.stack([item["label_len"] for item in batch])
    texts = [item.get("text", "") for item in batch]
    sources = [item.get("source", "unknown") for item in batch]

    max_frame_len = max(frames.shape[0] for frames in frames_list)
    frames_padded = []
    landmarks_padded = []
    for frames, landmarks in zip(frames_list, landmarks_list):
        pad_len = max_frame_len - frames.shape[0]
        if pad_len > 0:
            frames = torch.cat([frames, torch.zeros(pad_len, *frames.shape[1:], dtype=frames.dtype)], dim=0)
            landmarks = torch.cat([landmarks, torch.zeros(pad_len, landmarks.shape[1], dtype=landmarks.dtype)], dim=0)
        frames_padded.append(frames)
        landmarks_padded.append(landmarks)

    class_targets = []
    class_mask = []
    for item in batch:
        is_letter_like = item.get("source") in {"images", "letters"} and item["label_len"].item() == 1
        class_mask.append(is_letter_like)
        class_targets.append(int(item["labels"][0].item()) if is_letter_like else 0)

    return {
        "frames": torch.stack(frames_padded),
        "landmarks": torch.stack(landmarks_padded),
        "labels": torch.cat(labels_list),
        "frame_lens": frame_lens,
        "label_lens": label_lens,
        "texts": texts,
        "sources": sources,
        "class_targets": torch.tensor(class_targets, dtype=torch.long),
        "class_mask": torch.tensor(class_mask, dtype=torch.bool),
    }


class CharacterErrorRate:
    @staticmethod
    def edit_counts(predicted, reference):
        if len(reference) == 0:
            return {"distance": len(predicted), "substitutions": 0, "insertions": len(predicted), "deletions": 0,
                    "correct": 0, "ref_len": 0, "pred_len": len(predicted)}
        rows = len(predicted) + 1
        cols = len(reference) + 1
        d = [[0] * cols for _ in range(rows)]
        op = [[""] * cols for _ in range(rows)]
        for i in range(rows):
            d[i][0] = i
            op[i][0] = "ins"
        for j in range(cols):
            d[0][j] = j
            op[0][j] = "del"
        for i in range(1, rows):
            for j in range(1, cols):
                cost = 0 if predicted[i - 1] == reference[j - 1] else 1
                choices = (
                    (d[i - 1][j] + 1, "ins"),
                    (d[i][j - 1] + 1, "del"),
                    (d[i - 1][j - 1] + cost, "ok" if cost == 0 else "sub"),
                )
                d[i][j], op[i][j] = min(choices, key=lambda item: item[0])
        i = len(predicted)
        j = len(reference)
        subs = ins = dels = correct = 0
        while i > 0 or j > 0:
            action = op[i][j]
            if action == "ok":
                correct += 1; i -= 1; j -= 1
            elif action == "sub":
                subs += 1; i -= 1; j -= 1
            elif action == "ins":
                ins += 1; i -= 1
            else:
                dels += 1; j -= 1
        return {"distance": d[-1][-1], "substitutions": subs, "insertions": ins, "deletions": dels,
                "correct": correct, "ref_len": len(reference), "pred_len": len(predicted)}


class CTCLossWithAuxiliary(nn.Module):
    def __init__(self, num_classes, blank=0, ctc_weight=0.5, aux_weight=1.0,
                 blank_penalty_weight=0.35, short_target_blank_penalty_weight=0.80,
                 classification_weight=1.0, class_weights=None):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)
        self.aux = nn.CrossEntropyLoss(ignore_index=blank)
        self.class_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=blank)
        self.blank = blank
        self.ctc_weight = ctc_weight
        self.aux_weight = aux_weight
        self.blank_penalty_weight = blank_penalty_weight
        self.short_target_blank_penalty_weight = short_target_blank_penalty_weight
        self.classification_weight = classification_weight

    @staticmethod
    def repeated_frame_targets(labels, label_lens, time_steps):
        targets = []
        idx = 0
        for length in label_lens.tolist():
            seq = labels[idx:idx + length]
            idx += length
            if length == 0:
                targets.append(torch.zeros(time_steps, dtype=torch.long, device=labels.device))
                continue
            positions = torch.linspace(0, length - 1, steps=time_steps, device=labels.device).long()
            targets.append(seq[positions])
        return torch.stack(targets)

    def forward(self, logits, labels, frame_lens, label_lens, class_logits=None, class_targets=None, class_mask=None):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        ctc_loss = self.ctc(log_probs.permute(1, 0, 2), labels, frame_lens, label_lens)
        targets = self.repeated_frame_targets(labels, label_lens, logits.shape[1])
        aux_loss = self.aux(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        probs = torch.softmax(logits, dim=-1)
        blank_probs = probs[..., self.blank]
        blank_penalty = blank_probs.mean()
        short_mask = label_lens <= 1
        short_blank_penalty = blank_probs[short_mask].mean() if short_mask.any() else blank_probs.new_tensor(0.0)

        total = (
            self.ctc_weight * ctc_loss
            + self.aux_weight * aux_loss
            + self.blank_penalty_weight * blank_penalty
            + self.short_target_blank_penalty_weight * short_blank_penalty
        )
        if class_logits is not None and class_targets is not None and class_mask is not None and class_mask.any():
            total = total + self.classification_weight * self.class_loss(class_logits[class_mask], class_targets[class_mask])
        return total


class CTCTrainer:
    def __init__(self, config, class_weights=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)
        self.model = ASLCTCModel(config.num_classes, use_landmarks=config.use_landmarks, dropout=config.dropout).to(self.device)
        self.model.train()
        self.criterion = CTCLossWithAuxiliary(
            num_classes=config.num_classes,
            blank=BLANK_ID,
            blank_penalty_weight=config.blank_penalty_weight,
            short_target_blank_penalty_weight=config.short_target_blank_penalty_weight,
            classification_weight=config.classification_weight,
            class_weights=class_weights.to(self.device) if class_weights is not None else None,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=config.use_amp and self.device.type == "cuda")
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=config.lr_scheduler_factor, patience=config.lr_scheduler_patience)
        self.beam_search = CTCBeamSearch(beam_width=config.beam_width, blank=BLANK_ID)
        self.cer_fn = CharacterErrorRate()
        self.best_val_cer = float("inf")
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_path = None
        logger.info("Model parameters: %s", f"{sum(p.numel() for p in self.model.parameters()):,}")

    @staticmethod
    def labels_to_text(labels, label_lens):
        texts = []
        idx = 0
        for length in label_lens.tolist():
            seq = labels[idx:idx + length].tolist()
            idx += length
            texts.append("".join(ID_TO_CHAR.get(int(token), "") for token in seq))
        return texts

    @staticmethod
    def summarize_counts(counts):
        precision_den = counts["correct"] + counts["substitutions"] + counts["insertions"]
        recall_den = counts["correct"] + counts["substitutions"] + counts["deletions"]
        precision = counts["correct"] / max(precision_den, 1)
        recall = counts["correct"] / max(recall_den, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            "cer": counts["distance"] / max(counts["ref_len"], 1),
            "exact": counts["exact"] / max(counts["samples"], 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            frames = batch["frames"].to(self.device)
            landmarks = batch["landmarks"].to(self.device)
            labels = batch["labels"].to(self.device)
            frame_lens = batch["frame_lens"].to(self.device)
            label_lens = batch["label_lens"].to(self.device)
            class_targets = batch["class_targets"].to(self.device)
            class_mask = batch["class_mask"].to(self.device)
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=self.config.use_amp and self.device.type == "cuda"):
                outputs = self.model(frames, landmarks if self.config.use_landmarks else None, return_aux=True)
                loss = self.criterion(
                    outputs["ctc_logits"],
                    labels,
                    frame_lens,
                    label_lens,
                    class_logits=outputs["class_logits"],
                    class_targets=class_targets,
                    class_mask=class_mask,
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                logger.info("Batch %d: Loss=%.4f", batch_idx + 1, loss.item())
        return total_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        aggregate = defaultdict(int)
        by_source = defaultdict(lambda: defaultdict(int))
        for batch in dataloader:
            frames = batch["frames"].to(self.device)
            landmarks = batch["landmarks"].to(self.device)
            labels = batch["labels"].to(self.device)
            frame_lens = batch["frame_lens"].to(self.device)
            label_lens = batch["label_lens"].to(self.device)
            class_targets = batch["class_targets"].to(self.device)
            class_mask = batch["class_mask"].to(self.device)
            outputs = self.model(frames, landmarks if self.config.use_landmarks else None, return_aux=True)
            logits = outputs["ctc_logits"]
            loss = self.criterion(logits, labels, frame_lens, label_lens, outputs["class_logits"], class_targets, class_mask)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1).permute(1, 0, 2)
            refs = self.labels_to_text(labels, label_lens)
            class_pred_ids = outputs["class_logits"].argmax(dim=-1).cpu().tolist()
            for b in range(probs.shape[1]):
                source = batch["sources"][b]
                if source in {"images", "letters"} and label_lens[b].item() == 1:
                    pred_text = ID_TO_CHAR.get(int(class_pred_ids[b]), "")
                else:
                    pred_text = self.beam_search.decode(probs[:, b, :].cpu().numpy(), ID_TO_CHAR)
                ref_text = refs[b]
                counts = self.cer_fn.edit_counts(pred_text, ref_text)
                counts["samples"] = 1
                counts["exact"] = int(pred_text == ref_text)
                for key, value in counts.items():
                    aggregate[key] += value
                    by_source[source][key] += value
        metrics = self.summarize_counts(aggregate)
        source_metrics = {source: self.summarize_counts(counts) for source, counts in sorted(by_source.items())}
        return total_loss / max(len(dataloader), 1), metrics, source_metrics

    def save_checkpoint(self, epoch, loss, val_cer=None, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "val_cer": val_cer,
            "config": self.config.__dict__,
        }
        path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)
        torch.save(checkpoint, self.config.checkpoint_dir / "latest_model.pt")
        if is_best:
            self.best_model_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, self.best_model_path)
            logger.info("Best model saved: %s", self.best_model_path)

    def train(self, train_loader, val_loader):
        logger.info("Starting training...")
        for epoch in range(self.config.num_epochs):
            logger.info("\n%s", "=" * 50)
            logger.info("Epoch %d/%d", epoch + 1, self.config.num_epochs)
            logger.info("%s", "=" * 50)
            train_loss = self.train_epoch(train_loader)
            logger.info("Train Loss: %.4f", train_loss)
            val_loss, val_metrics, source_metrics = self.validate(val_loader)
            val_cer = val_metrics["cer"]
            logger.info(
                "Val Loss: %.4f, CER: %.4f, Exact: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                val_loss, val_metrics["cer"], val_metrics["exact"], val_metrics["precision"], val_metrics["recall"], val_metrics["f1"],
            )
            for source, metrics in source_metrics.items():
                logger.info(
                    "  %s: CER=%.4f Exact=%.4f Precision=%.4f Recall=%.4f F1=%.4f",
                    source, metrics["cer"], metrics["exact"], metrics["precision"], metrics["recall"], metrics["f1"],
                )
            self.scheduler.step(val_loss)
            is_best = val_cer < self.best_val_cer - self.config.early_stopping_delta
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_cer = val_cer
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_cer=val_cer, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_loss, val_cer=val_cer, is_best=False)
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info("Early stopping triggered after %d epochs", epoch + 1)
                break
        logger.info("\nTraining complete! Best model: %s", self.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid ASL letter model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-landmarks", action="store_true")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--crop-padding", type=float, default=None)
    parser.add_argument("--sources", default="images,letters", help="Comma-separated dataset sources.")
    args = parser.parse_args()

    config = TrainingConfig()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.no_landmarks:
        config.use_landmarks = False
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.crop_padding is not None:
        config.crop_padding = args.crop_padding
    include_sources = tuple(source.strip() for source in args.sources.split(",") if source.strip())
    config.include_sources = include_sources

    full_dataset = MultiSourceSignLanguageDataset(
        root_dir="data/videos",
        image_root="data/images",
        num_frames=config.num_frames,
        image_size=config.image_size,
        include_sources=include_sources,
        use_landmarks=config.use_landmarks,
        drop_missing_image_landmarks=config.use_landmarks,
        crop_padding=config.crop_padding,
    )
    class_weights = full_dataset.get_class_weights()
    train_dataset, val_dataset = stratified_train_val_split(full_dataset, val_fraction=0.2, seed=42)
    logger.info("Train samples: %d, validation samples: %d", len(train_dataset), len(val_dataset))

    all_weights = full_dataset.get_sample_weights()
    train_weights = all_weights[train_dataset.indices]
    source_weight_totals = defaultdict(float)
    for sample_idx, sample_weight in zip(train_dataset.indices, train_weights.tolist()):
        source_weight_totals[full_dataset.samples[sample_idx]["source"]] += sample_weight
    total_sampler_weight = sum(source_weight_totals.values())
    for source, value in sorted(source_weight_totals.items()):
        logger.info("Sampler share %s: %.1f%%", source, 100.0 * value / max(total_sampler_weight, 1e-12))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True),
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    CTCTrainer(config, class_weights=class_weights).train(train_loader, val_loader)
