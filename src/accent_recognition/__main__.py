import os, random, math, logging, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.cuda.amp import autocast, GradScaler
from datasets import load_from_disk, Audio
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Define LSTM model for feature-based approach
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 2 for bidirectional
    
    def forward(self, x, lengths):
        # x: [B, T, F] padded sequence, lengths: list of actual lengths
        # Pack the sequence for variable length handling in LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n shape: (num_layers*directions, B, hidden_dim). For 1-layer biLSTM, shape = (2, B, H).
        # Separate final forward and backward hidden states:
        h_forward = h_n[-2]  # last forward hidden
        h_backward = h_n[-1] # last backward hidden
        h_cat = torch.cat([h_forward, h_backward], dim=1)  # shape (B, 2*hidden_dim)
        logits = self.fc(h_cat)
        return logits

# Data collator for custom features (MFCC or Log-Mel)
class FeatureCollator:
    def __init__(self, feature_type="mel", apply_specaugment=False):
        self.feature_type = feature_type
        self.apply_specaugment = apply_specaugment
        # Define audio feature transforms
        self.sr = 16000
        # 25ms window (n_fft=400), 10ms hop (hop_length=160)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=400, hop_length=160, n_mels=64
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr, n_mfcc=20, melkwargs={"n_fft":400, "hop_length":160, "n_mels":64}
        )
        # SpecAugment masks
        if self.apply_specaugment:
            # mask param – maximum width of the mask
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)   # mask up to 8 mel bands
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)       # mask up to 30 time frames
    
    def __call__(self, batch):
        # batch is a list of dataset examples
        specs = []
        lengths = []
        labels = []
        for example in batch:
            # Load audio array and label
            waveform = example["audio"]["array"]  # numpy array
            label = example["accent"]
            # Convert to torch tensor
            wave_tensor = torch.tensor(waveform, dtype=torch.float32)
            # Compute features
            if self.feature_type == "mfcc":
                feat = self.mfcc_transform(wave_tensor)
                # MFCC transform returns shape (n_mfcc, time)
                feat = feat.transpose(0, 1)  # shape: (time, n_mfcc)
            else:
                # Mel spectrogram, then log (dB)
                mel_spec = self.mel_transform(wave_tensor)  # shape: (n_mels, time)
                mel_spec_db = self.to_db(mel_spec)          # convert to log scale (dB)
                feat = mel_spec_db.transpose(0, 1)          # shape: (time, n_mels)
                # Apply SpecAugment if training
                if self.apply_specaugment:
                    # Note: apply two masks of each type
                    feat = self.freq_mask(feat)
                    feat = self.freq_mask(feat)
                    feat = self.time_mask(feat)
                    feat = self.time_mask(feat)
            specs.append(feat)
            lengths.append(feat.shape[0])
            labels.append(label)
        # Pad sequences to the same length (max in batch)
        batch_size = len(specs)
        max_len = max(lengths)
        feat_dim = specs[0].shape[1]
        batch_feats = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
        for i, feat in enumerate(specs):
            L = feat.shape[0]
            batch_feats[i, :L, :] = feat  # pad remaining with zeros
        # Convert labels to tensor
        batch_labels = torch.tensor(labels, dtype=torch.long)
        return batch_feats, lengths, batch_labels

class AudioCollator:
    def __init__(self, feature_extractor, max_seconds=4.0, train_crop=False):
        self.feat_extractor = feature_extractor
        self.sr = feature_extractor.sampling_rate
        self.max_len = int(max_seconds * self.sr)
        self.train_crop = train_crop

        if hasattr(self.feat_extractor, "return_attention_mask"):
            self.feat_extractor.return_attention_mask = True

    def _trim(self, x):
        if len(x) <= self.max_len:
            return x
        if self.train_crop:
            start = random.randint(0, len(x) - self.max_len)
            return x[start:start + self.max_len]
        return x[:self.max_len]

    def __call__(self, batch):
        audio_list = [self._trim(ex["audio"]["array"]) for ex in batch]
        labels = torch.tensor([ex["accent"] for ex in batch], dtype=torch.long)

        inputs = self.feat_extractor(
            audio_list,
            sampling_rate=self.sr,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_values = inputs["input_values"]
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)

        return input_values, attention_mask, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/English_Accent_DataSet_speaker_stratified_audio",
                        help="Path to the prepared dataset directory")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base",
                        help="Model identifier (e.g., 'facebook/wav2vec2-base' or 'microsoft/wavlm-base' or 'lstm')")
    parser.add_argument("--feature_type", type=str, default="mel", choices=["mel", "mfcc", "raw"],
                        help="Type of input features to use (ignored if using HF model that expects raw).")
    parser.add_argument("--specaugment", action="store_true", help="Apply SpecAugment (only for mel features).")
    parser.add_argument("--weighted_loss", action="store_true", help="Use weighted cross-entropy for class imbalance.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for optimizer.")
    parser.add_argument("--output_dir", type=str, default="accent_model_output", help="Directory to save logs and models.")
    parser.add_argument("--max_seconds", type=float, default=4.0)
    parser.add_argument("--train_crop", action="store_true", help="random crop during training")
    parser.add_argument("--fp16", action="store_true", help="use mixed precision (CUDA only)")
    parser.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--freeze_backbone", action="store_true",
                    help="freeze pretrained encoder and train only classifier head")
    parser.add_argument("--unfreeze_last_n", type=int, default=0,
                        help="if >0, unfreeze last N transformer layers (still freeze rest)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Setup logging to file and console
    log_file = os.path.join(args.output_dir, "training.log")
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.FileHandler(log_file), logging.StreamHandler()
    ])
    logger = logging.getLogger()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path} ...")
    dsd = load_from_disk(args.data_path)
    # Ensure audio is 16k and map labels to 0..num_labels-1
    # Cast audio to 16k using datasets.Audio feature
    dsd = dsd.cast_column("audio", Audio(sampling_rate=16000))
    # Get unique accents and map to [0, ..., num_labels-1]
    labels = sorted(set(dsd["train"]["accent"]))
    label2id = {str(label): idx for idx, label in enumerate(labels)}
    id2label = {idx: str(label) for idx, label in enumerate(labels)}
    num_labels = len(labels)
    # Map accent column to label indices in each split
    def encode_label(example):
        example["accent"] = label2id[str(example["accent"])]
        return example
    dsd = dsd.map(encode_label)
    logger.info(f"Labels mapped to indices: {label2id}")
    # Remove unnecessary columns to speed up
    drop_cols = [col for col in dsd["train"].column_names if col not in ("audio", "accent")]
    if drop_cols:
        dsd = dsd.remove_columns(drop_cols)
    
    # Prepare model and collator
    custom_model = None
    hf_model = None
    collator = None
    if args.model_name.lower() in ["lstm", "cnn", "rnn"] or args.model_name.lower() == "custom":
        # Use custom LSTM model on features
        feat_type = args.feature_type
        if feat_type == "raw":
            # raw not supported for custom, default to mel
            feat_type = "mel"
        logger.info(f"Using custom LSTM model with {feat_type.upper()} features")
        input_dim = 20 if feat_type == "mfcc" else 64  # MFCC outputs 20 dims, Mel has 64 dims
        custom_model = LSTMClassifier(input_dim, hidden_dim=128, num_classes=num_labels)
        collator = FeatureCollator(feature_type=feat_type, apply_specaugment=args.specaugment)
    else:
        # Use a pretrained HF model (Wav2Vec2, WavLM, etc.)
        model_id = args.model_name
        logger.info(f"Using pretrained model {model_id} (fine-tuning for classification)")
        # Load feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        hf_model = AutoModelForAudioClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        )
        if args.freeze_backbone:
            # Freeze everything first
            for p in hf_model.parameters():
                p.requires_grad = False

            head_modules = []
            for name in ["classifier", "projector"]:
                if hasattr(hf_model, name):
                    head_modules.append(getattr(hf_model, name))

            if not head_modules:
                raise RuntimeError("Couldn't find classifier/projector modules to unfreeze.")

            for m in head_modules:
                for p in m.parameters():
                    p.requires_grad = True

            # Optionally unfreeze last N transformer layers for a small amount of adaptation
            if args.unfreeze_last_n > 0:
                base = getattr(hf_model, "wav2vec2", None) or getattr(hf_model, "wavlm", None)
                if base is None:
                    # AutoModel wrapper sometimes stores it under .wav2vec2 even for wavlm-class models,
                    # but keep this explicit and fail loudly if unknown.
                    raise RuntimeError("Couldn't locate base model (wav2vec2/wavlm) to unfreeze layers.")

                enc_layers = base.encoder.layers
                n = args.unfreeze_last_n
                for layer in enc_layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        collator = AudioCollator(feature_extractor, max_seconds=args.max_seconds, train_crop=True)
        # If feature_extractor has normalization, it will be applied in collator
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hf_model:
        hf_model.to(device)
    if custom_model:
        custom_model.to(device)
    # Setup optimizer
    if hf_model:
        trainable_params = [p for p in hf_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(custom_model.parameters(), lr=(args.lr if args.lr else 1e-3))
    use_amp = args.fp16 and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    
    # Setup loss function
    if args.weighted_loss:
        # compute class weights from train split
        counts = [0] * num_labels
        for lbl in dsd["train"]["accent"]:
            counts[lbl] += 1
        weights = [0] * num_labels
        total = len(dsd["train"])
        for i, c in enumerate(counts):
            # Avoid division by zero (shouldn't happen since each class in train has >=1 due to stratification)
            weights[i] = total / (num_labels * c) if c > 0 else 0.0
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        logger.info(f"Using weighted loss, class_weights = {weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Prepare data loaders
    train_loader = torch.utils.data.DataLoader(dsd["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader   = torch.utils.data.DataLoader(dsd["validation"], batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader  = torch.utils.data.DataLoader(dsd["test"], batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    
    # Training loop
    best_val_f1 = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        # Train
        if hf_model: hf_model.train()
        if custom_model: custom_model.train()
        total_loss = 0.0
        for batch in train_loader:

            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                if hf_model:
                    input_values, attention_mask, labels = batch
                    input_values = input_values.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                else:
                    features, lengths, labels = batch
                    features = features.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                with autocast(enabled=use_amp):
                    if hf_model:
                        outputs = hf_model(input_values=input_values, attention_mask=attention_mask)
                        logits = outputs.logits
                    else:
                        logits = custom_model(features, lengths)

                    loss = criterion(logits, labels)
                    loss = loss / args.grad_accum

                scaler.scale(loss).backward()

                if (step + 1) % args.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item() * args.grad_accum

            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        
        # Validate
        if hf_model: hf_model.eval()
        if custom_model: custom_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                if hf_model:
                    input_values, attention_mask, labels = batch
                    input_values = input_values.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    outputs = hf_model(input_values=input_values, attention_mask=attention_mask)
                    logits = outputs.logits
                else:
                    features, lengths, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    logits = custom_model(features, lengths)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Acc = {val_acc:.4f}, Val F1 = {val_f1:.4f}")
        
        # Checkpoint every other epoch
        if epoch % 2 == 0:
            ckpt_name = f"model_epoch{epoch}"
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            if hf_model:
                hf_model.save_pretrained(ckpt_path)
            else:
                torch.save(custom_model.state_dict(), ckpt_path + ".pt")
            logger.info(f"Saved model checkpoint at epoch {epoch} to {ckpt_path}")
        # Track best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            # Save best model (you could also copy state_dict in memory)
            best_path = os.path.join(args.output_dir, "best_model")
            if hf_model:
                hf_model.save_pretrained(best_path)
            else:
                torch.save(custom_model.state_dict(), best_path + ".pt")
            logger.info(f"New best model (Val F1={best_val_f1:.4f}) saved to {best_path}")
    
    # After training, evaluate on test set using the best model
    # (Reload best model from disk to ensure we use the best weights)
    if best_epoch > 0:
        logger.info(f"Loading best model from epoch {best_epoch} for final evaluation on test set")
        if hf_model:
            # reload into hf_model (same architecture)
            if os.path.isdir(best_path):
                hf_model = AutoModelForAudioClassification.from_pretrained(best_path)
            hf_model.to(device).eval()
        else:
            if os.path.isfile(best_path + ".pt"):
                custom_model.load_state_dict(torch.load(best_path + ".pt"))
            custom_model.to(device).eval()
    # Evaluate on test data
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            if hf_model:
                input_values, attention_mask, labels = batch
                input_values = input_values.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                logits = hf_model(input_values=input_values, attention_mask=attention_mask).logits
            else:
                features, lengths, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                logits = custom_model(features, lengths)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    logger.info(f"Test Set Performance: Accuracy = {test_acc:.4f}, Macro F1 = {test_f1:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}, Macro F1: {test_f1:.4f}")
    
if __name__ == "__main__":
    main()
