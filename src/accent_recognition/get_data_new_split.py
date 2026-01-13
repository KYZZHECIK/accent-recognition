from datasets import load_dataset, DatasetDict, concatenate_datasets
from collections import Counter, defaultdict
import random

ACCENTS = [15, 22, 19, 17, 4]
ACCENT_SET = set(ACCENTS)

SPEAKER_COL = "speaker_id"
LABEL_COL = "accent"

def concat_all_splits(ds_dict: DatasetDict):
    return concatenate_datasets([ds_dict[k] for k in ds_dict.keys()])

def build_speaker_majority_label(ds, speaker_col=SPEAKER_COL, label_col=LABEL_COL):
    """
    Returns speaker -> majority label (accent) mapping.
    """
    counts = defaultdict(Counter)
    for spk, lab in zip(ds[speaker_col], ds[label_col]):
        counts[spk][lab] += 1
    return {spk: ctr.most_common(1)[0][0] for spk, ctr in counts.items()}

def split_speakers_stratified(
    speaker_to_label,
    test_size=0.13,
    val_size=0.13,
    seed=42,
    labels=None,
    min_speakers_per_class_per_split=1,
):
    """
    Stratified by label at SPEAKER level.
    Ensures each split has at least `min_speakers_per_class_per_split` speakers per label.
    """
    rng = random.Random(seed)
    labels = sorted(list(labels)) if labels is not None else sorted(set(speaker_to_label.values()))

    by_label = defaultdict(list)
    for spk, lab in speaker_to_label.items():
        by_label[lab].append(spk)

    train_spk, val_spk, test_spk = set(), set(), set()

    for lab in labels:
        spks = by_label.get(lab, [])
        rng.shuffle(spks)
        n = len(spks)

        need = 3 * min_speakers_per_class_per_split
        if n < need:
            raise ValueError(
                f"Label {lab} has only {n} speakers; need >= {need} "
                f"to guarantee {min_speakers_per_class_per_split} per split."
            )

        n_test = max(min_speakers_per_class_per_split, int(round(n * test_size)))
        n_val  = max(min_speakers_per_class_per_split, int(round(n * val_size)))

        # keep train non-empty
        if n_test + n_val >= n:
            n_test = min_speakers_per_class_per_split
            n_val  = min_speakers_per_class_per_split

        test_chunk = spks[:n_test]
        val_chunk  = spks[n_test:n_test + n_val]
        train_chunk= spks[n_test + n_val:]

        # final guard
        if len(train_chunk) < min_speakers_per_class_per_split:
            # move one from val/test if possible
            if len(val_chunk) > min_speakers_per_class_per_split:
                train_chunk.append(val_chunk.pop())
            elif len(test_chunk) > min_speakers_per_class_per_split:
                train_chunk.append(test_chunk.pop())
            else:
                raise ValueError(f"Could not keep train non-empty for label {lab}.")

        test_spk.update(test_chunk)
        val_spk.update(val_chunk)
        train_spk.update(train_chunk)

    return train_spk, val_spk, test_spk

def indices_for_speakers(ds, speakers_set, speaker_col=SPEAKER_COL):
    """
    Return indices of rows whose speaker_id is in speakers_set.
    Uses only speaker_col, so safe even if audio decoding is broken.
    """
    idx = []
    col = ds[speaker_col]
    for i, spk in enumerate(col):
        if spk in speakers_set:
            idx.append(i)
    return idx

def assert_no_speaker_leakage(dsd: DatasetDict, speaker_col=SPEAKER_COL):
    s_train = set(dsd["train"][speaker_col])
    s_val   = set(dsd["validation"][speaker_col])
    s_test  = set(dsd["test"][speaker_col])
    if (s_train & s_val) or (s_train & s_test) or (s_val & s_test):
        raise AssertionError("Speaker leakage detected across splits!")

def assert_class_representation(dsd: DatasetDict, label_col=LABEL_COL, labels=None):
    labels = set(labels) if labels is not None else None
    for split in ["train", "validation", "test"]:
        present = set(dsd[split][label_col])
        missing = labels - present
        if missing:
            raise AssertionError(f"Split '{split}' missing labels: {sorted(missing)}")

def print_stats(dsd: DatasetDict):
    for split in ["train", "validation", "test"]:
        n = len(dsd[split])
        n_spk = len(set(dsd[split][SPEAKER_COL]))
        counts = Counter(dsd[split][LABEL_COL])
        print(f"{split}: rows={n}, speakers={n_spk}, accent_counts={dict(counts)}")

if __name__ == "__main__":
    # Load + merge all splits WITH audio (we won't touch audio yet)
    ds = load_dataset("westbrook/English_Accent_DataSet")
    all_with_audio = concat_all_splits(ds)

    # Build a metadata-only view to avoid triggering audio decoding
    if "audio" in all_with_audio.column_names:
        meta = all_with_audio.remove_columns(["audio"])
    else:
        meta = all_with_audio

    # Filter to accents (safe: only uses 'accent' column)
    meta = meta.filter(lambda a: a in ACCENT_SET, input_columns=[LABEL_COL])

    # Build speaker->accent on filtered meta
    speaker_to_accent = build_speaker_majority_label(meta)

    # Stratified speaker split
    train_spk, val_spk, test_spk = split_speakers_stratified(
        speaker_to_accent,
        test_size=0.13,
        val_size=0.13,
        seed=42,
        labels=ACCENT_SET,
        min_speakers_per_class_per_split=1,  # bump to 2+ if you want stronger guarantees
    )

    # Convert speaker sets -> row indices on *filtered meta*
    train_idx = indices_for_speakers(meta, train_spk)
    val_idx   = indices_for_speakers(meta, val_spk)
    test_idx  = indices_for_speakers(meta, test_spk)

    # Now reattach audio by selecting the SAME indices from a filtered version WITH audio.
    # IMPORTANT: filter the audio dataset using the meta indices so they align 1:1.
    # Easiest: apply the same accent filter to the full dataset *without decoding audio*:
    all_with_audio_no_decode = all_with_audio
    # Avoid decoding during filter by restricting input_columns to accent only:
    all_with_audio_no_decode = all_with_audio_no_decode.filter(
        lambda a: a in ACCENT_SET,
        input_columns=[LABEL_COL],
    )

    dsd = DatasetDict(
        train=all_with_audio_no_decode.select(train_idx),
        validation=all_with_audio_no_decode.select(val_idx),
        test=all_with_audio_no_decode.select(test_idx),
    )

    # Checks
    assert_no_speaker_leakage(dsd)
    assert_class_representation(dsd, labels=ACCENT_SET)

    print(dsd)
    print_stats(dsd)

    dsd.save_to_disk("data/English_Accent_DataSet_speaker_stratified_audio")
