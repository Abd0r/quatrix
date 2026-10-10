"""Token-weighted causal LM evaluation. Original artifacts remain unchanged."""
import math
import torch
import torch.nn.functional as F


def collate_ids(sequences):
    """JSONL token ids already contain any protocol-specific BOS/EOS tokens.

    Padding is detected by length, never token identity: GPT-2 id 0 is valid.
    Right padding is harmless for earlier causal predictions, not scored itself.
    """
    if not sequences or any(len(s) < 2 for s in sequences):
        raise ValueError('Each sequence must contain at least two tokens')
    width = max(map(len, sequences))
    ids = torch.zeros(len(sequences), width, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, seq in enumerate(sequences):
        values = torch.as_tensor(seq, dtype=torch.long)
        if (values < 0).any():
            raise ValueError('Input token IDs must be nonnegative')
        ids[i, :len(seq)] = values
        labels[i, :len(seq)] = values
    return ids, labels


@torch.no_grad()
def evaluate(model, sequences, batch_size=4, device='cpu'):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    model.eval()
    nll_sum = 0.0
    token_count = 0
    sample_count = 0
    per_sample = []
    for start in range(0, len(sequences), batch_size):
        group = sequences[start:start+batch_size]
        ids, labels = collate_ids(group)
        ids, labels = ids.to(device), labels.to(device)
        logits = model(input_ids=ids)['logits']
        if logits.shape[:2] != ids.shape:
            raise ValueError('This evaluator supports text-only matching-length logits')
        targets = labels[:, 1:]
        nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                              targets.reshape(-1), ignore_index=-100, reduction='none')
        if not torch.isfinite(nll).all():
            raise ValueError('Non-finite evaluation loss')
        per_row_nll = nll.reshape(targets.shape).double().sum(dim=1).cpu().tolist()
        per_row_counts = (targets != -100).sum(dim=1).cpu().tolist()
        for offset, (row_nll, row_count) in enumerate(zip(per_row_nll, per_row_counts)):
            per_sample.append({'row': start+offset, 'nll_sum': row_nll, 'n_tokens': row_count})
        nll_sum += sum(per_row_nll)
        token_count += sum(per_row_counts)
        sample_count += len(group)
    if token_count == 0:
        raise ValueError('Empty evaluation data')
    mean = nll_sum / token_count
    return {'n_samples':sample_count, 'n_supervised_tokens':token_count,
            'nll_sum':nll_sum, 'mean_nll':mean, 'perplexity':math.exp(mean),
            'per_sample':per_sample}
