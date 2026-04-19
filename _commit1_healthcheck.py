"""Commit 1: Rename diagnose -> healthcheck with per-channel dead detection + param tally."""
import json
from pathlib import Path

NB_PATH = Path(r'c:/Users/ManosChatzigeorgiou/Documents/ntua/dnn-lab/DL_LabProject_25_26.ipynb')

new_healthcheck_src = '''def healthcheck(model, loader, device, criterion):
    """
    Runs one forward+backward pass on a single batch and reports:
      - Per-activation-layer: fraction of outputs <=0 overall, and count of
        "dead" channels (>95% of their activations in the <=0 region). Works
        for ReLU / LeakyReLU / SiLU / GELU — detects effective channel death
        even when the activation never outputs literal zero.
      - Per-parameter: #params, ||w||, ||g||, g.std(), #zero_g, %zero_g.
      - Overall tallies: total params, total zero-grad params,
        total ||w||, total ||g||, weight penalty.
    """
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device).float()

    # --- Per-channel dead activation check ---
    dead_stats = {}
    hooks = []
    def make_hook(name):
        def _hook(_, __, out):
            neg_mask = (out <= 0).float()
            overall_neg = neg_mask.mean().item()
            # Per-channel negative fraction: reduce over batch + spatial dims
            if out.ndim == 4:        # [B, C, H, W]
                per_ch_neg = neg_mask.mean(dim=(0, 2, 3))
            elif out.ndim == 2:      # [B, C]
                per_ch_neg = neg_mask.mean(dim=0)
            else:
                per_ch_neg = neg_mask.flatten()
            n_ch = per_ch_neg.numel()
            dead_ch = (per_ch_neg > 0.95).sum().item()
            dead_stats[name] = (overall_neg, dead_ch, n_ch)
        return _hook
    for name, m in model.named_modules():
        if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.GELU)):
            hooks.append(m.register_forward_hook(make_hook(name)))

    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    for h in hooks: h.remove()

    print("=== Activation health (<=0 fraction, dead channels) ===")
    for n, (neg_frac, dead_ch, n_ch) in dead_stats.items():
        flag = "  DEAD" if dead_ch > 0.5 * n_ch else ""
        print(f"  {n:15s} neg={neg_frac:.1%}  dead_ch={dead_ch:>3d}/{n_ch}{flag}")

    # --- Weight + gradient norms + param tally ---
    print("\\n=== Weights & Gradients per layer ===")
    print(f"  {'layer':25s} {'#params':>9s} {'|w|':>10s} {'|g|':>10s} "
          f"{'g_std':>10s} {'#zero_g':>9s} {'%zero_g':>8s}")
    total_w_sq, total_g_sq = 0.0, 0.0
    total_params, total_zero_g = 0, 0
    for name, p in model.named_parameters():
        if p.grad is None: continue
        n_params = p.numel()
        w_norm = p.data.norm().item()
        g_norm = p.grad.norm().item()
        n_zero_g = int((p.grad.abs() < 1e-8).sum().item())
        pct_zero = n_zero_g / n_params
        total_w_sq += w_norm ** 2
        total_g_sq += g_norm ** 2
        total_params += n_params
        total_zero_g += n_zero_g
        print(f"  {name:25s} {n_params:>9d} {w_norm:10.2e} {g_norm:10.2e} "
              f"{p.grad.std().item():10.2e} {n_zero_g:>9d} {pct_zero:>7.1%}")

    total_w = total_w_sq ** 0.5
    total_g = total_g_sq ** 0.5
    overall_zero_pct = total_zero_g / max(total_params, 1)
    print(f"\\n  TOTAL params:     {total_params:,} | zero-grad: {total_zero_g:,} ({overall_zero_pct:.1%})")
    print(f"  TOTAL weight norm:   {total_w:.4f}")
    print(f"  TOTAL gradient norm: {total_g:.4f}")
    print(f"  Weight penalty (wd * ||w||^2 / 2, wd=1e-4): {1e-4 * total_w_sq / 2:.6f}")
    return total_w, total_g
'''


def src_to_lines(src: str):
    """Convert a multi-line string into notebook source-list format (each line ends with \\n except last)."""
    lines = src.split('\n')
    return [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def main():
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

    # --- Cell 41: replace diagnose with healthcheck ---
    cell41 = nb['cells'][41]
    current = ''.join(cell41['source'])
    assert 'def diagnose' in current, f'Cell 41 unexpected: {current[:100]}'
    cell41['source'] = src_to_lines(new_healthcheck_src)
    cell41['outputs'] = []
    cell41['execution_count'] = None

    # --- Cell 42: update call site diagnose -> healthcheck ---
    cell42 = nb['cells'][42]
    src = ''.join(cell42['source'])
    assert 'diagnose(cv_model' in src, f'Cell 42 missing diagnose call: {src[:200]}'
    new_src = src.replace('diagnose(cv_model', 'healthcheck(cv_model')
    cell42['source'] = src_to_lines(new_src)

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('Commit 1 applied.')
    print('  - Cell 41: diagnose -> healthcheck (per-channel dead detection + param tally)')
    print('  - Cell 42: call site updated')


if __name__ == '__main__':
    main()
