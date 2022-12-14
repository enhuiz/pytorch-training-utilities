import torch
import matplotlib.pyplot as plt
from torch import Tensor
from collections import defaultdict

from .trainer import get_cfg, get_iteration


def is_saving():
    cfg = get_cfg()
    itr = get_iteration()
    return (
        cfg is not None
        and cfg.save_artifact_every is not None
        and itr is not None
        and itr % cfg.save_artifact_every == 0
    )


def get_cfg_itr_strict():
    assert is_saving()
    cfg = get_cfg()
    itr = get_iteration()
    assert cfg is not None
    assert itr is not None
    return cfg, itr


def save_plot(name):
    cfg, itr = get_cfg_itr_strict()
    path = (cfg.log_dir / "artifact" / name / f"{itr:06d}").with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    print(path, "saved.")
    plt.close()


def save_wav(name, wav, sr):
    cfg, itr = get_cfg_itr_strict()

    # Lazy import
    import soundfile

    path = (cfg.log_dir / "artifact" / name / f"{itr:06d}").with_suffix(".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), wav, sr)
    print(path, "saved.")


def save_tsne(name, x: Tensor | list[Tensor], label):
    # Lazy import
    from sklearn.manifold import TSNE

    if isinstance(x, list):
        x = torch.cat(x)  # (n d)

    assert isinstance(x, Tensor)

    tsne = TSNE(n_components=2)
    y = tsne.fit_transform(x.cpu().numpy())
    y_by_label = defaultdict(list)
    for i, l in enumerate(map(len, x)):
        y_by_label[label[i].item()].extend(y[:l])
        y = y[l:]
    for k, v in y_by_label.items():
        plt.scatter(*zip(*v), marker="x", alpha=0.5, label=k)
    plt.legend()

    save_plot(name)
