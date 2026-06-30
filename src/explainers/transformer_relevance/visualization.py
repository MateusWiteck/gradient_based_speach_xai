"""Readable, time-aligned visualizations for token-level audio relevance."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch


def plot_relevance_timeline(
    waveform,
    sampling_rate: int,
    df_relevance,
    output_path: str | None = None,
    relevance_title: str = "Transformer relevance rollout",
    top_k: int = 5,
):
    """Plot an audio spectrogram and directly readable per-token relevance bars.

    Each bar corresponds to one HuBERT time token. Its height is the actual
    relevance score, while its colour gives the same information at a glance.
    The most relevant intervals are outlined both in the bar chart and on the
    spectrogram, making it easy to relate a peak to its audio region.
    """
    required_columns = {"start_time", "end_time", "relevance"}
    missing_columns = required_columns.difference(df_relevance.columns)
    if missing_columns:
        raise ValueError(f"df_relevance is missing columns: {sorted(missing_columns)}")

    waveform_np = waveform.detach().cpu().numpy().squeeze()
    if waveform_np.ndim != 1:
        raise ValueError("waveform must contain one mono audio channel.")

    starts = df_relevance["start_time"].to_numpy(dtype=float)
    ends = df_relevance["end_time"].to_numpy(dtype=float)
    relevance = df_relevance["relevance"].to_numpy(dtype=float)
    if relevance.size == 0 or not np.isfinite(relevance).all():
        raise ValueError("relevance must contain at least one finite score.")

    widths = ends - starts
    if np.any(widths <= 0):
        raise ValueError("Each relevance row must have end_time > start_time.")

    duration = waveform_np.size / sampling_rate
    relevance_max = max(float(relevance.max()), np.finfo(float).eps)
    colour_norm = colors.Normalize(vmin=0, vmax=relevance_max)
    colour_map = plt.get_cmap("magma")
    bar_colours = colour_map(colour_norm(relevance))

    top_k = max(0, min(top_k, relevance.size))
    top_indices = np.argsort(relevance)[-top_k:] if top_k else np.array([], dtype=int)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.35]},
        constrained_layout=True,
    )
    spectrogram_axis, relevance_axis = axes

    # A spectrogram makes the time/frequency content of speech or other audio
    # readable where a dense waveform would look like visual noise.
    nfft = min(512, waveform_np.size)
    noverlap = min(int(nfft * 0.75), nfft - 1)
    spectrogram_axis.specgram(
        waveform_np,
        NFFT=nfft,
        Fs=sampling_rate,
        noverlap=noverlap,
        cmap="Greys",
        scale="dB",
    )

    # Shade token intervals faintly over the audio representation. This
    # connects the bar peaks below to their corresponding parts of the signal.
    for start, end, colour in zip(starts, ends, bar_colours):
        spectrogram_axis.axvspan(start, end, color=colour, alpha=0.16, linewidth=0)

    spectrogram_axis.set_xlim(0, duration)
    # Log frequency gives the speech-relevant low frequencies enough visual
    # room instead of compressing them into a thin band at the bottom.
    spectrogram_axis.set_yscale("log")
    spectrogram_axis.set_ylim(50, sampling_rate / 2)
    spectrogram_axis.set_ylabel("Frequency (Hz)")
    spectrogram_axis.set_title("Spectrogram with token-relevance intervals")

    bars = relevance_axis.bar(
        starts,
        relevance,
        width=widths,
        align="edge",
        color=bar_colours,
        edgecolor="none",
        linewidth=0,
    )
    for index in top_indices:
        bars[index].set_edgecolor("#00e5ff")
        bars[index].set_linewidth(1.4)
        spectrogram_axis.axvspan(
            starts[index],
            ends[index],
            facecolor="none",
            edgecolor="#00acc1",
            linewidth=1.1,
        )

    relevance_axis.set_xlim(0, duration)
    relevance_axis.set_ylim(0, relevance_max * 1.15)
    relevance_axis.set_xlabel("Time (s)")
    relevance_axis.set_ylabel("Token\nrelevance")
    relevance_axis.set_title(relevance_title)
    relevance_axis.grid(axis="y", alpha=0.25)
    relevance_axis.legend(
        handles=[Patch(facecolor="none", edgecolor="#00e5ff", label=f"Top {top_k} token intervals")],
        loc="upper right",
        frameon=False,
    )

    colourbar = fig.colorbar(
        ScalarMappable(norm=colour_norm, cmap=colour_map),
        ax=axes,
        pad=0.015,
        aspect=35,
    )
    colourbar.set_label("Token relevance")

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_class_relevance_heatmaps(
    waveform,
    sampling_rate: int,
    relevance_by_label: dict[str, object],
    output_path: str | None = None,
    title: str = "Class-specific temporal relevance",
):
    """Plot one spectrogram with a time-aligned relevance heatmap per class.

    ``relevance_by_label`` should contain normalized 1-D relevance tensors or
    arrays that share the same HuBERT token count. It is intended for direct
    predicted-vs-runner-up inspection on the exact same audio utterance.
    """
    if len(relevance_by_label) < 2:
        raise ValueError("Provide relevance for at least two classes to compare.")

    waveform_np = waveform.detach().cpu().numpy().squeeze()
    if waveform_np.ndim != 1:
        raise ValueError("waveform must contain one mono audio channel.")

    labels = list(relevance_by_label)
    relevance_rows = []
    for label in labels:
        relevance = relevance_by_label[label]
        if hasattr(relevance, "detach"):
            relevance = relevance.detach().cpu().numpy()
        relevance = np.asarray(relevance, dtype=float).reshape(-1)
        if relevance.size == 0 or not np.isfinite(relevance).all():
            raise ValueError(f"Invalid relevance values for {label!r}.")
        relevance_rows.append(relevance)
    token_counts = {row.size for row in relevance_rows}
    if len(token_counts) != 1:
        raise ValueError("All class relevance vectors must have the same token count.")

    relevance_matrix = np.vstack(relevance_rows)
    duration = waveform_np.size / sampling_rate
    relevance_max = max(float(relevance_matrix.max()), np.finfo(float).eps)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    spectrogram_axis, heatmap_axis = axes

    nfft = min(512, waveform_np.size)
    spectrogram_axis.specgram(
        waveform_np,
        NFFT=nfft,
        Fs=sampling_rate,
        noverlap=min(int(nfft * 0.75), nfft - 1),
        cmap="Greys",
        scale="dB",
    )
    spectrogram_axis.set_yscale("log")
    spectrogram_axis.set_ylim(50, sampling_rate / 2)
    spectrogram_axis.set_ylabel("Frequency (Hz)")
    spectrogram_axis.set_title("Audio spectrogram")

    image = heatmap_axis.imshow(
        relevance_matrix,
        aspect="auto",
        interpolation="nearest",
        extent=[0, duration, 0, len(labels)],
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=relevance_max,
    )
    heatmap_axis.set_yticks(np.arange(len(labels)) + 0.5, labels)
    heatmap_axis.set_xlabel("Time (s)")
    heatmap_axis.set_ylabel("Target class")
    heatmap_axis.set_title(title)
    figure.colorbar(image, ax=heatmap_axis, pad=0.015, label="Token relevance")

    if output_path is not None:
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved class-specific heatmap to: {output_path}")
    else:
        plt.show()
    plt.close(figure)
