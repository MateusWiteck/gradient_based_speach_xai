import pandas as pd


def temporal_relevance_to_dataframe(
    temporal_relevance,
    audio_duration_seconds: float,
):
    """
    Converts temporal relevance [1, tokens] or [tokens]
    into a dataframe with start/end time for each token.
    """

    relevance = temporal_relevance.squeeze().detach().cpu().numpy()
    num_tokens = len(relevance)
    seconds_per_token = audio_duration_seconds / num_tokens

    rows = []

    for token_idx, score in enumerate(relevance):
        start_time = token_idx * seconds_per_token
        end_time = (token_idx + 1) * seconds_per_token

        rows.append(
            {
                "token_idx": token_idx,
                "start_time": start_time,
                "end_time": end_time,
                "relevance": float(score),
            }
        )

    return pd.DataFrame(rows)