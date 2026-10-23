
from collections.abc import Collection, Sequence


def _validate_inputs(
    retrieved_ids: Sequence[Sequence[str]],
    relevant_ids: Sequence[Collection[str]],
    k: int,
) -> None:
    if k < 1:
        raise ValueError("k must be at least 1.")

    if len(retrieved_ids) != len(relevant_ids):
        raise ValueError(
            "Retrieved results and relevant IDs must have equal lengths."
        )

    if not retrieved_ids:
        raise ValueError("At least one evaluation query is required.")

    if any(not ids for ids in relevant_ids):
        raise ValueError(
            "Each evaluation query must have at least one relevant ID."
        )


def hit_rate_at_k(
    retrieved_ids: Sequence[Sequence[str]],
    relevant_ids: Sequence[Collection[str]],
    *,
    k: int = 5,
) -> float:
    """Return the fraction of queries with a relevant result in top K."""
    _validate_inputs(retrieved_ids, relevant_ids, k)

    hits = sum(
        bool(set(retrieved[:k]) & set(relevant))
        for retrieved, relevant in zip(
            retrieved_ids, relevant_ids, strict=True
        )
    )

    return hits / len(retrieved_ids)


def mean_reciprocal_rank_at_k(
    retrieved_ids: Sequence[Sequence[str]],
    relevant_ids: Sequence[Collection[str]],
    *,
    k: int = 5,
) -> float:
    """Return the mean reciprocal rank of the first relevant result."""
    _validate_inputs(retrieved_ids, relevant_ids, k)

    reciprocal_ranks: list[float] = []

    for retrieved, relevant in zip(
        retrieved_ids, relevant_ids, strict=True
    ):
        relevant_set = set(relevant)

        reciprocal_rank = next(
            (
                1.0 / rank
                for rank, item_id in enumerate(retrieved[:k], start=1)
                if item_id in relevant_set
            ),
            0.0,
        )

        reciprocal_ranks.append(reciprocal_rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
