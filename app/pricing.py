from __future__ import annotations

from app.config import GatewayConfig, Price


def _cost(price: Price, prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000


def actual_cost(cfg: GatewayConfig, upstream_model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = cfg.pricing.get(upstream_model)
    return _cost(price, prompt_tokens, completion_tokens) if price else 0.0


def avoided_cost(
    cfg: GatewayConfig, upstream_model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """What a cache hit saved. Local models cost 0, so they are valued at their
    reference model's price; the README states this explicitly next to the numbers."""
    price = cfg.pricing.get(upstream_model)
    if price is None:
        return 0.0
    if price.input == 0 and price.output == 0 and price.reference:
        price = cfg.pricing.get(price.reference, price)
    return _cost(price, prompt_tokens, completion_tokens)
