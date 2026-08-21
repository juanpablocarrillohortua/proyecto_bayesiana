"""Pipeline stages, in execution order.

Each module exposes ``run(ctx) -> StageResult`` and is independently
runnable via ``python -m scraper run --stage N``.
"""
