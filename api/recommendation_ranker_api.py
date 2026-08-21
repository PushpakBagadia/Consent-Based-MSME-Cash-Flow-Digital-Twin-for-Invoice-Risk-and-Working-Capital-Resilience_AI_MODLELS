"""
Model 7 - Non-Debt-First Recommendation Ranker API.

Wraps the ranking logic from model_7.py behind a router, mounted into
main.py the same way as Models 2, 3, and 8.

If model_2_url isn't given, falls back to a local model_2_input.json,
then to CSV-only liquidity estimates (fail-soft, same behaviour as the
CLI script).
"""
from fastapi import APIRouter, HTTPException, Query

from model_7 import (
    load_dataset,
    validate_dataset,
    calculate_customer_metrics,
    calculate_overall_metrics,
    load_model_2_output,
    calculate_liquidity_gap,
    rank_recovery_options,
    create_output,
    CSV_FILE,
)

router = APIRouter(prefix="/recommend", tags=["Model 7 - Recovery Ranker"])


@router.get("/health")
def recommend_health():
    return {"status": "ok"}


@router.get("")
def get_recommendations(
    model_2_url: str | None = Query(
        None,
        description="URL to fetch Model 2's live forecast/summary JSON from, "
                    "e.g. http://localhost:8000/cashflow/forecast. "
                    "Falls back to local model_2_input.json, then CSV-only.",
    ),
):
    """
    Rank non-debt recovery strategies against the current liquidity gap.

    Returns the same shape as model_7_output.json: recommendations[],
    summary, weights.
    """
    try:
        df = load_dataset(CSV_FILE)
        validate_dataset(df)

        customer_metrics = calculate_customer_metrics(df)
        overall_metrics = calculate_overall_metrics(df)

        model_2_data = load_model_2_output(model_2_url)
        liquidity_data = calculate_liquidity_gap(model_2_data, df)

        recommendations = rank_recovery_options(liquidity_data)

        return create_output(
            recommendations,
            liquidity_data,
            customer_metrics,
            overall_metrics,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model 7 failed: {e}")