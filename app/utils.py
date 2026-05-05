def fallback_reason(data: dict):
    """
    Fallback reasoning logic.
    Used only when model-driven explanation is unavailable.
    """

    reasons = []

    try:
        # Safe extraction with defaults
        session = float(data.get('SessionTime', 0))
        inactivity = float(data.get('DaysSinceLastPurchase', 0))
        review = float(data.get('ReviewScore', 5))

        # Simple heuristics
        if session < 100:
            reasons.append("Low engagement")

        if inactivity > 30:
            reasons.append("High inactivity")

        if review < 3:
            reasons.append("Low satisfaction")

    except Exception:
        # Fail-safe
        return ["Insufficient data for reasoning"]

    return reasons if reasons else ["Stable behavior"]
