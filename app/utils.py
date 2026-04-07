def get_reason(data):
    reasons = []

    if data['SessionTime'] < 100:
        reasons.append("Low engagement")

    if data['DaysSinceLastPurchase'] > 30:
        reasons.append("High inactivity")

    if data['ReviewScore'] < 3:
        reasons.append("Low satisfaction")

    return reasons if reasons else ["Normal behavior"]