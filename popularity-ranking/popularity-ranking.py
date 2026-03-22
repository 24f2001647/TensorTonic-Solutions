def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    Returns items sorted by weighted score (descending).
    """
    results = []
    for item in items:
        R = item[0]  # rating
        v = item[1]  # votes
        m = min_votes
        C = global_mean
        score = (v / (v + m)) * R + (m / (v + m)) * C
        results.append(score)
    return results