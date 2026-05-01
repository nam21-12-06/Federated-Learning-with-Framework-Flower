import numpy as np

def flatten_weights(parameters):
    return np.concatenate([p.flatten() for p in parameters])


def pairwise_squared_distances(vectors):
    n = len(vectors)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((vectors[i] - vectors[j]) ** 2)
            dist[i, j] = d
            dist[j, i] = d

    return dist


def krum(vectors, f):
    n = len(vectors)

    if n < 2 * f + 3:
        raise ValueError(f"Krum requires n >= 2f + 3, got n={n}, f={f}")

    dist = pairwise_squared_distances(vectors)

    scores = []

    for i in range(n):
        dists = np.sort(dist[i])   # includes 0 (distance to itself)
        score = np.sum(dists[1 : n - f - 1])  
        scores.append(float(score))

    # tie-breaking random
    min_score = np.min(scores)
    candidates = np.where(scores == min_score)[0]
    krum_index = np.random.choice(candidates)

    return krum_index, scores