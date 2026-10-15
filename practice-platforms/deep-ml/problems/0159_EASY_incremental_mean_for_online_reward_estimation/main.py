def incremental_mean(Q_prev, k, R):
    """
    Q_prev: previous mean estimate (float)
    k: number of times the action has been selected (int)
    R: new observed reward (float)
    Returns: new mean estimate (float)
    """
    return Q_prev + (R - Q_prev) / k

if __name__ == "__main__":
    Q = 0.0
    k = 1
    R = 5.0
    result = round(incremental_mean(Q, k, R), 4)
    print(result) # Expected output: 5.0