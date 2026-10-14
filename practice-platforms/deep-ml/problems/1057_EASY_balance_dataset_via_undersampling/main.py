def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    if len(data) == 0:
        return []

    class_counter = {}
    for entry in data:
        class_counter[entry[1]] = class_counter.get(entry[1], 0) + 1

    min_counter = min(class_counter.values())
    seen_counter = {}
    result = []
    for entry in data:
        seen_counter[entry[1]] = seen_counter.get(entry[1], 0) + 1
        if seen_counter[entry[1]] <= min_counter:
            result.append(entry)

    return result

if __name__ == "__main__":
    data = [('a', 0), ('b', 0), ('c', 0), ('d', 1), ('e', 1)]
    result = balance_undersample(data)
    print(result) # Expected output: [('a', 0), ('b', 0), ('d', 1), ('e', 1)]