import math
import numpy

def softmax(scores: list[float]) -> list[float]:
    scores_np = numpy.array(scores)
    scores_npStable = scores_np - numpy.max(scores_np)
    numerator = numpy.exp(scores_npStable)
    denominator = numerator.sum()

    return numerator / denominator