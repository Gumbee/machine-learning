from Tests import TestRuns as Tester
from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer
from Training.adadelta import AdaDeltaOptimizer as AdaDeltaOptimizer


def main():
    Tester.nn_optimizer_comparison(GradientDescentOptimizer, AdaDeltaOptimizer, epochs=50)


main()
