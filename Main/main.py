from Tests import TestRuns as Tester
from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer
from Training.adadelta import AdaDeltaOptimizer as AdaDeltaOptimizer


def main():
    print("\nSGD Test -------------------------------------")
    Tester.NeuralNetTest(GradientDescentOptimizer)
    print("\nAdaDelta Test -------------------------------------")
    Tester.NeuralNetTest(AdaDeltaOptimizer)


main()
