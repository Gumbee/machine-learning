from Tests import TestRuns as TestRunner
from Training.gradient_descent import GradientDescentOptimizer
from Training.adadelta import AdaDeltaOptimizer


def main():
    TestRunner.neural_net_test(AdaDeltaOptimizer, epochs=1, batch_size=32, network_name='Random Network')
    # TestRunner.nn_optimizer_comparison(GradientDescentOptimizer, AdaDeltaOptimizer, epochs=1)
    # TestRunner.linear_regression_test(AdaDeltaOptimizer)
    # TestRunner.anomaly_test()

main()
