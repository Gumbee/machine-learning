from Tests import TestRuns as TestRunner
from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer
from Training.adadelta import AdaDeltaOptimizer as AdaDeltaOptimizer


def main():
    TestRunner.neural_net_test(GradientDescentOptimizer, epochs=1, batch_size=32, network_name='Small Network')
    # TestRunner.nn_optimizer_comparison(GradientDescentOptimizer, AdaDeltaOptimizer, epochs=1)
    # TestRunner.linear_regression_test(AdaDeltaOptimizer)
    # TestRunner.anomaly_test()

main()
