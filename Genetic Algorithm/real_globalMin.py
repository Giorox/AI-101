"""
    Multimodal Function Testing By Exhaustion Timing Demonstrator
    Author: Giovanni Rebouças
    Date: 15 July 2021

    This script seeks to demonstrate how slow it is to test through exhaustion all 400 million combinations possible for (x,y) pairs in the [-10,10] interval with 4 decimal places precision.
    It is meant as a demonstrator and unless you're absolutely crazy, it will never finish. This seeks to demonstrate how efficient metaheuristics such as Genetic Algorithms are at search problems.
"""
# Default Library Packages
import os
from datetime import datetime

# Third Party Packages
import numpy as np


def evalFunction(x1, x2):
    """
    Function of interest - Multimodal function
    """
    # Calculate each part of each sum separately
    listValuesFirstSum = [index * np.cos(((index + 1) * x1) + index) for index in range(1, 6)]
    listValuesSecondSum = [index * np.cos(((index + 1) * x2) + index) for index in range(1, 6)]

    # Calculate the final value of f(x) by summing and multiplying the values in the list built previously
    fx = sum(listValuesFirstSum) * sum(listValuesSecondSum)

    return fx


if __name__ == "__main__":
    # Script Start Time
    scriptStartTime = datetime.now()

    # Set start and stop values
    start = -10
    stop = 10

    # Set precision
    precision = 0.0001

    # Generate all values between start and stop with the given precision
    valuesX = valuesY = np.linspace(start, stop, num=int((stop - start) / precision))

    # List of results and individuals from our evaluation function
    results = []
    individuals = []

    # Progress counter
    progress = 0

    # Call evalFunction over all values
    for x in valuesX:
        for y in valuesY:
            results.append(evalFunction(x, y))
            individuals.append((x, y))
            progress += 1

            os.system('cls')
            elapsedTime = datetime.now() - scriptStartTime
            print("Current Progress: " + str(progress) + "/400.000.000 - " + str((progress * 100) / 400000000) + "%")
            print("Elapsed Time: " + str(elapsedTime.seconds) + " seconds")

    # Get min value from results
    minima = min(results)

    # Print our result <- Never gonna happen
    print("The global minima for the evaluation function is: " + str(minima) + " - Individual: " + str(individuals[results.index(minima)]))
