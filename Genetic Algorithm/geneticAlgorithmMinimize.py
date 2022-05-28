"""
    Genetic Algorithm for Multimodal Function Minimization
    Author: Giovanni Rebouças
    Date: 29 June 2021
"""
# Default Library Packages
from datetime import datetime
from math import trunc

# Third Party Packages
import numpy as np
from matplotlib import pyplot as plt

# Self-authored packages
import common


def selection(population, fitness, k=2):
    """
    Tournament selection
    """
    # Select K individuals to participate in this tournament
    tournamentPlayers = np.random.randint(0, len(population), k)

    # Select current_best as None so we don't get random fitness values from anywhere
    current_best = None

    # From the entire tournament, check and find which individual has the best fitness by checking against current best in this iteration of the tournament
    for individual in tournamentPlayers:
        # If current best is None, this is the first individual to be tested in this tournament, set him as the current best
        if current_best is None:
            current_best = individual
        # Else, check if this individual's fitness is better (lower == better) than the current best
        elif fitness[individual] < fitness[current_best]:
            current_best = individual

    # Once we know who this tournament's winner is, add him to the list of selected individuals
    return population[current_best]


def crossover(firstParent, secondParent, crossoverRate):
    """
    Crossover two parents to create two children
    """
    # Children are copies of the parents, by default
    firstChild, secondChild = firstParent, secondParent

    # Check for a crossover through recombination
    if np.random.rand() < crossoverRate:
        # Select the crossover point, making sure that it is not on the end of either string
        crossoverPoint = np.random.randint(1, len(firstParent) - 2)

        # Perform crossover
        firstChild = firstParent[:crossoverPoint] + secondParent[crossoverPoint:]
        secondChild = secondParent[:crossoverPoint] + firstParent[crossoverPoint:]

    return [firstChild, secondChild]


def mutation(chromosome, mutationRate):
    """
    Mutation operator
    """
    # Go through each gene in the chromosome
    for i in range(len(chromosome)):
        # Check for a mutation
        if np.random.rand() < mutationRate:
            # Flip the bit
            chromosome = chromosome[:i] + str(1 - int(chromosome[i])) + chromosome[i + 1:]
            break


def generatePopulation(populationSize, numGenes):
    """
    Generate the initial Generation 0 population of individuals
    """
    # List of individuals
    population = []

    # Generate a list of 100 individuals, each a bitstring of 38 genes
    rawPopulation = [np.random.randint(0, 2, numGenes).tolist() for _ in range(populationSize)]

    # Transform each individual from a list to a string and add it to our population list
    for chromosome in rawPopulation:
        aux = [str(bit) for bit in chromosome]
        finalBitString = "".join(aux)
        population.append(finalBitString)

    return population


def executeElitism(newGeneration, oldGenerationBestIndividual, oldGenerationBestIndividualFitness, objective):
    """
    Elitism operator
    """
    # Evaluate all candidates in the new population against our desired function
    newGenPreliminaryFitness = []
    for chromosome in newGeneration:
        # Split and decode our chromosome into 2 floats with 4 decimal point precision
        x1, x2 = binaryMappingToDecimal(chromosome)

        # Calculate current individual's fitness
        newGenPreliminaryFitness.append(objective(x1, x2))

    # Get min value from new generation's fitness
    newGenBestFitness = min(newGenPreliminaryFitness)

    # If the new generation's best fitness is worse than the previous generations best, conduct elitism
    if newGenBestFitness > oldGenerationBestIndividualFitness:
        # Find the worse fitness for the new generation and the individual that causes it
        newGenWorstFitness = max(newGenPreliminaryFitness)
        newGenWorstIndividual = newGenPreliminaryFitness.index(newGenWorstFitness)

        # Remove the worse individual from the new generation and add previous generation's best to new Generation
        newGeneration.pop(newGenWorstIndividual)
        newGeneration.append(oldGenerationBestIndividual)

    return newGeneration


def geneticAlgorithm(objective, numberGenerations, numGenes, populationSize, crossoverRate, mutationRate):
    """
    The "genetic algorithm" itself
    """
    # Generate initial population of pseudo-random individuals
    population = generatePopulation(populationSize, numGenes)

    # Calculate an initial fitness value to be used for the best fitness per generation. Always save the best solution for posterity
    initialX1, initialX2 = binaryMappingToDecimal(population[0])
    bestIndividual, bestFitness = population[0], objective(initialX1, initialX2)

    # List where we will store each generation's best, worst and mean fitness
    evolutionData = []

    # Enumerate a fixed number of iterations (generations)
    for generation in range(numberGenerations):
        # Evaluate all candidates in the population against our desired function
        fitness = []
        for chromosome in population:
            # Split and decode our chromosome into 2 floats with 4 decimal point precision
            x1, x2 = binaryMappingToDecimal(chromosome)
            fitness.append(objective(x1, x2))

        # Check for new best solution
        for i in range(populationSize):
            if fitness[i] < bestFitness:
                bestIndividual, bestFitness = population[i], fitness[i]
                print("Generation: " + str(generation) + " - New best chromosome f(" + population[i] + ") - Fitness: " + str(fitness[i]))

        # Build a dictionary containing the generation number and fitness values for the best, worst and mean fitness
        generationDataNode = {
            "generation": generation,
            "bestFitness": min(fitness),
            "worstFitness": max(fitness),
            "meanFitness": sum(fitness) / len(fitness)
        }

        # Append the dictionary node to our evolution data list
        evolutionData.append(generationDataNode)

        # Prepare new generation based upon our selection by tournament, crossover and mutation
        # Select the parents through tournament
        selected = [selection(population, fitness, k=2) for _ in range(len(population))]

        # After selecting the parents, start building the new generation
        children = []

        # Go through the entire population size
        for _ in range(0, populationSize):
            # Select a random pair of indices from the population
            first, second = np.random.randint(0, populationSize, 2).tolist()

            # Get parents from the selected list in pairs
            firstParent, secondParent = selected[first], selected[second]

            # Generate 2 children from each couple of parents through the crossover operator
            for child in crossover(firstParent, secondParent, crossoverRate):
                # Mutate each child
                mutation(child, mutationRate)
                # Add the child to the next generation
                children.append(child)

        # Elitism: Copy the best individual from the current generation, to the next generation
        children = executeElitism(children, bestIndividual, bestFitness, objective)

        # Set population as the new generation
        population = children

    return bestIndividual, bestFitness, evolutionData


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


def binaryMappingToDecimal(fullChromosome):
    """
    Function that decodes our bitstring mapping to a pair of real, continous values
    """
    # Our full chromosomes are presumed to be even sized (len(fullchromosome)%2 == 0)
    # But we don't assume size, get the value of half it's length to be used in properly splitting into half chromosomes and for conversion
    n = len(fullChromosome) // 2

    # Split the full lenght chromosome, into 2 half chromosomes
    # This is needed because our GA's objective function takes 2 arguments
    firstSubChromosome = fullChromosome[:n]
    secondSubChromosome = fullChromosome[n:]

    # Convert both of our subchromosomes from binary to integer
    firstChromosomeXb = int(firstSubChromosome, 2)
    secondChromosomeXb = int(secondSubChromosome, 2)

    # Calculate our conversion values
    x1 = common.XL + (((common.XU - common.XL) / ((2**n) - 1)) * firstChromosomeXb)
    x2 = common.XL + (((common.XU - common.XL) / ((2**n) - 1)) * secondChromosomeXb)

    return x1, x2


def transformPlotGenerationalData(generationalData):
    """
    Get generation data in regards to worst, best and mean fitness per generation and plot it
    """
    # Transform data from generationalData into it's respective lists
    generationList = [generation["generation"] for generation in generationData]
    worstFitnessList = [generation["worstFitness"] for generation in generationData]
    meanFitnessList = [generation["meanFitness"] for generation in generationData]
    bestFitnessList = [generation["bestFitness"] for generation in generationData]

    # Plot the graphs for the best, worst and mean fitness
    plt.plot(generationList, worstFitnessList, label="Worst")
    plt.plot(generationList, meanFitnessList, label="Mean")
    plt.plot(generationList, bestFitnessList, label="Best")

    # Set X and Y labels
    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness Value")

    # Show legend
    plt.legend()

    # Show plot
    plt.show()


def plotPopulationDistribution(population):
    """
    Plot a population's distribution
    """
    convertedPopulation = [binaryMappingToDecimal(individual) for individual in population]
    popX = [x[0] for x in convertedPopulation]
    popY = [x[1] for x in convertedPopulation]
    plt.scatter(popX, popY, label="Population")
    plt.show()


def plotMultiIterationFitness(fitnessList):
    """
    Get best fitness for each algorithm iteration and plot it. Also calculate statistical values
    """
    # Generate X axis that contains which iteration of the genetic algorithm
    numIterationList = [i + 1 for i in range(len(fitnessList))]
    fitnessList = [truncate(value, 4) for value in fitnessList]

    # Plot the graphs for the best, worst and mean fitness
    plt.plot(numIterationList, fitnessList, label="Best Fitness per iteration")

    # Prevent offset
    plt.ticklabel_format(useOffset=False)

    # Set X and Y labels
    plt.xlabel("Number of Execution")
    plt.ylabel("Best Fitness Value")

    # Calculate mean information
    mean = np.mean(bestFitnessList)
    sigma = np.std(bestFitnessList)
    variance = sigma**2

    plt.plot(numIterationList, [mean] * len(numIterationList), label="Mean")

    plt.plot(numIterationList, [mean + 3 * sigma] * len(numIterationList), 'c', label="+3*sigma and -3*sigma")
    plt.plot(numIterationList, [mean - 3 * sigma] * len(numIterationList), 'c')

    # Set some further details on the plot
    plt.figtext(.8, .92, "Mean: " + str(mean))
    plt.figtext(.8, .9, "Std. Deviation: " + str(sigma))
    plt.figtext(.8, .88, "Variance: " + str(variance))

    # Show legend
    plt.legend()

    # Show plot
    plt.show()


def plotHistogramFitness(fitnessList):
    """
    Plot the histogram of best fitness over all iterations
    """
    # Generate X axis that contains which iteration of the genetic algorithm
    fitnessList = [truncate(value, 4) for value in fitnessList]

    # Plot the histogram for best fitness
    binCount, _, _ = plt.hist(fitnessList, bins=150)

    # Prevent offset
    plt.ticklabel_format(useOffset=False)

    # Set X and Y labels
    plt.xlabel("Best Fitness Value")
    plt.ylabel("Occurences")

    # Show plot
    plt.show()


def truncate(number, precision):
    """
    Function made to truncate number up to precision decimal places
    """
    stepper = 10.0 ** precision
    return trunc(number * stepper) / stepper


if __name__ == "__main__":
    # Control variables
    singleExecution = False
    numExecutions = 100

    # Script Start Time
    scriptStartTime = datetime.now()

    # Hyperparameters

    # Note: this is the concatenated solution, each chromosome is made up of a pair of sub-chromosomes with a 19-bit lenght that make up each position in our search space
    numGenes = 38  # Number of bits (genes) that make up a solution (chromosome)
    populationSize = 500  # Population size
    numberGenerations = 100  # Number of iterations (generations) the algorithm with go at one time
    crossoverRate = 0.95  # Rate of crossover
    mutationRate = 1 / float(numGenes)  # Rate of mutation

    # Control system to collect single or multi-variable execution data
    if singleExecution:
        print("---------------------------------------------------------- Executing single iteration of the genetic algorithm ----------------------------------------------------------")
        # Call the genetic algorithm with our desired fitness function and hyperparameters
        bestIndividual, bestFitness, generationData = geneticAlgorithm(evalFunction, numberGenerations, numGenes, populationSize, crossoverRate, mutationRate)

        # Call the function that will transform the generation data and plot the fitness by generation graphs
        transformPlotGenerationalData(generationData)
    else:
        print("---------------------------------------------------------- Executing " + str(numExecutions) + " iteration(s) of the genetic algorithm ----------------------------------------------------------")
        # Best final fitness history
        bestFitnessList = []

        # Execute the desired number of times
        for i in range(numExecutions):
            print("################################ Iteration " + str(i + 1) + "/" + str(numExecutions) + " ################################")
            # Call the genetic algorithm with our desired fitness function and hyperparameters
            _, bestFitness, _ = geneticAlgorithm(evalFunction, numberGenerations, numGenes, populationSize, crossoverRate, mutationRate)
            bestFitnessList.append(bestFitness)

        # Plot and show fitness value for each generation
        plotMultiIterationFitness(bestFitnessList)
        plotHistogramFitness(bestFitnessList)

    # Calculate script execution time
    elapsedTime = datetime.now() - scriptStartTime

    # Print some script execution parameters
    print("Script Completed.")
    print("Elapsed time: " + str(elapsedTime.seconds) + " seconds")
