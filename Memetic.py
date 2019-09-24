import Job, Cluster, random, copy, operator
# import matplotlib.pyplot as plt


class Genetic:
    logger = None
    tmpLastSolution = None
    tmpEilteValue = -100
    debugMode = False

    def __init__(self, logger):
        self.logger = logger

    def encode(self, lastSolution):
        encodedSolution = []
        for job in lastSolution[0]:
            encodedSolution.append(job.alocated_cluster.id)
        return encodedSolution

    def decode(self, encodedSolution, jobs, clusters):
        index = 0
        for job in jobs:
            job.refresh()
        for cluster1 in clusters:
            cluster1.refresh()

        for job in jobs:
            for cluster1 in clusters:
                if cluster1.id == encodedSolution[index]:
                    cluster1.allocate(job)
                    break
            index += 1

    def find_optimum_allocation(self, jobs, clusters, debugMode, solutionOfFFD):

        self.tmpEilteValue = -100
        self.debugMode = debugMode
        populationSize = 12
        bestSample = 4
        luckyFew = 2
        numberOfChildren = 4
        chanceOfMutation = 1

        if len(jobs) < 5:
            populationSize = 4 * len(jobs)
            bestSample = populationSize / 4
            luckyFew = populationSize / 4
            numberOfChildren = 4
            chanceOfMutation = 1
        elif len(jobs) < 6:
            populationSize = (len(jobs) - len(jobs) % 3) * 4
            bestSample = populationSize / 3
            luckyFew = populationSize / 6
            numberOfChildren = 4
            chanceOfMutation = 1
        else:
            populationSize = 60
            bestSample = 18
            luckyFew = 12
            numberOfChildren = 4
            chanceOfMutation = 1

        # print("number of Generations: ", numberOfGenerations, " pop size: ", populationSize, " best sample: ", bestSample, " lucky few: ", luckyFew)
        bestSolution = self.multipleGeneration(jobs, clusters, populationSize, int(bestSample), int(luckyFew), numberOfChildren, chanceOfMutation, solutionOfFFD)

        # if self.debugMode: self.evolutionBestFitness(generations, lastSolution)

        # print("Best solution found by Genetic Alg: ", self.fitness(bestSolution, jobs, clusters))

        self.decode(bestSolution, jobs, clusters)

        return jobs, clusters

    # def evolutionBestFitness(self, historic, lastSolution):
    #     plt.axis([0, len(historic), 0, 10])
    #     plt.title("a title")
    #
    #     evolutionFitness = []
    #     for population in historic:
    #         evolutionFitness.append(
    #             self.fitness(self.getBestIndividualFromPopulation(population, lastSolution)[0], lastSolution[0]))
    #     plt.plot(evolutionFitness)
    #     plt.ylabel('fitness best individual')
    #     plt.xlabel('generation')
    #     plt.show()

    def getBestIndividualFromPopulation(self, population, lastSolution):
        return self.computePerfPopulation(population, lastSolution)[0]

    def generate_random_solution(self, numberOfJobs, numberOfClusters):
        randomSol = []
        for i in range(numberOfJobs):
            randomSol.append(random.randrange(1, numberOfClusters + 1, 1))
        return randomSol

    def generate_first_poulation(self, pop_size, jobs, clusters, solutionOfFFD):

        population = []
        i = 0
        while i < pop_size - 1:
            population.append(self.generate_random_solution(len(jobs), len(clusters)))
            i += 1

        if solutionOfFFD is not None:
            population.append(solutionOfFFD)
        else:
            population.append(self.generate_random_solution(len(jobs), len(clusters)))

        # lastSolution = []
        # lastSolution.append(jobs)
        # lastSolution.append(clusters)
        # population.append(lastSolution)

        if self.debugMode: self.logger.debug("first population:")
        for r in population:
            if self.debugMode: self.logger.debug("    value: " + str(self.fitness(r, jobs, clusters)) + " sol: " + str(r))

        # try to validate invalid random solution in a way that increases utilization
        if self.debugMode: self.logger.debug("\n\n--------> validation phase:")
        validatedPop = self.validate_population(population, jobs, clusters)

        # if self.debugMode: self.logger.debug("first validated population:")
        # for r in validatedPop:
        #     if self.debugMode: self.logger.debug("    value: " + str(self.fitness(r, jobs, clusters)) + " sol: " + str(r))

        return validatedPop

    def validate_population(self, population, jobs, clusters):
        validatedPop = []

        for individual in population:
            if self.debugMode: self.logger.debug("  before validation: " + " solution: " + str(individual) + " value: " + str(self.fitness(individual, jobs, clusters)))
            self.decode(individual, jobs, clusters)

            for cluster in clusters:

                jobCounter = len(cluster.allocatedJobs)
                while cluster.hasExceeded and jobCounter > 0:
                    jobCounter -= 1
                    could_allocate = False
                    job = cluster.removeOneJob()
                    for secondCluster in clusters:
                        if secondCluster.id != cluster.id and not secondCluster.isEmpty():
                            if secondCluster.can_allocate(job):
                                could_allocate = True
                                secondCluster.allocate(job)
                                break
                    if not could_allocate:
                        cluster.allocate(job)

                jobCounter = len(cluster.allocatedJobs)
                while cluster.hasExceeded and jobCounter > 0:
                    jobCounter -= 1
                    could_allocate = False
                    job = cluster.removeOneJob()
                    for secondCluster in clusters:
                        if secondCluster.id != cluster.id and secondCluster.isEmpty():
                            if secondCluster.can_allocate(job):
                                could_allocate = True
                                secondCluster.allocate(job)
                                break
                    if not could_allocate:
                        cluster.allocate(job)

            validatedSolution = []
            for job in jobs:
                validatedSolution.append(job.allocated_cluster_id)

            if self.debugMode: self.logger.debug("  after validation : " + " solution: " + str(validatedSolution) + " value: " + str(self.fitness(validatedSolution, jobs, clusters)) + "\n")
            validatedPop.append(validatedSolution)

        return validatedPop

    def computePerfPopulation(self, population, jobs, clusters):

        populationPerf = {}
        for individual in range(len(population)):
            populationPerf[individual] = self.fitness(population[individual], jobs, clusters)

        sorted_pop = []
        c = 0
        for i in sorted(populationPerf.items(), key=operator.itemgetter(1), reverse=True):
            sorted_pop.append(population[i[0]])

        return sorted_pop

    def selectFromPopulation(self, populationSorted, best_sample, lucky_few, jobs, clusters):
        selectedPop = []
        tmpList = []
        for i in range(best_sample):
            selectedPop.append(populationSorted[i])
            tmpList.append(populationSorted[i])

        for selected in selectedPop:
            populationSorted.remove(selected)

        if self.debugMode: self.logger.debug("selected population for parenting:")
        if self.debugMode: self.logger.debug("    best sample:")
        for r in tmpList:
            if self.debugMode: self.logger.debug("        value: " + str(self.fitness(r, jobs, clusters)))
        tmpList.clear()

        for i in range(lucky_few):
            t = random.choice(populationSorted)
            selectedPop.append(t)
            tmpList.append(t)

        if self.debugMode: self.logger.debug("    lucky few:")
        for r in tmpList:
            if self.debugMode: self.logger.debug("        value: " + str(self.fitness(r, jobs, clusters)))

        random.shuffle(selectedPop)

        return selectedPop

    def create_children(self, breeders, numberOfChildren, jobs, clusters):
        nextPopulation = []
        for i in range(int(len(breeders) / 2)):
            parent1 = breeders[i]
            parent2 = breeders[len(breeders) - 1 - i]
            if self.debugMode: self.logger.debug("  parent #1: " + str(parent1) + " value: " + str(self.fitness(parent1, jobs, clusters)))
            if self.debugMode: self.logger.debug("  parent #2: " + str(parent2) + " value: " + str(self.fitness(parent2, jobs, clusters)))

            children = self.create_child(parent1, parent2, numberOfChildren,jobs, clusters)

            if self.debugMode: self.logger.debug("  children:")
            i = 1
            for individual in children:
                nextPopulation.append(individual)
                if self.debugMode: self.logger.debug("      child #" + str(i) + ": " + str(individual) + " value: " + str(self.fitness(individual, jobs, clusters)))

            if self.debugMode: self.logger.debug("")

        return nextPopulation

    def mutateSolution(self, solution, isElite, jobs, clusters):

        index_modification = int(random.random() * len(solution))
        preAllocatedCluster = solution[index_modification]
        preMutateValue = self.fitness(solution, jobs, clusters)
        solution[index_modification] = random.randrange(1, len(clusters), 1)
        postMutateValue = self.fitness(solution, jobs, clusters)
        if isElite and preMutateValue > postMutateValue:
            solution[index_modification] = preAllocatedCluster

        return solution

    def mutatePopulation(self, population, chance_of_mutation, jobs, clusters):
        sortedPopulation = self.computePerfPopulation(population, jobs, clusters)
        mutatedPop = []
        for i in range(len(sortedPopulation)):
            if random.random() < chance_of_mutation:
                if self.debugMode: self.logger.info("   before mutation: " + " sol: " + str(sortedPopulation[i]) + " value: " + str(self.fitness(sortedPopulation[i], jobs, clusters)))
                temp = self.mutateSolution(sortedPopulation[i], i == 0 or i == 1, jobs, clusters)
                if self.debugMode: self.logger.info("   after mutation : " + " sol: " + str(temp) + " value: " + str(self.fitness(temp, jobs, clusters)) + "\n")
                mutatedPop.append(temp)
            else:
                mutatedPop.append(sortedPopulation[i])

        return mutatedPop

    def multipleGeneration(self, jobs, clusters, populationSize, best_sample, lucky_few,
                           number_of_children, chance_of_mutation, solutionOfFFD):

        first_pop = self.generate_first_poulation(populationSize, jobs, clusters, solutionOfFFD)
        best_of_all = copy.deepcopy(self.computePerfPopulation(first_pop, jobs, clusters)[0])
        best_of_all_value = self.fitness(best_of_all, jobs, clusters)
        counter = 1
        print("---> best of first Gen: ", " value ", best_of_all_value)

        previous_generation = first_pop
        no_progress_counter = 10
        while no_progress_counter >= 0:
            if self.debugMode: self.logger.debug("\n\n--------------    generation #" + str(counter) + "    --------------\n\n")
            counter += 1

            previous_generation = self.nextGeneration(previous_generation, jobs, clusters, best_sample, lucky_few, number_of_children,
                                chance_of_mutation)

            best_of_generation = self.computePerfPopulation(previous_generation, jobs, clusters)[0]
            value_of_best_of_generation = self.fitness(best_of_generation, jobs, clusters)
            if value_of_best_of_generation > best_of_all_value:
                # print("---> new best in gen #: ", counter, " sol: ", best_of_generation, " value ", value_of_best_of_generation)
                best_of_all_value = value_of_best_of_generation
                best_of_all = copy.deepcopy(best_of_generation)
                no_progress_counter = 10

            no_progress_counter -= 1

        print("Best of Gen: ", self.fitness(best_of_all, jobs, clusters))

        return best_of_all

    def nextGeneration(self, currentGeneration, jobs, clusters, best_sample, lucky_few, numberOfChildren,
                       chance_of_mutation):

        populationSorted = self.computePerfPopulation(currentGeneration, jobs, clusters)
        if self.debugMode: self.logger.debug("\n\n--------> sorted population:")
        for r in populationSorted:
            if self.debugMode: self.logger.debug("    value: " + str(self.fitness(r, jobs, clusters)) + " sol: " + str(r))

        nextBreeders = self.selectFromPopulation(populationSorted, best_sample, lucky_few, jobs, clusters)

        if self.debugMode: self.logger.debug("\n\n--------> breeding phase:")
        nextPopulation = self.create_children(nextBreeders, numberOfChildren, jobs, clusters)
        # if self.debugMode: self.logger.debug("\n\n--------> population of childs:")
        # for r in nextPopulation:
        #     if self.debugMode: self.logger.debug("    value: " + str(self.fitness(r, jobs, clusters)) + " sol: " + str(r))

        if self.debugMode: self.logger.debug("\n\n--------> mutation phase:")
        nextGeneration = self.mutatePopulation(nextPopulation, chance_of_mutation, jobs, clusters)

        if self.debugMode: self.logger.debug("\n\n--------> validation phase:")
        validatedGeneration = self.validate_population(nextGeneration, jobs, clusters)

        # if self.debugMode: self.logger.debug("\n\n--------> population after validation:")
        # for r in validatedGeneration:
        #     if self.debugMode: self.logger.debug("    value: " + str(self.fitness(r, jobs, clusters)) + " sol: " + str(r))

        return validatedGeneration

    def fitness(self, solution, jobs, clusters):

        value = 0

        isInvalid = False
        self.decode(solution, jobs, clusters)
        for cluster in clusters:
            if cluster.hasExceeded:
                isInvalid = True
                break

        # if isInvalid:
        #     value = len(clusters)*0.01
        # else:
        for cluster in clusters:
            if cluster.isEmpty():
                value += 1.3
            elif cluster.hasExceeded:
                value -= cluster.get_utilization_info()
            else:
                value += cluster.get_utilization_info() ** 2

        if value > self.tmpEilteValue and not isInvalid:
            if self.debugMode: self.logger.debug("\n\n******** new best solution value: " + str(value) + " solution: " + str(solution) + " ********\n\n")
            self.tmpEilteValue = value

        return value

    def create_child(self, solution1, solution2, numberOfChildren, jobs, clusters):

        children = []

        fitness1 = self.fitness(solution1, jobs, clusters)
        fitness2 = self.fitness(solution2, jobs, clusters)

        for i in range(numberOfChildren - 2):
            child = []
            for j in range(len(solution1)):
                r = random.random()
                if r < fitness1/(fitness1 + fitness2):
                    child.append(solution1[j])
                else:
                    child.append(solution2[j])
            children.append(child)

        children.append(solution1)
        children.append(solution2)

        return children
