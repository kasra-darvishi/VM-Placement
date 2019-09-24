import random, copy


class SimulatedAnnealing:


    def __init__(self, logger):
        self.logger = logger
        self.debugMode = False

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
        
        self.debugMode = debugMode
        opSolution = self.mainLoop(1000, 100, jobs, clusters)
        self.decode(opSolution, jobs, clusters)

    def mainLoop(self, tempreture, iterationPerTempreture, jobs, clusters):

        bestSolutionEver = None
        bestFitnessEver = -100
        initialTempreture = tempreture
        solution = self.randomValidSolution(jobs, clusters)
        fitness = -100

        while tempreture > 0:
            # print(fitness)
            if self.debugMode: self.logger.debug("\n\n\n")
            if self.debugMode: self.logger.debug("---------------------  Tempreture: " + str(tempreture) + "  ---------------------")

            for i in range(iterationPerTempreture):
                if self.debugMode: self.logger.debug("best of all: " + str(bestSolutionEver) + " " + str(bestFitnessEver) + " solution: " + str(solution) + " " + str(fitness))
                newSolution = self.getNeighbourSolution(solution, len(clusters))
                accept = False
                # move to new state or not
                if not self.isInvalid(newSolution, jobs, clusters):
                    newFitness = self.fitness(newSolution, jobs, clusters)
                    if self.debugMode: self.logger.debug("new sol: " + str(newFitness) + " recent: " + str(fitness))
                    if newFitness > fitness:
                        if self.debugMode: self.logger.debug("new solution is better")
                        accept = True
                        if newFitness > bestFitnessEver:
                            bestFitnessEver = newFitness
                            bestSolutionEver = newSolution
                            if self.debugMode: self.logger.debug("best sol fitness: " + str(newFitness))
                    else:
                        if self.debugMode: self.logger.debug("new solution is worse")
                        deviationEnergy = self.computeDeviationEnergy(tempreture, initialTempreture, len(clusters))
                        if fitness - newFitness < deviationEnergy:
                            if self.debugMode: self.logger.debug("but it was accepted. deviationEnergy: " + str(deviationEnergy) + " fitness - newFitness: " + str(fitness - newFitness))
                            accept = True

                if accept:
                    solution = newSolution
                    fitness = newFitness
                    if self.debugMode: self.logger.debug("last accepted solution: " + str(solution))
                    accept = False

            tempreture -= 5

        if self.debugMode: self.logger.debug("best of all: " + str(bestSolutionEver) + " solution: " + str(solution))

        print("best of SA: ", bestFitnessEver)
        return bestSolutionEver

    def getNeighbourSolution(self, inputSolution, numberOfservers):

        if self.debugMode: self.logger.debug("\n")
        if self.debugMode: self.logger.debug("initial solution: " + str(inputSolution))

        solution = copy.deepcopy(inputSolution)
        r = random.randrange(1, 5, 1)
        if r == 1:
            r1 = random.randrange(0, len(solution) - 5, 1)
            r2 = random.randrange(2, len(solution) - r1, 2)
            for i in range(int(r2/2)):
                tmp = solution[r1 + r2 - i]
                solution[r1 + r2 - i] = solution[r1 + i]
                solution[r1 + i] = tmp

        elif r == 2:
            r1 = random.randrange(0, len(solution) - 5, 1)
            r2 = random.randrange(2, len(solution) - r1, 1)
            r3 = random.randrange(1, r2, 1)
            tmp = []
            for i in range(r3):
                tmp.append(solution[r1 + i])
            for i in range(r2 - r3):
                solution[r1 + i] = solution[r1 + r3 + i]
            for i in range(r3):
                solution[r1 + r2 - r3 + i] = tmp[i]

        elif r == 3:
            r1 = random.randrange(0, len(solution), 1)
            r2 = random.randrange(0, len(solution), 1)
            while r2 == r1:
                r2 = random.randrange(0, len(solution), 1)
            tmp = solution[r1]
            solution[r1] = solution[r2]
            solution[r2] = tmp

        elif r == 4:
            r1 = random.randrange(0, len(solution), 1)
            r2 = random.randrange(1, numberOfservers, 1)
            solution[r1] = r2

        if self.debugMode: self.logger.debug("neighbour solution: " + str(solution))

        return solution

    def isInvalid(self, solution, jobs, clusters):
        self.decode(solution, jobs, clusters)
        for cluster in clusters:
            if cluster.hasExceeded:
                return True
        return False

    def fitness(self, solution, jobs, clusters):

        value = 0
        isInvalid = self.isInvalid(solution, jobs, clusters)

        # if isInvalid:
        #     value = len(clusters)*0.01
        # else:
        for cluster in clusters:
            if cluster.isEmpty():
                value += 1.3
            elif cluster.hasExceeded:
                value -= 1.5
            else:
                value += cluster.get_utilization_info() ** 2

        return value

    def computeDeviationEnergy(self, currentTempreture, initialTempreture, numberOfServers):
        return (1/2) * (numberOfServers * 1.3) * (currentTempreture / initialTempreture)

    def randomValidSolution(self, jobs, clusters):

        randomSol = []
        for i in range(len(jobs)):
            randomSol.append(random.randrange(1, len(clusters)+ 1, 1))

        self.decode(randomSol, jobs, clusters)

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

        return validatedSolution