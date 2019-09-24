import operator
import random

class GrayWolf:
    logger = None
    tmpLastSolution = None
    tmpEilteValue = -100
    debugMode = False

    def __init__(self, logger):
        self.logger = logger

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

    def find_optimum_allocation(self, jobs, clusters, debugMode, solutionOfFFD):


        bestSolution = self.gwo(50, jobs, clusters)
        print("best of GWO: ", self.fitness(bestSolution, jobs, clusters))
        self.decode(bestSolution, jobs, clusters)
        self.debugMode = debugMode

        return [jobs, clusters]

    def gwo(self, numberOfwolves, jobs, clusters):

        # intial random allocations as the initial position of the wolves
        wolvesList = self.intialWolves(numberOfwolves, jobs, clusters)

        if self.debugMode: self.logger.debug("\n\ninitial wolves: ")
        for wol in wolvesList:
            if self.debugMode: self.logger.debug(str(wol) + " " + str(self.fitness(wol, jobs, clusters)))

        a = 2

        while a >= 0:
            if self.debugMode: self.logger.debug("\n\n----------------           a: " + str(a))
            counter = 1
            eliteWolves = []
            wolvesList = self.sortWolves(wolvesList, jobs, clusters)
            for wolf in wolvesList:

                # set aside 3 wolves that have highest scores
                if counter < 4:
                    eliteWolves.append(wolf)

                # guide other wolves by the 3 dominant wolves
                else:
                    if self.debugMode: self.logger.debug("\n\n")
                    if self.debugMode: self.logger.debug("wolf: " + str(wolf) + " " + str(self.fitness(wolf, jobs, clusters)))
                    if counter == 4:
                        if self.debugMode: self.logger.debug("\n\n")
                        if self.debugMode: self.logger.debug("elite wolves: ")
                        for wol in eliteWolves:
                            if self.debugMode: self.logger.debug(str(wol) + " " + str(self.fitness(wol, jobs, clusters)))
                        if self.debugMode: self.logger.debug("\n\n")

                    # guide Wolf To Elites
                    x1 = self.guidByElite(wolf, eliteWolves[0], a, len(clusters))
                    x2 = self.guidByElite(wolf, eliteWolves[1], a, len(clusters))
                    x3 = self.guidByElite(wolf, eliteWolves[2], a, len(clusters))

                    # set the final position of the wolf to a combination of where 3 dominant wolves guided non-dominant wolf to
                    for i in range(len(wolf)):
                        rand = random.random()
                        wolf[i] = x1[i] if rand < 0.334 else (x2[i] if rand < 0.667 else x3[i])

                    if self.debugMode: self.logger.debug("final wolf: " + str(wolf) + " " + str(self.fitness(wolf, jobs, clusters)))

                counter += 1

            a -= 0.005


        return self.sortWolves(wolvesList, jobs, clusters)[0]

    def guidByElite(self, wolf, eliteWolf, a, numberOfClusters):

        numberOfDifferentAllocations = 0
        for i in range(len(wolf)):
            if wolf[i] != eliteWolf[i]:
                numberOfDifferentAllocations += 1

        # percentage of difference is 'D' or dictance
        capital_d = numberOfDifferentAllocations/len(wolf)

        if self.debugMode: self.logger.debug("wolf and elite: " + str(wolf) + " - " + str(eliteWolf))
        if self.debugMode: self.logger.debug("numberOfDifferentAllocations and capital_d: " + str(numberOfDifferentAllocations) + " - " + str(capital_d))


        capital_a = a * random.random()
        # set the distance that wolf should have from elite wolf
        finalDifference = capital_a * capital_d
        if finalDifference > 1:
            finalDifference = 1

        if self.debugMode: self.logger.debug("capital_a " + str(capital_a) + " finalDiff: " + str(finalDifference))

        resultWolf = []
        for i in range(len(wolf)):
            randNum = random.random()
            # (1 - finalDifference) is the prcentage of the similarity that the wolf should have with the dominant wolf
            if randNum < 1 - finalDifference:
                # allocate this VM to the server that the dominant wolf has allocated it to
                resultWolf.append(eliteWolf[i])
                if self.debugMode: self.logger.debug("index " + str(i) + " from elite: " + str(resultWolf[i]))
            else:
                # choose a random cluster for this job
                resultWolf.append(random.randrange(1, numberOfClusters+1, 1))
                if self.debugMode: self.logger.debug("index " + str(i) + " from random: " + str(resultWolf[i]))

        return resultWolf

    def generate_random_solution(self, numberOfJobs, numberOfClusters):
        randomSol = []
        for i in range(numberOfJobs):
            randomSol.append(random.randrange(1, numberOfClusters + 1, 1))
        return randomSol

    def intialWolves(self, numberOfWolves, jobs, clusters):
        initialWolves = []
        for i in range(numberOfWolves):
            initialWolves.append(self.generate_random_solution(len(jobs), len(clusters)))
        return initialWolves

    def sortWolves(self, wolves, jobs, clusters):

        populationPerf = {}
        for index in range(len(wolves)):
            populationPerf[index] = self.fitness(wolves[index], jobs, clusters)
        sorted_wolves = []
        c = 0
        for i in sorted(populationPerf.items(), key=operator.itemgetter(1), reverse=True):
            sorted_wolves.append(wolves[i[0]])

        return sorted_wolves

