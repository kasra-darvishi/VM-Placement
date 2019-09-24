import Job, Cluster, random, copy, operator
import matplotlib.pyplot as plt

class Genetic:

    logger = None
    tmpLastSolution = None
    tmpEilteValue = -100
    debugMode = False

    def __init__(self, logger):
        self.logger = logger


    def find_optimum_allocation(self, lastSolution, debugMode):

        self.tmpEilteValue = -100
        self.tmpLastSolution = lastSolution
        self.debugMode = debugMode
        numberOfGenerations = 6
        populationSize = 12
        bestSample = 4
        luckyFew = 2
        numberOfChildren = 4
        chanceOfMutation = 1



        if len(lastSolution[0]) < 5:
            numberOfGenerations = 10
            populationSize = 4 * len(lastSolution[0])
            bestSample = populationSize / 4
            luckyFew = populationSize / 4
            numberOfChildren = 4
            chanceOfMutation = 1
        elif len(lastSolution[0]) < 6:
            numberOfGenerations = 8
            populationSize = (len(lastSolution[0]) - len(lastSolution[0]) % 3) * 4
            bestSample = populationSize / 3
            luckyFew = populationSize / 6
            numberOfChildren = 4
            chanceOfMutation = 1
        else:
            numberOfGenerations = 8
            populationSize = 20
            bestSample = 6
            luckyFew = 4
            numberOfChildren = 4
            chanceOfMutation = 1

        # print("number of Generations: ", numberOfGenerations, " pop size: ", populationSize, " best sample: ", bestSample, " lucky few: ", luckyFew)

        generations = self.multipleGeneration(numberOfGenerations, lastSolution, populationSize, int(bestSample), int(luckyFew), numberOfChildren, chanceOfMutation)
        bestSolution = self.getBestIndividualFromPopulation(generations[numberOfGenerations - 1], lastSolution)
        if self.debugMode: self.evolutionBestFitness(generations, lastSolution)

        print("Best solution found by Genetic Alg: ", self.fitness(bestSolution[0], lastSolution[0]))

        return bestSolution


    def evolutionBestFitness(self, historic, lastSolution):
        plt.axis([0, len(historic), 0, 10])
        plt.title("a title")

        evolutionFitness = []
        for population in historic:
            evolutionFitness.append(self.fitness(self.getBestIndividualFromPopulation(population, lastSolution)[0], lastSolution[0]))
        plt.plot(evolutionFitness)
        plt.ylabel('fitness best individual')
        plt.xlabel('generation')
        plt.show()


    def getBestIndividualFromPopulation(self, population, lastSolution):
        return self.computePerfPopulation(population, lastSolution)[0]


    def generate_random_solution(self, jobs, clusters):

        for i in jobs:
            i.refresh()
        random.shuffle(jobs)

        for i in clusters:
            i.refresh()
        random.shuffle(clusters)

        for job in jobs:
            for clust in clusters:
                if clust.allocate(job):
                    break
            random.shuffle(clusters)

        newSolution = []
        newSolution.append(jobs)
        newSolution.append(clusters)

        return newSolution


    def generate_first_poulation(self, pop_size, jobs, clusters):

        population = []
        i = 0
        while i < pop_size - 1:
            tmpList = self.generate_random_solution(copy.deepcopy(jobs), copy.deepcopy(clusters))
            population.append(tmpList)
            i += 1

        lastSolution = []
        lastSolution.append(jobs)
        lastSolution.append(clusters)
        population.append(lastSolution)

        if self.debugMode: self.logger.debug("first population:")
        for r in population:
            if self.debugMode: self.logger.debug("    value: " + str(self.fitness(r[0], self.tmpLastSolution[0])))

        return population


    def computePerfPopulation(self, population: list, lastSolution):

        populationPerf = {}
        for individual in population:
            populationPerf[population.index(individual)] = self.fitness(individual[0], lastSolution[0])
            # print("pop: ", individual, " val: ", populationPerf[population.index(individual)])

        sorted_pop = []
        c = 0
        for i in sorted(populationPerf.items(), key=operator.itemgetter(1), reverse=True):
            sorted_pop.append(copy.deepcopy(population[i[0]]))

        return sorted_pop


    def selectFromPopulation(self, populationSorted, best_sample, lucky_few):
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
            if self.debugMode: self.logger.debug("        value: " + str(self.fitness(r[0], self.tmpLastSolution[0])))
        tmpList.clear()

        for i in range(lucky_few):
            t = copy.deepcopy(random.choice(populationSorted))
            selectedPop.append(t)
            tmpList.append(t)

        if self.debugMode: self.logger.debug("    lucky few:")
        for r in tmpList:
            if self.debugMode: self.logger.debug("        value: " + str(self.fitness(r[0], self.tmpLastSolution[0])))

        random.shuffle(selectedPop)

        return selectedPop


    def create_children(self, breeders, numberOfChildren, lastSolution):
        nextPopulation = []
        for i in range(int(len(breeders) / 2)):
            for individual in self.create_child(breeders[i][0], breeders[i][1], breeders[len(breeders) - 1 - i][0],
                                           breeders[len(breeders) - 1 - i][1], numberOfChildren, lastSolution):
                nextPopulation.append(individual)

        return nextPopulation

    def mutateSolution(self, solution, isElite):

        index_modification = int(random.random() * len(solution[0]))

        preMutateValue = self.fitness(solution[0], self.tmpLastSolution[0])

        tempSolution = [copy.deepcopy(solution[0]), copy.deepcopy(solution[1])]
        jobToBeModified = tempSolution[0][index_modification]
        if self.debugMode: self.logger.debug("jobToBeModified id " + str(jobToBeModified.id) + " " + str(jobToBeModified.needed_core_num) + " " + str(jobToBeModified.needed_ram))

        if jobToBeModified.alocated_cluster == None:
            for clstr1 in tempSolution[1]:
                if self.debugMode: self.logger.debug(
                    "before change cluster id " + str(clstr1.id) + " " + str(clstr1.core_num) + " " + str(clstr1.ram))
                if clstr1.allocate(jobToBeModified):
                    postMutateValue = self.fitness(tempSolution[0], self.tmpLastSolution[0])
                    if self.debugMode: self.logger.debug("successfuly mutated 111 ...")
                    if self.debugMode: self.logger.debug(
                        "value before, after: " + str(preMutateValue) + ", " + str(postMutateValue))
                    if self.debugMode: self.logger.debug("changed cluster id " + str(clstr1.id) + " " + str(clstr1.core_num) + " " + str(clstr1.ram))
                    return tempSolution
            for job in tempSolution[0]:
                if job.alocated_cluster != None:
                    for clstr2 in tempSolution[1]:
                        if clstr2.id == job.alocated_cluster.id:
                            if clstr2.contains_job(job):
                                clstr2.deAllocate(job)
                                if clstr2.allocate(jobToBeModified):
                                    postMutateValue = self.fitness(tempSolution[0], self.tmpLastSolution[0])
                                    if self.debugMode: self.logger.debug("successfuly mutated 222 ...")
                                    if self.debugMode: self.logger.debug(
                                        "value before, after: " + str(preMutateValue) + ", " + str(postMutateValue))
                                    if isElite:
                                        if postMutateValue > preMutateValue:
                                            if self.debugMode: self.logger.debug("  an elite mutated!")
                                            return tempSolution
                                        else:
                                            if self.debugMode: self.logger.debug(
                                                "  ignored the mutation because of elitism!")
                                            return solution
                                    else:
                                        return tempSolution
                                else:
                                    clstr2.deAllocate(jobToBeModified)
                                    clstr2.allocate(job)
                            break
        else:
            for job in tempSolution[0]:
                if job.alocated_cluster != None:
                    if job.alocated_cluster.id != jobToBeModified.alocated_cluster.id:
                        for clstr1 in tempSolution[1]:
                            if clstr1.id == jobToBeModified.alocated_cluster.id:
                                for clstr2 in tempSolution[1]:
                                    if clstr2.id == job.alocated_cluster.id:
                                        if clstr1.contains_job(jobToBeModified) and clstr2.contains_job(job):
                                            clstr1.deAllocate(jobToBeModified)
                                            clstr2.deAllocate(job)
                                            if clstr1.allocate(job) and clstr2.allocate(jobToBeModified):
                                                postMutateValue = self.fitness(tempSolution[0], self.tmpLastSolution[0])
                                                if self.debugMode: self.logger.debug("successfuly mutated 333 ...")
                                                if self.debugMode: self.logger.debug(
                                                    "value before, after: " + str(preMutateValue) + ", " + str(
                                                        postMutateValue))
                                                if isElite:
                                                    if postMutateValue > preMutateValue:
                                                        if self.debugMode: self.logger.debug("  an elite mutated!")
                                                        return tempSolution
                                                    else:
                                                        if self.debugMode: self.logger.debug(
                                                            "  ignored the mutation because of elitism!")
                                                        return solution
                                                else:
                                                    return tempSolution
                                            else:
                                                clstr1.deAllocate(job)
                                                clstr2.deAllocate(jobToBeModified)
                                                clstr1.allocate(jobToBeModified)
                                                clstr2.allocate(job)
                                        break
                                break
                else:
                    for clstr1 in tempSolution[1]:
                        if clstr1.id == jobToBeModified.alocated_cluster.id:
                            if clstr1.contains_job(jobToBeModified):
                                clstr1.deAllocate(jobToBeModified)
                                if clstr1.allocate(job):
                                    postMutateValue = self.fitness(tempSolution[0], self.tmpLastSolution[0])
                                    if self.debugMode: self.logger.debug("successfuly mutated 444 ...")
                                    if self.debugMode: self.logger.debug(
                                        "value before, after: " + str(preMutateValue) + ", " + str(postMutateValue))
                                    if isElite:
                                        if postMutateValue > preMutateValue:
                                            if self.debugMode: self.logger.debug("  an elite mutated!")
                                            return tempSolution
                                        else:
                                            if self.debugMode: self.logger.debug(
                                                "  ignored the mutation because of elitism!")
                                            return solution
                                    else:
                                        return tempSolution
                                else:
                                    clstr1.deAllocate(job)
                                    clstr1.allocate(jobToBeModified)
                            break

        if self.debugMode: self.logger.debug("could not mutate...")

        return solution


    def mutatePopulation(self, population, chance_of_mutation, lastSolution):
        sortedPopulation = self.computePerfPopulation(population, lastSolution)
        mutatedPop = []
        for i in range(len(sortedPopulation)):
            if random.random() < chance_of_mutation:
                tmpp = self.mutateSolution(sortedPopulation[i], i == 0 or i == 1)
                if self.debugMode: self.logger.info("---> " + str(self.fitness(tmpp[0], self.tmpLastSolution[0])) + " qablan: " + str(self.fitness(sortedPopulation[i][0], self.tmpLastSolution[0])))
                mutatedPop.append(copy.deepcopy(tmpp))
            else:
                mutatedPop.append(sortedPopulation[i])

        if self.debugMode: self.logger.info("\n\n\n************************")
        for a in mutatedPop:
            if self.debugMode: self.logger.info(str(self.fitness(a[0], self.tmpLastSolution[0])))
        if self.debugMode: self.logger.info("\n************************\n\n\n")

        return mutatedPop


    def multipleGeneration(self, numOfGenerations, lastSolution, populationSize, best_sample, lucky_few, number_of_children,
                           chance_of_mutation):
        generations = []
        generations.append(self.generate_first_poulation(populationSize, lastSolution[0], lastSolution[1]))
        counter = 1
        for i in range(numOfGenerations - 1):
            if self.debugMode: self.logger.debug("\n\n-------    generation #" + str(counter) + "    -------\n\n")
            counter += 1
            generations.append(self.nextGeneration(generations[i], lastSolution, best_sample, lucky_few, number_of_children,
                                              chance_of_mutation))

        return generations


    def nextGeneration(self, currentGeneration, lastSolution, best_sample, lucky_few, numberOfChildren, chance_of_mutation):

        populationSorted = self.computePerfPopulation(currentGeneration, lastSolution)
        if self.debugMode: self.logger.debug("sorted population:")
        for r in populationSorted:
            if self.debugMode: self.logger.debug("    value: " + str(self.fitness(r[0], lastSolution[0])))

        nextBreeders = self.selectFromPopulation(populationSorted, best_sample, lucky_few)

        nextPopulation = self.create_children(nextBreeders, numberOfChildren, lastSolution)
        if self.debugMode: self.logger.debug("population of childs:")
        for r in nextPopulation:
            if self.debugMode: self.logger.debug("    value: " + str(self.fitness(r[0], lastSolution[0])))

        nextGeneration = self.mutatePopulation(nextPopulation, chance_of_mutation, lastSolution)
        if self.debugMode: self.logger.debug("population after mutation:")
        for r in nextGeneration:
            if self.debugMode: self.logger.debug("    value: " + str(self.fitness(r[0], lastSolution[0])))

        return nextGeneration


    def fitness(self, newJobs, oldJobs):

        value = 0

        for newJob in newJobs:
            for oldJob in oldJobs:
                if oldJob.id == newJob.id:
                    # if job was allocated but now its not
                    if newJob.alocated_cluster == None and oldJob.alocated_cluster != None:
                        value -= 6
                        # print("\njob #", newJob.id, " got out")
                    # if job's cluster has changed
                    elif newJob.alocated_cluster != None and oldJob.alocated_cluster != None and newJob.alocated_cluster.id != oldJob.alocated_cluster.id:
                        value -= 1
                        # print("\njob #", newJob.id, " cluster changed from ", oldJob.alocated_cluster.id, " to ", newJob.alocated_cluster.id)
                    # if job is newly allocated to a cluster
                    elif newJob.alocated_cluster != None and oldJob.alocated_cluster == None:
                        value += 4
                        # print("\njob #", newJob.id, " now is allocated")
                    break

        if value > self.tmpEilteValue:
            if self.debugMode: self.logger.debug("\n\nnew elite found: " + str(self.tmpEilteValue) + " " + str(value))
            self.tmpEilteValue = value
            if self.debugMode: self.logger.debug("\n    previous allocation:")
            for job in oldJobs:
                if self.debugMode: self.logger.debug("  " + str(job.id) + " : " + (str(job.alocated_cluster.id) if job.alocated_cluster != None else str(-1)))
            if self.debugMode: self.logger.debug("\n    new allocation:")
            for job in newJobs:
                if self.debugMode: self.logger.debug("  " + str(job.id) + " : " + (str(job.alocated_cluster.id) if job.alocated_cluster != None else str(-1)))

        return value


    def create_child(self, jobs1, clusters1, jobs2, clusters2, numberOfChildren, lastSolution):

        jobToClusterMap1 = {}
        jobToClusterMap2 = {}

        childClusters = copy.deepcopy(clusters1)
        childJobs = copy.deepcopy(jobs1)
        for c in childClusters:
            c.refresh()
        for j in childJobs:
            j.refresh()

        for job1 in jobs1:
            jobToClusterMap1[job1.id] = job1.alocated_cluster.id if job1.alocated_cluster != None else -1

        for job2 in jobs2:
            jobToClusterMap2[job2.id] = job2.alocated_cluster.id if job2.alocated_cluster != None else -1

        results = []

        parentsAreTheSame = True
        for j in jobToClusterMap1.keys():
            if jobToClusterMap1[j] != jobToClusterMap2[j]:
                parentsAreTheSame = False
                break

        if parentsAreTheSame:
            for i in range(numberOfChildren):
                tmp = []
                tmp.append(copy.deepcopy(jobs1))
                tmp.append(copy.deepcopy(clusters1))
                results.append(tmp)
        else:
            firstJobID = 0
            for i in jobToClusterMap1.keys():
                firstJobID = i
                break

            self.recursive_child_maker(jobToClusterMap1, jobToClusterMap2, firstJobID, 0, childJobs, childClusters, results, "&")

            if len(results) < numberOfChildren:
                b = True
                for i in range(numberOfChildren - len(results)):
                    tmp = []
                    if b:
                        tmp.append(copy.deepcopy(jobs1))
                        tmp.append(copy.deepcopy(clusters1))
                    else:
                        tmp.append(copy.deepcopy(jobs2))
                        tmp.append(copy.deepcopy(clusters2))
                    results.append(tmp)
                    b = not b

        results.append((jobs1, clusters1))
        results.append((jobs2, clusters2))
        results = self.computePerfPopulation(results, lastSolution)

        # self.root2.debug("parents of children: (are they the same?) " + str(parentsAreTheSame))
        # self.root2.debug("    parent 1 value: " + str(self.fitness(jobs1, self.tmpLastSolution[0])))
        # self.root2.debug("    parent 2 value: " + str(self.fitness(jobs2, self.tmpLastSolution[0])))
        # self.root2.debug("all of children: ")
        # for r in results:
        #     self.root2.debug("    value: " + str(self.fitness(r[0], self.tmpLastSolution[0])))

        finalResults = []
        finalResults.append(results[0])
        results.remove(finalResults[0])
        finalResults.append(results[0])
        results.remove(finalResults[1])

        # self.root2.debug("Elite children: ")
        # for r in finalResults:
        #     self.root2.debug("    value: " + str(self.fitness(r[0], self.tmpLastSolution[0])))

        random.shuffle(results)
        for i in range(numberOfChildren - 2):
            finalResults.append(results[i])

        # self.root2.debug("final selected children: ")
        # for r in finalResults:
        #     self.root2.debug("    value: " + str(self.fitness(r[0], self.tmpLastSolution[0])))

        return finalResults


    def recursive_child_maker(self, allocationInfo1, allocationInfo2, nextJobID, counter, childJobs, childClusters, results, debugStr):

        if allocationInfo1[nextJobID] == allocationInfo2[nextJobID]:

            childClusters_clone = copy.deepcopy(childClusters)
            childJobs_clone = copy.deepcopy(childJobs)
            if allocationInfo1[nextJobID] != -1:
                for clstr in childClusters_clone:
                    if clstr.id == allocationInfo1[nextJobID]:
                        for jb in childJobs_clone:
                            if jb.id == nextJobID:
                                if clstr.allocate(jb):
                                    i = counter + 1
                                    if i == len(allocationInfo1):
                                        tmp = []
                                        tmp.append(childJobs_clone)
                                        tmp.append(childClusters_clone)
                                        results.append(tmp)
                                        # print("#1", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                                        debugStr += "1"
                                        # self.root2.debug(debugStr)

                                        return
                                    else:
                                        # print("#2", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                                        debugStr += "2"
                                        b = False
                                        tmpId = -1
                                        for idx in allocationInfo1.keys():
                                            if b:
                                                tmpId = idx
                                                break
                                            if idx == nextJobID:
                                                b = True
                                        self.recursive_child_maker(allocationInfo1, allocationInfo2, tmpId, i,
                                                              childJobs_clone, childClusters_clone, results, debugStr)
                                        return
                                break
                        break
            else:
                i = counter + 1
                if i == len(allocationInfo1):
                    tmp = []
                    tmp.append(childJobs_clone)
                    tmp.append(childClusters_clone)
                    results.append(tmp)
                    # print("#3", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                    debugStr += "3"
                    # self.root2.debug(debugStr)

                    return
                else:
                    # print("#4", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                    debugStr += "4"

                    b = False
                    tmpId = -1
                    for idx in allocationInfo1.keys():
                        if b:
                            tmpId = idx
                            break
                        if idx == nextJobID:
                            b = True
                    self.recursive_child_maker(allocationInfo1, allocationInfo2, tmpId, i,
                                          childJobs_clone, childClusters_clone, results, debugStr)
                    return

        else:

            childClusters_clone = copy.deepcopy(childClusters)
            childJobs_clone = copy.deepcopy(childJobs)
            if allocationInfo1[nextJobID] != -1:

                for clstr in childClusters_clone:
                    if clstr.id == allocationInfo1[nextJobID]:
                        for jb in childJobs_clone:
                            if jb.id == nextJobID:
                                if clstr.allocate(jb):
                                    i = counter + 1
                                    if i == len(allocationInfo1):
                                        tmp = []
                                        tmp.append(childJobs_clone)
                                        tmp.append(childClusters_clone)
                                        results.append(tmp)
                                        # print("#5", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                                        debugStr2 = copy.deepcopy(debugStr)
                                        debugStr2 += "5"
                                        # self.root2.debug(debugStr2)

                                    else:
                                        # print("#6", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                                        debugStr2 = copy.deepcopy(debugStr)
                                        debugStr2 += "6"


                                        b = False
                                        tmpId = -1
                                        for idx in allocationInfo1.keys():
                                            if b:
                                                tmpId = idx
                                                break
                                            if idx == nextJobID:
                                                b = True
                                        self.recursive_child_maker(allocationInfo1, allocationInfo2, tmpId, i,
                                                              childJobs_clone, childClusters_clone, results, debugStr2)
                                break
                        break
            else:

                i = counter + 1
                if i == len(allocationInfo1):
                    tmp = []
                    tmp.append(childJobs_clone)
                    tmp.append(childClusters_clone)
                    results.append(tmp)
                    # print("#7", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                    debugStr2 = copy.deepcopy(debugStr)
                    debugStr2 += "7"
                    # self.root2.debug(debugStr2)


                else:
                    b = False
                    # print("#8", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                    debugStr2 = copy.deepcopy(debugStr)
                    debugStr2 += "8"

                    tmpId = -1
                    for idx in allocationInfo1.keys():
                        if b:
                            tmpId = idx
                            break
                        if idx == nextJobID:
                            b = True
                    self.recursive_child_maker(allocationInfo1, allocationInfo2, tmpId, i,
                                          childJobs_clone, childClusters_clone, results, debugStr2)

            childClusters_clone2 = copy.deepcopy(childClusters)
            childJobs_clone2 = copy.deepcopy(childJobs)
            if allocationInfo2[nextJobID] != -1:
                for clstr in childClusters_clone2:
                    if clstr.id == allocationInfo2[nextJobID]:
                        for jb in childJobs_clone2:
                            if jb.id == nextJobID:
                                if clstr.allocate(jb):
                                    i = counter + 1
                                    if i == len(allocationInfo2):
                                        tmp = []
                                        tmp.append(childJobs_clone2)
                                        tmp.append(childClusters_clone2)
                                        results.append(tmp)
                                        # print("#9", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                                        # self.root2.debug("#9 " + str(nextJobID) + " - " + str(counter) + " cls1: " + str(
                                        #     allocationInfo1[nextJobID]) + " cls2: " + str(allocationInfo2[nextJobID]))
                                        debugStr2 = copy.deepcopy(debugStr)
                                        debugStr2 += "9"
                                        # self.root2.debug(debugStr2)

                                        return
                                    else:
                                        # print("#10", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                                        debugStr2 = copy.deepcopy(debugStr)
                                        debugStr2 += "'10'"

                                        b = False
                                        tmpId = -1
                                        for idx in allocationInfo1.keys():
                                            if b:
                                                tmpId = idx
                                                break
                                            if idx == nextJobID:
                                                b = True
                                        self.recursive_child_maker(allocationInfo1, allocationInfo2, tmpId, i,
                                                              childJobs_clone2, childClusters_clone2, results, debugStr2)
                                        return
                                break
                        break
            else:
                i = counter + 1
                if i == len(allocationInfo1):
                    tmp = []
                    tmp.append(childJobs_clone2)
                    tmp.append(childClusters_clone2)
                    results.append(tmp)
                    # print("#11", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                    # self.root2.debug("#11 " + str(nextJobID) + " - " + str(counter) + " cls1: " + str(allocationInfo1[nextJobID]) + " cls2: " + str(allocationInfo2[nextJobID]))
                    debugStr2 = copy.deepcopy(debugStr)
                    debugStr2 += "'11'"
                    # self.root2.debug(debugStr2)

                    return
                else:
                    # print("#12", nextJobID, " - ", counter, " cls1: ", allocationInfo1[nextJobID], " cls2: ", allocationInfo2[nextJobID])
                    debugStr2 = copy.deepcopy(debugStr)
                    debugStr2 += "'12'"

                    b = False
                    tmpId = -1
                    for idx in allocationInfo1.keys():
                        if b:
                            tmpId = idx
                            break
                        if idx == nextJobID:
                            b = True
                    self.recursive_child_maker(allocationInfo1, allocationInfo2, tmpId, i,
                                          childJobs_clone2, childClusters_clone2, results, debugStr2)
                    return




