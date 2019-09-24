import copy, operator

def search_space_traverser(lastSolution):

    lastSolution_clone = copy.deepcopy(lastSolution)
    for cluster in lastSolution_clone[1]:
        cluster.refresh()
    for job in lastSolution_clone[0]:
        job.refresh()

    allTheSolutions = []
    recursive_search_space_traverser(0, lastSolution_clone[0], lastSolution_clone[1], allTheSolutions, 0)

    # ctr = 1
    # for sol in allTheSolutions:
    #     print("sol#", ctr)
    #     ctr += 1
    #     for e in sol[0]:
    #         print("jobID: ", e.id, " : ", e.alocated_cluster.id if e.alocated_cluster != None else "None")
    #     print("----------------------------")

    global_optimum = computePerfPopulation(allTheSolutions, lastSolution)[0]
    for e in global_optimum[0]:
        print("    jobID: ", e.id, " : ", e.alocated_cluster.id if e.alocated_cluster != None else "None")
    print("global optimum value: ", fitness(global_optimum[0], lastSolution[0]))

    return

def recursive_search_space_traverser(jobIndex, jobs, clusters, allTheSolutions, counter):

    for cluster in clusters:
        myJob = jobs[jobIndex]
        couldAllocate = cluster.allocate(myJob)
        if jobIndex < len(jobs) - 1:
            if couldAllocate:
                recursive_search_space_traverser(jobIndex + 1, jobs, clusters, allTheSolutions, counter + 1)
        else:
            if couldAllocate:
                temp = []
                temp.append(copy.deepcopy(jobs))
                temp.append(copy.deepcopy(clusters))
                allTheSolutions.append(temp)
        cluster.deAllocate(myJob)

    if jobIndex < len(jobs) - 1:
        recursive_search_space_traverser(jobIndex + 1, jobs, clusters, allTheSolutions, counter + 1)
    else:
        temp = []
        temp.append(copy.deepcopy(jobs))
        temp.append(copy.deepcopy(clusters))
        allTheSolutions.append(temp)


def computePerfPopulation(population: list, lastSolution):

    populationPerf = {}
    for individual in population:
        populationPerf[population.index(individual)] = fitness(individual[0], lastSolution[0])
        # print("pop: ", individual, " val: ", populationPerf[population.index(individual)])

    sorted_pop = []
    c = 0
    for i in sorted(populationPerf.items(), key=operator.itemgetter(1), reverse=True):
        sorted_pop.append(population[i[0]])

    return sorted_pop


def fitness(newJobs, oldJobs):

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

    return value