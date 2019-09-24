import sqlite3, operator
import init_db, queue, random, Job, Cluster, genetic, copy
import BruteForce
import logging
import logging.handlers
import os, time, Memetic

from GrayWolf import GrayWolf
from SimulatedAnnealing import SimulatedAnnealing


def decode(encodedSolution, jobs, clusters):
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


def getKey(cluster):
    return cluster.init_core_num


def getKey2(job):
    return job.needed_core_num


def fitness(clusters):
    value = 0
    isInvalid = False
    for cluster in clusters:
        if cluster.hasExceeded:
            isInvalid = True
            break
    for cluster in clusters:
        if cluster.isEmpty():
            value += 1.3
        elif cluster.hasExceeded:
            value -= cluster.get_utilization_info()
        else:
            value += cluster.get_utilization_info() ** 2
    return value


def report(newSolution, previousSolution, logger, cycleNumber):
    logger.info("\n\n\n------------------      Cycle #" + str(cycleNumber) + "      ------------------")
    for newJob in newSolution[0]:
        for oldJob in previousSolution[0]:
            if oldJob.id == newJob.id:
                # if job was allocated but now its not
                if newJob.alocated_cluster == None and oldJob.alocated_cluster != None:
                    logger.info("   job #" + str(newJob.id) + " got out of its cluster " + str(
                        oldJob.alocated_cluster.id) + "\n")
                # if job's cluster has changed
                elif newJob.alocated_cluster != None and oldJob.alocated_cluster != None and newJob.alocated_cluster.id != oldJob.alocated_cluster.id:
                    logger.info("   job #" + str(newJob.id) + " cluster changed from " + str(
                        oldJob.alocated_cluster.id) + " to " + str(newJob.alocated_cluster.id) + "\n")
                # if job is newly allocated to a cluster
                elif newJob.alocated_cluster != None and oldJob.alocated_cluster == None:
                    logger.info(
                        "   job #" + str(newJob.id) + " newly allocated to " + str(newJob.alocated_cluster.id) + "\n")
                break


def getClustersInfo(clusterList, logger):
    maxCoreNum = 0
    maxRam = 0
    conn = sqlite3.connect('info.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Cluster ")
    rows = cursor.fetchall()
    logger.info("\n\nClusters: ")
    for i in range(len(rows)):
        print(rows[i][5], " ", rows[i][6])
        maxCoreNum = rows[i][5] if rows[i][5] > maxCoreNum else maxCoreNum
        maxRam = rows[i][6] if rows[i][6] > maxRam else maxRam
        clusterList.append(Cluster.Cluster(rows[i][0], rows[i][5], rows[i][6]))
        logger.info(
            "   Cluster id: " + str(rows[i][0]) + " number of cores: " + str(rows[i][5]) + " ram: " + str(rows[i][6]))

    return [maxCoreNum, maxRam]


def fisrtFit(job, clusterList):
    for clstr in clusterList:
        if clstr.can_allocate(job):
            clstr.allocate(job)
            # logger.info("\n\n\n------------------      Cycle #" + str(cycleNumber) + "      ------------------")
            # logger.info("   job #" + str(job.id) + " newly allocated to " + str(clstr.id) + "\n")
            return True
    return False


def utilization_info(clusters, logger):
    nmbr1 = 0
    nmbr2 = 0
    nmbr3 = 0
    util = 0

    for cluster in clusters:
        if cluster.hasExceeded:
            logger.info("\n\n\n")
            logger.info("server #" + str(cluster.id) + " is over allocated")
        else:
            if cluster.isEmpty():
                logger.info("\n")
                logger.info("server #" + str(cluster.id) + " is empty")
                nmbr1 += 1
            else:
                tmp = cluster.get_utilization_info()
                nmbr2 += 1
                util += tmp
                logger.info("\n")
                logger.info("server #" + str(cluster.id) + " is " + str(tmp) + "% utilized")
                for job in cluster.allocatedJobs:
                    logger.info("job id: " + str(job.id))
                    nmbr3 += 1

    logger.info("\n")
    logger.info("-------------------------------------------------------------------------------")
    logger.info("number of empty servers: " + str(nmbr1))
    logger.info("average utilization: " + str(util / nmbr2))
    logger.info("number of allocated jobs: " + str(nmbr3))
    logger.info("-------------------------------------------------------------------------------")

def runAlgorithms(jobs, clusters, root):
    mid = time.time()
    sortedList = sorted(clusters, key=getKey, reverse=True)
    for newJob in sorted(jobs, key=getKey2, reverse=True):
        fisrtFit(newJob, sortedList)
    utilization_info(sortedList, root)
    print("value of FFD: ", fitness(sortedList))
    end = time.time()
    print("elapsed time FFD: ", end - mid)

    mid = time.time()
    gen = Memetic.Genetic(root)
    gen.find_optimum_allocation(jobs, clusters, False, None)
    root.debug("Genetic:")
    utilization_info(clusters, root)
    end = time.time()
    print("elapsed time genetic: ", end - mid)

    mid = time.time()
    gwo = GrayWolf(root)
    gwo.find_optimum_allocation(jobs, clusters, False, None)
    root.debug("Gray Wolf:")
    utilization_info(clusters, root)
    end = time.time()
    print("elapsed time GWO: ", end - mid)

    mid = time.time()
    sa = SimulatedAnnealing(root)
    sa.find_optimum_allocation(jobs, clusters, False, None)
    root.debug("Simulated annealing:")
    utilization_info(clusters, root)
    end = time.time()
    print("elapsed time SA: ", end - mid)


handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "/home/kasra/Desktop/Report.log"), 'w')
formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
root.addHandler(handler)



clusters = []
jobs = []

for i in range(30):
    clusters.append(Cluster.Cluster(i + 1, 16, 112))
for i in range(70):
    rn = random.random()
    if rn < 0.16:
        jobs.append(Job.Job(i + 1, 8, 56, root, False))
    elif rn < 0.33:
        jobs.append(Job.Job(i + 1, 1, 1.75, root, False))
    elif rn < 0.5:
        jobs.append(Job.Job(i + 1, 4, 28, root, False))
    elif rn < 0.66:
        jobs.append(Job.Job(i + 1, 4, 7, root, False))
    elif rn < 0.83:
        jobs.append(Job.Job(i + 1, 8, 14, root, False))
    else:
        jobs.append(Job.Job(i + 1, 1, 0.75, root, False))

runAlgorithms(jobs, clusters, root)

# ----------------------------------------------------------------------------------------------
# Totally random Servers and VMs

clusters = []
jobs = []

for i in range(30):
    r1 = random.randrange(10, 20, 1)
    r2 = random.randrange(r1, 40, 1)
    clusters.append(Cluster.Cluster(i+1, r1, r2))

for i in range(70):
    jobs.append(Job.Job(i+1, 10, 20, root, True))

runAlgorithms(jobs, clusters, root)


