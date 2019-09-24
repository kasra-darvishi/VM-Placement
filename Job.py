import random, Cluster, math


class Job:

    # def __init__(self, id, max_core_num, max_ram, logger):
    #     self.id = id
    #     randomNum = random.randrange(1, int(math.log(max_core_num, 2)), 1)
    #     self.needed_core_num = 2 ** randomNum
    #     self.needed_ram = 2 * self.needed_core_num
    #     self.allocated_cluster_id = -1
    #     logger.info("   Job id: " + str(id) + " number of cores: " + str(self.needed_core_num) + " ram: " + str(self.needed_ram))

    def __init__(self, id, coreNum, ram, logger, makeRandomValues):

        self.id = id
        self.allocated_cluster_id = -1
        if not makeRandomValues:
            self.needed_core_num = coreNum
            self.needed_ram = ram
            logger.info("   Job id: " + str(id) + " number of cores: " + str(self.needed_core_num) + " ram: " + str(self.needed_ram))
        else:
            self.needed_core_num = random.randrange(1, coreNum, 1)
            self.needed_ram = random.randrange(1, max(2, min(ram, 2*self.needed_core_num)), 1)
            logger.info("   Job id: " + str(id) + " number of cores: " + str(self.needed_core_num) + " ram: " + str(self.needed_ram))

    def refresh(self):
        self.allocated_cluster_id = None

    def set_cluster(self, cluster_id):
        self.allocated_cluster_id = cluster_id

    def get_cluster(self):
        return self.allocated_cluster_id