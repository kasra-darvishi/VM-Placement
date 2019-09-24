class Cluster:
    # core_num = 0
    # init_core_num = 0
    # ram = 0
    # init_ram = 0
    # allocatedJobs = []
    # id = 0
    # hasExceeded = False

    def __init__(self, id, core_num, ram):
        self.id = id
        self.core_num = core_num
        self.ram = ram
        self.init_core_num = core_num
        self.init_ram = ram
        self.allocatedJobs = []
        self.hasExceeded = False

    def allocate(self, job):

        for allocatedJob in self.allocatedJobs:
            if allocatedJob.id == job.id:
                print("!!!!!tekrari job allocate shod")
                return

        self.core_num -= job.needed_core_num
        self.ram -= job.needed_ram
        self.allocatedJobs.append(job)
        job.set_cluster(self.id)

        if self.ram < 0 or self.core_num < 0:
            self.hasExceeded = True

    def can_allocate(self, job):
        if job.needed_core_num <= self.core_num and job.needed_ram <= self.ram:
            return True
        else:
            return False

    def contains_job(self, job):
        for allocatedJob in self.allocatedJobs:
            if allocatedJob.id == job.id:
                return True
        return False

    def exchange(self, job_1, job_2):
        temp_coreNum = self.core_num + job_1.needed_core_num
        temp_ram = self.ram + job_1.needed_ram
        if job_2.needed_core_num <= temp_coreNum and job_2.needed_ram <= temp_ram:
            self.deAllocate(job_1)
            if job_2.alocated_cluster is not None:
                job_2.alocated_cluster.deAllocate(job_2)
            self.allocate(job_2)
            return True
        else:
            return False

    def deAllocate(self, job):

        jobToRemove = None
        for allocatedJob in self.allocatedJobs:
            if allocatedJob.id == job.id:
                self.core_num += job.needed_core_num
                self.ram += job.needed_ram
                jobToRemove = allocatedJob
                if self.ram < 0 or self.core_num < 0:
                    self.hasExceeded = True
                else:
                    self.hasExceeded = False
                break

        if jobToRemove is not None:
            self.allocatedJobs.remove(jobToRemove)
            jobToRemove.refresh()
            job.refresh()
            return True

        return False

    def refresh(self):
        # for job in self.allocatedJobs:
        #     self.core_num += job.needed_core_num
        #     self.ram += job.needed_ram
        #     job.refresh
        self.ram = self.init_ram
        self.core_num = self.init_core_num
        self.allocatedJobs.clear()
        self.hasExceeded = False

    def removeOneJob(self):
        job = self.allocatedJobs.pop(0)
        job.refresh()
        self.core_num += job.needed_core_num
        self.ram += job.needed_ram
        if self.ram < 0 or self.core_num < 0:
            self.hasExceeded = True
        else:
            self.hasExceeded = False
        return job

    def isEmpty(self):
        return self.core_num == self.init_core_num and self.ram == self.init_ram

    def printInfo(self):
        print("\nserver #", self.id)
        for job in self.allocatedJobs:
            print("job id: ", job.id, " its server: ", job.allocated_cluster_id)

    def get_utilization_info(self):
        return (self.init_core_num - self.core_num)/self.init_core_num