class Queue:

    def __init__(self):
        self.queue = list()

    # Insert method to add element
    def addtoq(self,dataval):
        if dataval not in self.queue:
            self.queue.insert(0,dataval)
            return True
        return False

    # Pop method to remove element
    def removefromq(self):
        if len(self.queue)>0:
            return self.queue.pop()
        return ("No elements in Queue!")