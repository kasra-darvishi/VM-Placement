import sqlite3

def initialize_table():
    conn = sqlite3.connect('info.db')
    #
    # conn.execute('''CREATE TABLE Cluster
    #         (ID INT PRIMARY KEY     NOT NULL,
    #         NODE_NUM       INT     NOT NULL,
    #         CPU_NUM        INT     NOT NULL,
    #         CORE_NUM       INT     NOT NULL,
    #         HYPER_THREAD   INT     NOT NULL,
    #         TOTAL_CORE_NUM   INT     NOT NULL,
    #         RAM_PER_NODE   INT     NOT NULL
    #         );''')
    #
    # idx = 1
    # nodeNum = 1
    # cpuNum = 8
    # coreNum = 8
    # hyperThread = 1
    #
    # conn.execute("INSERT INTO Cluster VALUES (1, 3, 2, 4, 1, 3*2*4*2, 64);")
    # conn.execute("INSERT INTO Cluster VALUES (2, 4, 2, 4, 1, 4*2*4*2, 64);")
    # conn.execute("INSERT INTO Cluster VALUES (3, 2, 2, 4, 1, 2*2*4*2, 33);")
    # conn.execute("INSERT INTO Cluster VALUES (4, 4, 2, 4, 1, 4*2*4*2, 59);")
    # conn.execute("INSERT INTO Cluster VALUES (5, 2, 2, 4, 1, 2*2*4*2, 63);")
    # conn.execute("INSERT INTO Cluster VALUES (6, 3, 2, 4, 1, 3*2*4*2, 23);")
    #
    # conn.commit();