import os


global root


def init():
    global root
    root = os.getcwd()
    os.system("cp  {}/results/* {}/results_backup/".format(root, root))



