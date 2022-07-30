import os
import sys

def main():
    path = (os.path.realpath(__file__))
    binPath = path.replace(".py", "")   
    cmd = binPath+" "+" ".join(sys.argv[1:])
    os.system(cmd)

