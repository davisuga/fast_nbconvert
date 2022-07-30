import os
from shutil import copyfile

print("Building...")
binPath ="./fast_nbconvert/fast_nbconvert"
copyfile("_build/default/bin/main.exe", binPath)
os.system("chmod 777 "+binPath)