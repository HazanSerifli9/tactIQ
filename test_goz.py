import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('göztepehub'))
from göztepehub.pages import goztepe
print("Imported goztepe")
print(goztepe.layout())
