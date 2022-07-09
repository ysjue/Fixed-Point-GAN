import os
from tkinter.filedialog import Open 
import tqdm

path = '/medical_backup/NII_data/felix_3mm/label_6cls'
all_files = os.listdir(path)
files = [f for f in all_files if 'VENOUS' in f or 'ARTERIAL' in f]
files = [f for f in all_files if 'VENOUS' in f]
a = []
cys = []
pdac = []
start_with_7 = []
health = []
errors = []
for name in tqdm.tqdm(files):
    if 'cys' in name.lower():
        cys.append(name)
    elif 'pdac' in name.lower() or 'FELIX5' in name:
        pdac.append(name)
    elif 'FELIX0' in name : #or 'FELIX1' in name:
        health.append(name)
    elif 'FELIX7' in name:
        start_with_7.append(name)
    else:
        errors.append(name)

print(len(files),len(all_files))
print(len(cys))
print(len(pdac))
print(len(health))
print(len(start_with_7))
print(len(errors))


