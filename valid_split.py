import os 
import tqdm
import numpy as np
import random

path = '/medical_backup/NII_data/felix_3mm/label_6cls'
all_files = os.listdir(path)
files = [f for f in all_files if 'VENOUS' in f or 'ARTERIAL' in f]
venous_files = [f for f in all_files if 'VENOUS' in f]
arterial_files = [f for f in all_files if 'ARTERIAL' in f]
a = []
cys = []
pdac = []
start_with_7 = []
health = []
errors = []
for name in tqdm.tqdm(venous_files):
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

print(len(venous_files),len(all_files))
print(len(cys))
print(len(pdac))
print(len(health))
print(len(start_with_7))
print(len(errors))

valid_set = []
train_set = []
val_cys = random.sample(cys, int(len(cys)*0.2))
valid_set += val_cys
train_set += [case for case in cys if case not in val_cys]
val_pdac = random.sample(pdac, int(len(pdac)*0.2))
valid_set += val_pdac
train_set += [case for case in pdac if case not in val_pdac]
val_health = random.sample(health, int(len(health)*0.2))
valid_set += val_health
train_set += [case for case in health if case not in val_health]
train_set += arterial_files

# with open('valid_set.txt', 'w') as f:
#     for name in valid_set:
#         f.write(name + '\n')
# with open('train_set.txt', 'w') as f:
#     for name in train_set:
#         f.write(name+'\n')