import json

name_dict = {}
namelist = {}
sets = {'train', 'test'}
namelist['train'] = 'E:/ARIH/3D_modeling/tools/train_10c1000s.txt'
namelist['test'] = 'E:/ARIH/3D_modeling/tools/test_10c1000s.txt'
for dataset in sets:
    name_dict[dataset] = {}
    classes = []
    with open(namelist[dataset], 'r') as f:
        for name in f.readlines():
            curt_class = name.split('/')[1]
            if curt_class not in classes:
                classes.append(curt_class)
                name_dict[dataset][curt_class] = []
            name_dict[dataset][curt_class] += [name.strip('\n')]
with open('G:/dataset/MCB_B/MCB_B/namelist/mcbb_10c1000s.json', 'w') as fp:
    json.dump(name_dict, fp)