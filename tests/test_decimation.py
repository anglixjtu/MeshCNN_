import os
import sys
import json
import trimesh as tm
from pySimplify import pySimplify
import pyvista as pv


dataroot = 'G:/dataset/MCB_B/MCB_B/'
namelist_file = 'G:/dataset/MCB_B/MCB_B/namelist/mcbb_5c1000s.json'
names = []
count = 0
with open(namelist_file, 'r') as f:
    namelist = json.load(f)
    for phase in ['train', 'test']:
        dataset = namelist[phase]
        classes = list(dataset.keys())
        for target in classes:
            for x in dataset[target]:
                name = os.path.join(dataroot, x)
                print((name, count))

                mesh_in = tm.load(name)
                simplify = pySimplify()
                simplify.setMesh(mesh_in)
                simplify.simplify_mesh(target_count = 500, preserve_border=True)

                mesh_out = simplify.getMesh()

                #mesh_out = mesh_in.simplify_quadratic_decimation(500)
                print('before: %d '%(len(mesh_in.faces)))
                print('after:  %d '%(len(mesh_out.faces)))
                p = pv.Plotter(shape=(2, 1))
                p.subplot(0, 0)
                p.add_mesh(mesh_in, color="tan", show_edges=True)
                p.subplot(1, 0)
                p.add_mesh(mesh_out, color="tan", show_edges=True)
                p.show()
                count += 1
        