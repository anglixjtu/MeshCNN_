import numpy as np
from faiss import IndexFlatIP, IndexFlatL2
import pyvista as pv  

class Retriever:
    def __init__(self, opt):
        self.opt = opt
        self.search_methods = opt.search_methods
        self.num_neigb = opt.num_neigb
        self.database_namefile = opt.train_namelist
        self.query_namefile = opt.test_namelist
        self.database_namelist = open(self.database_namefile).readlines()
        self.query_namelist = open(self.query_namefile ).readlines()
        self.dataroot = opt.dataroot
        self.methods = opt.search_methods
        self.which_layer = opt.which_layer


    def extract_database_features(self, model, dataset):
        data_length = len(dataset)

        for i, data in enumerate(dataset):
            out_label, features = self.extract_feature(model, data)
            if i == 0:
                x = features
            else:
                x= np.append(x, features, axis=0)
        return x

    def retrive_one_example(self, model, query, dataset, fea_db=None):
        label_q, fea_q = self.extract_feature(model, query)
        
        if fea_db is None:
            fea_db = self.extract_database_features(model, dataset)
        data_len, fea_len = fea_db.shape

        dist, ranked_list = self.search_indexflat(fea_q, fea_db, fea_len, self.num_neigb)

        return dist, ranked_list


    def retrive(self, model, query, dataset, fea_db=None):
       
        if fea_db is None:
            fea_db = self.extract_database_features(model, dataset)
        data_len, fea_len = fea_db.shape

        if self.database_namefile == self.query_namefile:
            fea_q = fea_db
        else:
            fea_q = self.extract_database_features(model, query)

        dist, ranked_list = self.search_indexflat(fea_q, fea_db, fea_len, self.num_neigb)

        return dist, ranked_list


    def search_indexflat(self, fea_q, fea_db, dim, k):
        D, I = {}, {}
        for method in self.methods:
            index = eval(method)(dim)
            index.add(fea_db) 
            D[method], I[method] = index.search(fea_q, k)
        #TODO: add threshold for the retrieval results
        return D, I
      

    def extract_feature(self, model, data):
        model.set_input(data)
        out_label, features = model.forward()
        return out_label.cpu().detach().numpy(),\
               features[self.which_layer].cpu().detach().numpy()

    def show_results(self, idx_query, idx_list):

        font_size = 10
        num_methods = len(self.methods)
        p = pv.Plotter(shape=(num_methods, self.num_neigb+1))
        query_file = self.dataroot + self.query_namelist[idx_query].strip('\n')
        mesh_q = pv.read(query_file)

        for m, method in enumerate(self.methods):      
            p.subplot(m, 0)
            p.add_text("{}-query".format(method), font_size=font_size)
            p.add_mesh(mesh_q, color="tan", show_edges=True)
            for i, index in enumerate(idx_list[method][0]):
                filename = self.dataroot + self.database_namelist[index].strip('\n')
                label = filename.split('/')[-2]
                mesh = pv.read(filename)
                p.subplot(m, i+1)
                p.add_text("{}-{}".format(method, label), font_size=font_size)
                p.add_mesh(mesh, color="tan", show_edges=True)

        p.show()



        


