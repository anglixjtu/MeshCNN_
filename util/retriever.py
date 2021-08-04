import numpy as np
from faiss import IndexFlatIP, IndexFlatL2
import pyvista as pv  
import os
import time

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
        self.threshold = 1 #TODO: get the threshold from args
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'retrieval_acc.txt')




    def extract_database_features(self, model, dataset):
        data_length = len(dataset)

        for i, data in enumerate(dataset):
            out_label, features = self.extract_feature(model, data)
            if i == 0:
                x = features
            else:
                x= np.append(x, features, axis=0)
        return x

    def retrieve_one_example(self, model, query, dataset, fea_db=None):
        label_q, fea_q = self.extract_feature(model, query)
        
        if fea_db is None:
            fea_db = self.extract_database_features(model, dataset)
        data_len, fea_len = fea_db.shape

        dist, ranked_list = self.search_indexflat(fea_q, fea_db, fea_len, self.num_neigb)

        return dist, ranked_list


    def retrieve(self, model, queryset, dataset, fea_db=None, fea_q=None):
       
        if fea_db is None:
            fea_db = self.extract_database_features(model, dataset)

        if fea_q is None:
            if self.database_namefile == self.query_namefile:
                fea_q = fea_db
            else:
                fea_q = self.extract_database_features(model, queryset)

        data_len, fea_len = fea_db.shape

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

    def get_patk(self, gt_label, label_list, K):
        """
        Calculate Precision@K
        """
        patk = 0
        for i, pred_label in enumerate(label_list[:K]):
            if gt_label == pred_label:
                patk += 1
        patk /= K

        return patk


    def get_map(self, gt_label, label_list):
        """
        Calculate mean average precision
        """
        map = 0
        for k in range(1, len(label_list)):
            map += self.get_patk(gt_label, label_list, k)
        map /= len(label_list)

        return map

    def get_ndcg(self, gt_label, label_list, K):
        """
        Calculate Normalized Cumulative Gain (NDCG) at rank K
        """
        dcg = 0
        dcg_gt = 0
        for i, pred_label in enumerate(label_list[:K]):
            if gt_label == pred_label:
                dcg += 1 * ((i+1)**-0.5)
            dcg_gt += 1 * ((i+1)**-0.5)
        dcg /= dcg_gt

        return dcg


    def get_labels_from_index(self, indices):
        labels = []
        for index in indices:
            label = self.database_namelist[index].strip('\n').split('/')[-2]
            labels += [label]
        return labels
    
    def reset_metrics(self):
        # for evaluation
        self.PatN = {m:0 for m in self.methods} 
        self.RatN = {m:0 for m in self.methods} 
        self.F1atN = {m:0 for m in self.methods} 
        self.mAP = {m:0 for m in self.methods} 
        self.NDCGatN = {m:0 for m in self.methods}
        self.counter = {m:0 for m in self.methods}
        

    def evaluate_results(self, idx_query, idx_list):
        self.reset_metrics()

        labels_query = self.get_labels_from_index(idx_query)
        for m, method in enumerate(self.methods):
            for i, gt_label in enumerate(labels_query):
                labels_list = self.get_labels_from_index(idx_list[method][i])
                self.PatN[method] += self.get_patk(gt_label, labels_list, len(labels_list))
                self.mAP[method] += self.get_map(gt_label, labels_list)
                self.NDCGatN[method] += self.get_ndcg(gt_label, labels_list, len(labels_list))
                self.counter[method] += 1

            self.PatN[method] /= self.counter[method]
            self.mAP[method] /= self.counter[method]
            self.NDCGatN[method] /= self.counter[method]
        
        now = time.strftime("%c")
        message = '================ Retrieval Acc (%s) ================\n'\
                  'Maximum retrieve %d nearest samples with threshold %.2f. \n'\
                  'Using the embeddings from layer [%s]. \n'\
                  %(now, self.num_neigb, self.threshold, self.which_layer)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        for m, method in enumerate(self.methods):
            message = 'Distance metric: %s \n'\
                      'P@N: %.3f, mAP: %.3f, NDCG@N: %.3f \n'\
                      %(method, self.PatN[method], self.mAP[method], self.NDCGatN[method])
            print(message)
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)

        












        


