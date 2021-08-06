import numpy as np
from faiss import IndexFlatIP, IndexFlatL2
import pyvista as pv  
import os
import time
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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
        self.pooling = opt.pooling
        self.feature_size = 512
        self.pooling_set = ['global_mean_pool', 'global_add_pool', 'global_max_pool']
        self.normalize = opt.normalize




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

        dist, ranked_list, dissm = self.search_indexflat(fea_q, fea_db, fea_len, self.num_neigb)

        return dist, ranked_list, dissm


    def retrieve(self, model, queryset, dataset, fea_db=None, fea_q=None):
       
        if fea_db is None:
            fea_db = self.extract_database_features(model, dataset)

        if fea_q is None:
            if self.database_namefile == self.query_namefile:
                fea_q = fea_db
            else:
                fea_q = self.extract_database_features(model, queryset)

        data_len, fea_len = fea_db.shape

        dist, ranked_list, dissm = self.search_indexflat(fea_q, fea_db, fea_len, self.num_neigb)

        return dist, ranked_list, dissm


    def search_indexflat(self, fea_q, fea_db, dim, k):
        D, I, dissm = {}, {}, {}
        for method in self.methods:
            index = eval(method)(dim)
            index.add(fea_db) 
            D[method], I[method] = index.search(fea_q, k)
            # compute dis-similarity score
            max_dist = D[method].max()
            dissm[method] = D[method] / max_dist
        #TODO: add threshold for the retrieval results
        return D, I, dissm
      

    def extract_feature(self, model, data):
        model.set_input(data)
        out_label, features = model.forward()
        features = features[self.which_layer]
        if self.pooling in self.pooling_set:
            features = eval(self.pooling)(features, model.data.batch)
        if self.normalize:
            features = F.normalize(features, p=self.normalize, dim=1)
        self.feature_size = features.shape[1]
        return out_label.cpu().detach().numpy(),\
               features.cpu().detach().numpy()

    def show_results(self, idx_query, idx_list, dissm=None):

        font_size = 10
        num_methods = len(self.methods)
        p = pv.Plotter(shape=(num_methods, self.num_neigb+1))
        query_file = self.dataroot + self.query_namelist[idx_query].strip('\n')
        mesh_q = pv.read(query_file)

        for m, method in enumerate(self.methods):      
            p.subplot(m, 0)
            p.add_text("{}-query".format(method), font_size=font_size)
            p.add_mesh(mesh_q, color="tan", show_edges=True)
            
            if len(idx_list[method]) == 1:
                indices = idx_list[method][0]
                if dissm is not None: ds = dissm[method][0]
            else:
                indices = idx_list[method][idx_query]
                if dissm is not None: ds = dissm[method][idx_query]

            for i, index in enumerate(indices):
                filename = self.dataroot + self.database_namelist[index].strip('\n')
                label = filename.split('/')[-2]
                mesh = pv.read(filename)
                p.subplot(m, i+1)
                p.add_text("{}-{}".format(method, label), font_size=font_size)
                if dissm is not None:
                    p.add_text("\n\ndissimilarity: %.3f"%(ds[i]), font_size=font_size)
                p.add_mesh(mesh, color="tan", show_edges=True)

        p.show()

    def get_patk(self, gt_label, label_list, K):
        """
        Calculate Precision@K

        Args:
            gt_label: the ground truth label for query
            label_list: labels for the retrieved ranked-list
            K: top K in the ranked-list for computing precision. Set to len(label_list) if compute P@N

        Returns:
            P@K score
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
            Args:
            gt_label: the ground truth label for query
            label_list: labels for the retrieved ranked-list

        Returns:
            AP score
        """
        #TODO: mAP and NDCG need to be divided by total number of relevant models in ground truth set, not in the retrieval list itself
        map = 0
        counter = 0
        for k in range(len(label_list)):
            if gt_label == label_list[k]: # at each relevant positions
                map += self.get_patk(gt_label, label_list, k+1)
                counter += 1
        '''if counter == 0:
            map = 0
        else:
            map /= counter'''

        map /= len(label_list)

        return map

    def get_ndcg(self, gt_label, label_list, K):
        """
        Calculate Normalized Cumulative Gain (NDCG) at rank K

        Args:
            gt_label: the ground truth label for query
            label_list: labels for the retrieved ranked-list
            K: top K in the ranked-list for computing precision. Set to len(label_list) if compute NDCG@N

        Returns:
            NDCG@K
        
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
                  'Using pooling                   [%s]. \n'\
                  'Feature length                  [%d]. \n'\
                  'Normalize                       [%d]. \n'\
                  %(now, self.num_neigb, self.threshold, self.which_layer, self.pooling, self.feature_size, self.normalize)
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


    def show_embedding(self, features, idx_list):
        label_list = self.get_labels_from_index(idx_list)
        writer = SummaryWriter('runs/embedding')
        writer.add_embedding(features,
                    metadata=label_list)
        writer.close()



        












        


