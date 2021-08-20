import numpy as np

class Opt:
    def __init__(self):
        self.arch='mmlpnrnet'
        self.batch_size=16
        self.checkpoints_dir='./checkpoints'
        self.dataroot='G:/dataset/MCB_B/MCB_B/'
        self.dataset_mode='mcb_b'
        self.export_folder=''
        self.fc_n=100
        self.feature_dir='./features/'
        self.gpu_ids=[]
        self.init_gain=0.02
        self.init_type='normal'
        self.is_train=False
        self.max_dataset_size=np.inf
        self.name='MCBB_5c1000s'
        self.ncf=[64, 64, 128, 256, 512]
        self.ninput_edges=750
        self.norm='batch'
        self.num_aug=1
        self.num_groups=16
        self.num_neigb=5
        self.num_threads=3
        self.phase='retrieval'
        self.pool_res=[]
        self.query_index=0
        self.resblocks=0
        self.results_dir='./results/'
        self.sample_mesh=1
        self.search_methods=['IndexFlatL2']
        self.seed=None
        self.serial_batches=False
        self.test_namelist='G:/dataset/MCB_B/MCB_B/namelist/mcbb_5c1000s.json'
        self.train_namelist='G:/dataset/MCB_B/MCB_B/namelist/mcbb_5c1000s.json'
        self.which_epoch='latest'
        self.which_layer='gb_pool'
        self.pooling='None'
        self.normalize=0
        self.input_nc = 5
        self.sample_mesh = 'trimesh'
        


