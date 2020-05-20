from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import os
import argparse
import json
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from pre_trained_model import RawNetWithSA

def evaluate_model(mode, model, db_gen, l_utt, l_trial, args, device):
    if mode not in ['val','eval']: raise ValueError('mode should be either "val" or "eval"')
    model.eval()
    with torch.set_grad_enabled(False):
        #1st, extract speaker embeddings.
        l_embeddings = []
        with tqdm(total = len(db_gen), ncols = 70) as pbar:
            for m_batch in db_gen:
                m_batch = m_batch.to(device)
                code = model(x = m_batch, is_test=True)
                l_embeddings.extend(code.cpu().numpy()) #>>> (batchsize, codeDim)
                pbar.update(1)
        d_embeddings = {}
        if not len(l_utt) == len(l_embeddings):
            print(len(l_utt), len(l_embeddings))
            exit()
        for k, v in zip(l_utt, l_embeddings):
            d_embeddings[k] = v
            
        #2nd, calculate EER
        y_score = [] # score for each sample
        y = [] # label for each sample         
        for line in l_trial:
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(cos_sim(d_embeddings[utt_a], d_embeddings[utt_b]))
        fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def cos_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_val_utts(l_val_trial):
    l_utt = []
    for line in l_val_trial:
        _, utt_a, utt_b = line.strip().split(' ')
        if utt_a not in l_utt: l_utt.append(utt_a)
        if utt_b not in l_utt: l_utt.append(utt_b)
    return l_utt

def get_utt_list(src_dir):
    '''
    Designed for VoxCeleb
    '''
    l_utt = []
    for path, dirs, files in os.walk(src_dir):
        base = '/'.join(path.split('/')[-2:])+'/'
        for file in files:
            #if file[-3:] != 'npy':
            if file[-3:] != 'wav':
                continue
            l_utt.append(base+file)
            
    return l_utt

def get_label_dic_Voxceleb(l_utt):
    d_label = {}
    idx_counter = 0
    for utt in l_utt:
        spk = utt.split('/')[0]
        if spk not in d_label:
            d_label[spk] = idx_counter
            idx_counter += 1
    return d_label

class Dataset_VoxCeleb2(data.Dataset):
    def __init__(self, list_IDs, base_dir, nb_samp = 0, labels = {}, cut = True, return_label = True, pre_emp = True):
        '''
		self.list_IDs	: list of strings (each string: utt key)
		self.labels		: dictionary (key: utt key, value: label integer)
		self.nb_samp	: integer, the number of timesteps for each mini-batch
		cut				: (boolean) adjust utterance duration for mini-batch construction
		return_label	: (boolean) 
		pre_emp			: (boolean) do pre-emphasis with coefficient = 0.97
        '''
        self.list_IDs = list_IDs
        self.nb_samp = nb_samp
        self.base_dir = base_dir
        self.labels = labels
        self.cut = cut
        self.return_label = return_label
        self.pre_emp = pre_emp
        if self.cut and self.nb_samp == 0: raise ValueError('when adjusting utterance length, "nb_samp" should be input')

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        try:
            X, _ = sf.read(self.base_dir+ID, dtype='int16') 
            X = np.asarray(X, dtype=np.int16).T 

            #X, _ = sf.read(self.base_dir+ID)
            #X = X.astype(np.float64)
        except:
            raise ValueError('%s'%ID)

        X = X.reshape(1,-1)
        if self.pre_emp: X = self._pre_emphasis(X)

        if self.cut:
            nb_time = X.shape[1]
            if nb_time > self.nb_samp:
                start_idx = np.random.randint(low = 0, high = nb_time - self.nb_samp)
                X = X[:, start_idx : start_idx + self.nb_samp]
            elif nb_time < self.nb_samp:
                nb_dup = int(self.nb_samp / nb_time) + 1
                X = np.tile(X, (1, nb_dup))[:, :self.nb_samp]
            else:
                X = X 
        if not self.return_label:
            return X
        y = self.labels[ID.split('/')[0]]
        return X, y

    def _pre_emphasis(self, x):
        '''
        Pre-emphasis for single channel input
        '''
        return np.asarray(x[:,1:] - 0.97 * x[:, :-1], dtype=np.float32) 


def main():
    parser = argparse.ArgumentParser()
    #dir
    parser.add_argument('-pretrained_name', type = str, required = True)
    parser.add_argument('-pretrained_dir', type = str, default = '../pre-trained_models/')
    parser.add_argument('-trials_dir', type = str, default = '../trials/')
    parser.add_argument('-save_dir', type = str, default = '../exp/')
    parser.add_argument('-DB', type = str, default = '../DB/VoxCeleb1_wav/')
    parser.add_argument('-DB_vox2', type = str, default = '../DB/VoxCeleb2_wav/')
    parser.add_argument('-dev_wav', type = str, default = 'wav/')
    parser.add_argument('-val_wav', type = str, default = 'dev_wav/')
    parser.add_argument('-eval_wav', type = str, default = 'eval_wav/')
    
    #hyper-params
    parser.add_argument('-bs', type = int, default = 60)
    parser.add_argument('-lr', type = float, default = 0.001)
    parser.add_argument('-nb_samp', type = int, default = 59049)
    parser.add_argument('-window_size', type = int, default = 11810)
    parser.add_argument('-nb_ta_samp_low', type = int, default = 16038) #1s
    parser.add_argument('-nb_ta_samp_high', type = int, default = 48114) #3s
    
    parser.add_argument('-wd', type = float, default = 0.0001)
    parser.add_argument('-epoch', type = int, default = 80)
    parser.add_argument('-optimizer', type = str, default = 'Adam')
    parser.add_argument('-nb_worker', type = int, default = 8)
    parser.add_argument('-temp', type = float, default = .5)
    parser.add_argument('-seed', type = int, default = 1234) 
    parser.add_argument('-nb_val_trial', type = int, default = 40000) 
    parser.add_argument('-lr_decay', type = str, default = 'keras')
    parser.add_argument('-model', type = json.loads, default = 
        '{"first_conv":3, "in_channels":1, "filts":[128, [128,128], [128,256], [256,256]],' \
        '"blocks":[2,4], "nb_fc_att_node":[1], "nb_fc_node":1024, "gru_node":1024, "nb_gru_layer":1}')
    
    #flag
    parser.add_argument('-amsgrad', type = bool, default = True)
    parser.add_argument('-make_val_trial', type = bool, default = False)
    parser.add_argument('-debug', type = bool, default = False)
    parser.add_argument('-save_best_only', type = bool, default = False)
    parser.add_argument('-do_lr_decay', type = bool, default = True)
    parser.add_argument('-mg', type = bool, default = True)

    args = parser.parse_args()

    #set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print('Device: {}'.format(device))

    #get utt_lists & define labels
    l_dev = sorted(get_utt_list(args.DB_vox2 + args.dev_wav))
    l_eval = sorted(get_utt_list(args.DB + args.eval_wav))
    d_label_vox2 = get_label_dic_Voxceleb(l_dev)
    args.model['nb_classes'] = len(list(d_label_vox2.keys()))
 
    with open(args.trials_dir + 'veri_test.txt', 'r') as f:
        l_eval_trial = f.readlines()        


    #define dataset generators
    evalset = Dataset_VoxCeleb2(list_IDs = l_eval,
		cut = False,
		return_label = False,
		base_dir = args.DB+args.eval_wav)
    evalset_gen = data.DataLoader(evalset,
		batch_size = 1, #because for evaluation, we do not modify its duration, thus cannot compose mini-batches
		shuffle = False,
		drop_last = False,
		num_workers = args.nb_worker)   
    sh_evalset_48114 = Dataset_VoxCeleb2(list_IDs = l_eval,
		return_label = False,
        nb_samp = 48114,
		base_dir = args.DB+args.eval_wav)
    sh_evalset_gen_48114 = data.DataLoader(sh_evalset_48114,
		batch_size = args.bs,
		shuffle = False,
		drop_last = False,
		num_workers = args.nb_worker)
    sh_evalset_32076 = Dataset_VoxCeleb2(list_IDs = l_eval,
		return_label = False,
        nb_samp = 32076,
		base_dir = args.DB+args.eval_wav)
    sh_evalset_gen_32076 = data.DataLoader(sh_evalset_32076,
		batch_size = args.bs,
		shuffle = False,
		drop_last = False,
		num_workers = args.nb_worker)
    sh_evalset_16038 = Dataset_VoxCeleb2(list_IDs = l_eval,
		return_label = False,
        nb_samp = 16038,
		base_dir = args.DB+args.eval_wav)
    sh_evalset_gen_16038 = data.DataLoader(sh_evalset_16038,
		batch_size = args.bs,
		shuffle = False,
		drop_last = False,
		num_workers = args.nb_worker)
  

    #pretrained model
    if bool(args.mg):
        model_1gpu = RawNetWithSA(args.model, device)
        model_1gpu.load_state_dict(torch.load(args.pretrained_dir + args.pretrained_name))
        model = nn.DataParallel(model_1gpu).to(device)
    else:
        model = RawNetWithSA(args.model, device).to(device)
        model.load_state_dict(torch.load(args.pretrained_dir + args.pretrained_name))


    ##########################################
	#Test####################################
	##########################################

    best_eval_eer = 99.
    best_sh_eval_eer_48114 = 99.
    best_sh_eval_eer_32076 = 99.  
    best_sh_eval_eer_16038 = 99.  
    
    eval_eer = evaluate_model(mode = 'eval',
	    model = model,
	    db_gen = evalset_gen, 
	    l_utt = l_eval,
	    device = device,
	    l_trial = l_eval_trial,
	    args = args)
    print('eval_eer : ', eval_eer)

    sh_eval_eer_48114 = evaluate_model(mode = 'eval',
        model = model,
        db_gen = sh_evalset_gen_48114,
        l_utt = l_eval,
        device = device,
        l_trial = l_eval_trial,
        args = args)
    print('sh_eval_eer_48114 : ', sh_eval_eer_48114)

    sh_eval_eer_32076 = evaluate_model(mode = 'eval',
        model = model,
        db_gen = sh_evalset_gen_32076,
        l_utt = l_eval,
        device = device,
        l_trial = l_eval_trial,
        args = args)
    print('sh_eval_eer_32076 : ', sh_eval_eer_32076)

    sh_eval_eer_16038 = evaluate_model(mode = 'eval',
        model = model,
        db_gen = sh_evalset_gen_16038,
        l_utt = l_eval,
        device = device,
        l_trial = l_eval_trial,
        args = args)
    print('sh_eval_eer_16038 : ', sh_eval_eer_16038)

if __name__ == '__main__':
    main()
