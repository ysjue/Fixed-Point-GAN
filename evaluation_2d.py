import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

pred_path = '/data/shuojue/code/Fixed-Point-GAN/predictions'
neg_lab_path = '/data/shuojue/data/felix_3mm/real_test/negative_lab'
pos_lab_path = '/data/shuojue/data/felix_3mm/real_test/positive_lab'
test_txt = '/data/shuojue/code/Fixed-Point-GAN/split/test_set.txt'

class evaluator(object):
    def __init__(self,pred_path,neg_lab_path,pos_lab_path,test_txt ):
        self.pred_path = pred_path
        self.neg_lab_path = neg_lab_path
        self.pos_lab_path = pos_lab_path
        self.test_txt = test_txt
        
    def __call__(self):
        pred_path = self.pred_path
        neg_lab_path = self.neg_lab_path
        pos_lab_path = self.pos_lab_path
        test_txt = self.test_txt
        with open(test_txt, 'r') as f:
            lines = f.readlines()
        cases = [l.split('\n')[0] for l in lines]

        def sort_slices(slice_paths):
            new_list = []
            for i in range(len(slice_paths)):
                new_list += [path for path in slice_paths if '_{0:}.jpg'.format(i) in path]
            assert len(new_list) == len(slice_paths)
            return new_list

        Normal =0
        Abnormal = 0
        TP = 0
        FP = 0
        PDAC_count = 0
        patient_level = False
        for case in tqdm(cases):
            case = case.split('.nii.gz')[0]
            lab_paths = glob(os.path.join(neg_lab_path,case + '*.jpg')) \
                        + glob(os.path.join(pos_lab_path,case + '*.jpg'))
            pred_paths = glob(os.path.join(pred_path,case + '*.jpg'))
            assert len(lab_paths) == len(pred_paths)
            lab_paths, pred_paths = sort_slices(lab_paths), sort_slices(pred_paths)
            lab, pred = [],[]
            for lab_2d_path,pred_2d_path in zip(lab_paths,pred_paths):
                lab_2d = np.array(Image.open(lab_2d_path))
                lab_2d = lab_2d == 3
                lab.append(lab_2d)
                
                pred_2d = np.array(Image.open(pred_2d_path))
                pred_2d = pred_2d > 0
                pred.append(pred_2d)
            tumor_label = np.array(lab)
            tumor_pred = np.array(pred)
            
            if case[5] in ['0','1','P']:
                    Normal+=1
                    if np.sum(tumor_pred)>0:
                        FP+=1
            else:
                if np.sum(tumor_label)>0:
                    Abnormal+=1
                    if patient_level:
                        if np.sum(tumor_pred) > 0:
                            TP+=1
                    else:
                        if np.sum((tumor_label)*(tumor_pred))>0:
                            TP+=1
                    if np.sum(tumor_pred)>0:
                        PDAC_count+=1
                


        # print sen and spe
        print('========================================')
        print('Total Abnormal Case ', Abnormal)
        print('Detected Abnormal Case ', TP)
        print('Sensitivity ', TP/(Abnormal+0.0001))
        print('Total Normal Case ', Normal)
        print('FP Normal Case ', FP)
        print('Specificity ', 1-FP/(Normal+0.00001))
        sen_spe_dict = {'sen':TP/(Abnormal+0.0001), 'spe':1-FP/(Normal+0.00001)}


        
        
    
    
