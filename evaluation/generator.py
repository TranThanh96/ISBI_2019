import os
import glob
import random
import numpy as np
import cv2


class Generator(object):
    sample = 'sample'
    label = 'label'
    augment = 'augment'

    # def __init__(self, path='/media/HDD-2T/DATA/ISBI_2019',labels=['cancer','normal'],folds = ['fold_1','fold_2','fold_3','fold_4'],type='train'):
    def __init__(self, path,labels,folds,type_set):
        '''
        example
        def __init__(self, path='/media/HDD-2T/DATA/ISBI_2019',labels=['cancer','normal'],folds = ['fold_1','fold_2','fold_3','fold_4'],type='train)
        '''
        self.type = type_set # 'train' or 'val'
        self.all_samples = self.generate_all_samples_dict(path, labels, folds)
        np.random.shuffle(self.all_samples['cancer'])
        np.random.shuffle(self.all_samples['normal'])
        self.length = len(self.all_samples['cancer']) + len(self.all_samples['normal'])
        self.pos_weight = len(self.all_samples['normal'])/len(self.all_samples['cancer'])
        print('>>>>>>>>>>>>>>. post weight: ', self.pos_weight)
        print('num_cancer: ', len(self.all_samples['cancer']))
        print('num_norma;: ', len(self.all_samples['normal']))
        # input('lakag matatg')

    def generate_all_samples_dict(self, path, labels, folds):
        # generates a dictionary between a person and all the photos of that person
        all_samples = {}
        
        for label in labels:
            all_samples[label] = []
            for fold in folds:
                sample_subject = glob.glob('{}/{}/{}/*'.format(path,fold,label))
                all_samples[label]+=(sample_subject)
        return all_samples

    def get_next(self):
        all_samples_names = list(self.all_samples.keys())

        labels = {
            'cancer': [0.0, 1.0],
            'normal': [1.0, 0.0]
        }
        # labels = {
        #     'cancer': [1.0, 0.0],
        #     'normal': [0.0, 1.0]
        # }

        if self.type == 'train':
            num_samples_cancer_used = 0
            num_samples_normal_used = 0
            while True:
                # label = random.choice(all_samples_names) # 'cancer' or 'normal'
                choose_cancer = random.random() > 2.0/3
                if choose_cancer:
                    label = 'cancer'
                    # augment = False
                else:
                    label = 'normal'
                    # augment = False
                if label == 'cancer':
                    sample = self.all_samples[label][num_samples_cancer_used]
                    num_samples_cancer_used += 1
                else:
                    sample = self.all_samples[label][num_samples_normal_used]
                    num_samples_normal_used += 1
                
                if num_samples_cancer_used == len(self.all_samples['cancer']):
                    num_samples_cancer_used = 0
                    np.random.shuffle(self.all_samples['cancer'])
                
                if num_samples_normal_used == len(self.all_samples['normal']):
                    num_samples_normal_used = 0
                    np.random.shuffle(self.all_samples['normal'])

                # sample = random.choice(self.all_samples[label])
                yield ({self.sample: sample,
                        self.label: labels[label],
                        self.augment: False})
                        
        elif self.type == 'eval':
            while True:
                for sample in self.all_samples['cancer']:
                    yield ({self.sample: sample,
                        self.label: labels['cancer'],
                        self.augment: False}) 
                for sample in self.all_samples['normal']:
                    yield ({self.sample: sample,
                        self.label: labels['normal'],
                        self.augment: False})
                    
        else:
            raise ValueError('type must be "train" or "eval"')
        # while True:
        #     label = random.choice(all_samples_names) # 'cancer' or 'normal'

        #     sample = random.choice(self.all_samples[label])

        #     yield ({self.sample: sample,
        #             self.label: labels[label]})
# a = Generator()
# b = a.get_next()
# print(b)
