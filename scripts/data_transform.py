import os
import numpy as np
import scipy.io
import sys
modelName='01_MorphableModel'
save_dir = '../npz_model'
load_dir = '../matlab_model'
NP_LOAD_FILE = modelName+'.mat'
NP_SAVE_FILE = modelName+'.npz'        
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

np_save_path = os.path.join(save_dir, NP_SAVE_FILE)   
np_load_path = os.path.join(load_dir, NP_LOAD_FILE)   

def main(args):    
    raw_model_data = scipy.io.loadmat(np_load_path)
    model_data_np = {}
    for k in raw_model_data.keys():
        print('------------')
        print(k)
        if(k in ['tl','shapeMU','shapePC','shapeEV','texMU','texPC','texEV']):            
            try:
                if(scipy.sparse.issparse(raw_model_data[k])):
                    print("sparse")
                else:
                    print("dense")
                print(raw_model_data[k].dtype)
                try:                
                    print(raw_model_data[k].shape)
                except:
                    print('')
                if(k=='tl'):
                    model_data_np[k]=np.require((raw_model_data[k]),dtype=np.int32,requirements=['C'])
                else:
                    model_data_np[k]=np.require((raw_model_data[k]),requirements=['C'])
                
            except:
                print('skipped '+ k)
    np.savez(np_save_path, **model_data_np)
    print('Save BFM Model to: ', os.path.abspath(save_dir))


if __name__ == '__main__':
    main(sys.argv)
