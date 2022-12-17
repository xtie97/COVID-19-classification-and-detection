import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable
import scipy
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
# from skimage import exposure
from CV19DataSet import CV19DataSet
from utils import compute_AUCs, delong_roc_variance
import pandas as pd
import argparse
import sys
from torchvision.models import vgg16_bn, squeezenet1_1, densenet121, convnext_base, swin_b, efficientnet_v2_m
from tqdm import * 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

N_CLASSES = 2
CLASS_NAMES = ['Covid', 'Non-covid']

def VGG_model():
    model = vgg16_bn(weights=None)
    model.classifier[6] = nn.Sequential(nn.Linear(4096, N_CLASSES),nn.Softmax(dim=1))
    model = nn.DataParallel(model.cuda())
    return model 

def Swin_model():
    model = swin_b(weights=None)
    model.head = nn.Sequential(nn.Linear(1024, N_CLASSES),nn.Softmax(dim=1))
    model = nn.DataParallel(model.cuda() ,device_ids=[0])
    return model 

def Squeeze_model():
    model = squeezenet1_1()
    model.classifier = nn.Sequential(nn.Dropout(p=0.2),nn.Conv2d(512, N_CLASSES, kernel_size=1),nn.AdaptiveAvgPool2d((1, 1)),nn.Softmax(dim=1))
    model = nn.DataParallel(model.cuda() ,device_ids=[0])
    return model 
   
def Efficient_model():
    model = efficientnet_v2_m(weights=None)
    model.classifier[1] = nn.Sequential(nn.Linear(1280, N_CLASSES), nn.Softmax(dim=1))
    model = nn.DataParallel(model.cuda() ,device_ids=[0])
    return model 
    
def Dense_model():
    model = densenet121(weights=None,drop_rate = 0.2)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, N_CLASSES), nn.Softmax(dim=1))
    model = nn.DataParallel(model.cuda() ,device_ids=[0])
    return model 
    
def Convx_model():
    model = convnext_base(weights=None)
    model.classifier[2] = nn.Sequential(nn.Linear(1024, N_CLASSES), nn.Softmax(dim=1))
    model = nn.DataParallel(model.cuda() ,device_ids=[0])
    return model          


def main(root_dir, test_model, df, num_models, BATCH_SIZE = 50):
    
    DATA_DIR = root_dir
    normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    transformSequence_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(normalizer[0], normalizer[1])])
    
    test_dataset = CV19DataSet(df=df, base_folder=DATA_DIR, transform=transformSequence_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=6, pin_memory=True, drop_last=False,persistent_workers=True)
    pred_np_total = np.zeros((len(df),num_models))
    
    for model_index in range(num_models):
        
        cudnn.benchmark = True
        # initialize and load the model
        
        model_name = '{}/{}_train_'.format(test_model, test_model) + str(model_index) + '.pth.tar'
        print('model path:', model_name)  
        
        if 'Swin' in test_model:
            model = Swin_model()
        elif 'VGG' in test_model:
            model = VGG_model() 
        elif 'Convnext' in test_model: 
            model = Convx_model() 
        elif 'Dense' in test_model: 
            model = Dense_model()
        elif 'Squeeze' in test_model: 
            model = Squeeze_model()
        elif 'Efficient' in test_model:
            model = Efficient_model() 
 
        if os.path.isfile(model_name):
            checkpoint = torch.load(model_name)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
        else:
            print("=> no checkpoint found")

        # Testing mode 
        gt_np, pred_np = epochVal(model, test_loader)
        alpha = 0.95
        auc_result, auc_cov = delong_roc_variance(gt_np, pred_np)
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        
        ci = scipy.stats.norm.ppf(
        lower_upper_q,
        loc=auc_result,
        scale=auc_std)
        ci[ci > 1] = 1
        
        print('AUC = {:.3f} [{:.3f},{:.3f}]'.format(auc_result,ci[0],ci[1]))
        pred_np_total[:,model_index] = pred_np
        
    pred_np_ensemble = np.sqrt(np.mean(pred_np_total**2, axis=1))
    auc_result, auc_cov = delong_roc_variance(gt_np, pred_np_ensemble)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        
    ci = scipy.stats.norm.ppf(
    lower_upper_q,
    loc=auc_result,
    scale=auc_std)
    ci[ci > 1] = 1
    print('Ensemble AUC = {:.3f} [{:.3f},{:.3f}]'.format(auc_result,ci[0],ci[1]))
    return pred_np_ensemble, gt_np, auc_result, ci
#-------------------------------------------------------------------------------- 

def epochVal (model, dataLoader):
    model.eval()
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    with torch.no_grad():
        for inp, target in tqdm(dataLoader):
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            
            output = model(inp.cuda())
            pred = torch.cat((pred, output.data), 0)
                
    torch.cuda.empty_cache()       
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    del gt, pred
    return gt_np[:,0], pred_np[:,0] 
        
def plot_roc_curve(gt_np, pred_np): 
    AUROCs = compute_AUCs(gt_np, pred_np)
    AUROC_avg = np.array(AUROCs).mean()
    
    COVID_predict_score = pred_np[:,COVID_label] # column 0 is COVID prediction score

    scores_covid = COVID_predict_score[gt_np[:,COVID_label]==1] # these are the covid+ cases
    scores_noncovid = COVID_predict_score[gt_np[:,COVID_label]==0] # these are the covid- cases
   
    covid_res = stats.relfreq(scores_covid, numbins=100)
    noncovid_res = stats.relfreq(scores_noncovid, numbins=100)
    scores_covid_ = covid_res.lowerlimit + np.linspace(0, covid_res.binsize*covid_res.frequency.size, covid_res.frequency.size)
    scores_noncovid_ = noncovid_res.lowerlimit + np.linspace(0, noncovid_res.binsize*noncovid_res.frequency.size, noncovid_res.frequency.size)                         
                                
    fig = plt.figure(figsize=(5, 4))
    plt.bar(scores_covid_, covid_res.frequency, width=covid_res.binsize, alpha=0.5, label='Positive cases')
    plt.bar(scores_noncovid_, noncovid_res.frequency, width=noncovid_res.binsize, alpha=0.5, label='Positive cases')
    plt.legend(loc='upper right')
    plt.title('{} Dataset: AUROC = {}'.format(target, str(np.around(AUROC_avg, decimals=3))))
    plt.xlim([0, 1])
    plt.savefig(ckpt_folder + 'Probability_score_' + target + '.png', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()
    
    save_auc = ckpt_folder + 'ROC_' + target + '.png'
    acu_curve(gt_np[:,COVID_label], pred_np[:,COVID_label], save_auc)
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for ij in range(len(gt_np)):
        if gt_np[ij,0] == 1: # covid
            if pred_np[ij, 0] > thres:
                TP = TP + 1
            else:
                FN = FN +1 
                
        else:  # non-covid 
            if pred_np[ij,0] < thres:
                TN = TN+1
            else:
                FP = FP +1 
                
    print('True Positive', TP)
    print('False Negative', FN)
    print('True Negative', TN)
    print('False Positive', FP)
    print('sensitivity', TP/(TP+FN))
    print('spercificty', TN/(TN+FP))
    
    alpha = 0.95
    auc_result, auc_cov = delong_roc_variance(gt_np[:,COVID_label], pred_np[:,COVID_label])
    
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    
    ci = scipy.stats.norm.ppf(
    lower_upper_q,
    loc=auc_result,
    scale=auc_std)
    ci[ci > 1] = 1
    
    approxi_digit = 3
    print('AUC:', round(auc_result, approxi_digit))
    ci_round = [round(ci[0], approxi_digit), round(ci[1], approxi_digit)]
    print('95% AUC CI:', ci_round)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    test_csv_list = ['SP'] 
    #test_model_list = ['Swin', 'VGG16', 'Convnext', 'DenseNet', 'SqueezeNet', 'EfficientNet'] 
    test_model_list = ['DenseNet']
    for test_csv in test_csv_list:
        for test_model in test_model_list:
            csv_path = './csv/{}_extra_info_opacity.csv'.format(test_csv) 
            df = pd.read_csv(csv_path)
            #df = df[(df.delta<=7) & (df.delta>=-7) & (df.Opacity_score_mimic>0.2)]
            df = df[(df.delta<=7) & (df.delta>=-7)] 
            root_dir = './Dataset/{}/'.format(test_csv) 
            pred_ensemble, gt, auc, ci = main(root_dir, test_model, df, 5)
            #plot_roc_curve(gt, pred_ensemble, 'Results/test_{}_{}_roc.png'.format(test_model, test_csv))
            
            df_test_new = pd.DataFrame({'Filename':df.Filename, 'label_positive':gt, 'score_positive': pred_ensemble})
            csv_save_file = 'Results/test_{}_{}_roc.csv'.format(test_model, test_csv)
            df_test_new.to_csv(csv_save_file, index=False)
    
    
