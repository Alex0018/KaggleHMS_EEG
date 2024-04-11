import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

from src.styles import set_styles, PALETTE, TXT_ACC, TXT_RESET

import warnings
warnings.filterwarnings('ignore')


# ---- REPRODICIBILITY ------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)






class Solution:

    def __init__(self, model, model_state_path):
        self.model = model

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(model_state_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device=self.device)
        self.model.eval()

        self.target_columns = ['seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']


    def predict_validation(self, dataset, batch_size=32):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = []
        loss = 0.
        with torch.no_grad():
            for b, batch in enumerate(pbar := tqdm(dataloader)):
                data = [batch[i].to(device=self.device) for i in range(len(batch) - 1)]
                target = batch[-1].to(device=self.device)
                output = self.model(*data)
                output = F.log_softmax(output, dim=1)
                
                preds.append(output.detach().cpu().numpy())

                cur_loss = torch.nn.KLDivLoss(reduction='batchmean')(output, target)
                loss += cur_loss

                pbar.set_description(f'validation batch:    batch loss {cur_loss: .5f}     mean loss {loss / (b+1): .5f}')
                                
        return pd.DataFrame(np.concatenate(preds, axis=0), columns=self.target_columns)
    

    def predict(self, dataset, batch_size=32):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for i, batch in enumerate(pbar := tqdm(dataloader)):
                data = batch.to(device=self.device) 
                output = self.model(data)
                output = F.softmax(output, dim=1)
                preds.append(output.detach().cpu().numpy())

        return pd.DataFrame(np.concatenate(preds, axis=0), columns=self.target_columns)
    








class OOF:

    @staticmethod
    def analyze_folds(df, fold_idx, oofs, label=''):
        for fold in range(len(oofs)):
            # print('-'*100)
            # print(' '*40, f'{label.upper()} FOLD {fold}')
            # print('-'*100)

            df_fold = df.loc[ fold_idx[fold][1] ].reset_index(drop=True)
            num_annotators = df_fold.iloc[:, -6:].sum(axis=1)
            for i in range(1,7):
                df_fold.iloc[:, -i] /= num_annotators

            OOF.display_analysis_by_label(df_fold, oofs[fold], title=f'     {label.upper()} FOLD {fold}     ')




    @staticmethod
    def get_scores(targets, preds):
        scores = []
        for i in range(len(preds)):
            s = torch.nn.KLDivLoss(reduction='sum')(torch.Tensor(preds[i]), 
                                                    torch.Tensor(targets[i]))



            scores.append(s.detach().numpy())
        return scores
    

    @staticmethod
    def display_stats_dataframe(df_stats):
        format_dict = {col: "{:.2f}" for col in df_stats.columns[:4]}
        format_dict.update({col: "{:.0f}" for col in df_stats.columns[4:]})
        mean_style = {'selector': 'th:not(.index_name), td:not(.index_name)',
                      'props': [('background-color', '#d3d3d0')]}
        count_style = {'selector': 'th, td',
                       'props': [('background-color', '#f1f1f1')]}
        index_row_style = {
                        'selector': '.index_name.level0',
                        'props': 'font-weight:bold;'
                    }
        index_columns_style = {
                        'selector': '.index_name.level1',
                        'props': 'font-style: italic; font-weight:bold;'
                    }
        column_header_style = [
            {'selector': 'th.col_heading', 'props': 'text-align: center; font-style: italic; font-weight:normal;'},
            {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.2em'},
        ]
        display(df_stats.style\
                        .format(format_dict) \
                        .set_table_styles([index_row_style, index_columns_style, *column_header_style]) \
                        .set_table_styles({col: [mean_style]  for col in df_stats.columns[:4]}, overwrite=False) \
                        .set_table_styles({col: [count_style] for col in df_stats.columns[4:]}, overwrite=False) \
                        )



    @staticmethod
    def get_score_distributions(df, scores, num_examples=4):
        df_preds = df.copy()
        df_preds['score'] = scores

        df_preds = df_preds.sort_values('score')
        best_rows = df_preds.index[:num_examples]
        worst_rows = df_preds.index[-num_examples:]

        df_stats =  df_preds.groupby(['expert_consensus', 'target_status'])['score'] \
                            .agg(['mean', 'count']) \
                            .unstack() \
                            .fillna(0)

        return df_stats, best_rows, worst_rows
    

    
    @staticmethod
    def display_analysis_by_label(df, oof, title):
        scores = OOF.get_scores(df.iloc[:, -6:].values, oof.iloc[:,:6].values)
        
        # _, ax = plt.subplots(1,1,figsize=(5, 0.25))
        # ax.text(0,0, f'\nMean score:   {np.mean(scores):.3f}\n', fontsize=20, fontweight='bold')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        _, ax = plt.subplots(1,1,figsize=(10, 0.25))
        ax.text(0,0, title, fontsize=24, fontweight='bold', 
                horizontalalignment='center', backgroundcolor='#d3d3d0')
        ax.text(-0.2, -4, f'\nMean score:   {np.mean(scores):.3f}\n', fontsize=18,)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        n_examples = 5
        df_stats, best, worst = OOF.get_score_distributions(df, scores, num_examples=n_examples)
        
        print(f'{TXT_ACC} Score statistics {TXT_RESET}')
        OOF.display_stats_dataframe(df_stats)

        df_full = pd.concat([df.reset_index(drop=True), oof.rename(columns={col:col+'_pred' for col in oof.columns[:6]})], axis=1)
        df_full['score'] = scores

        # display errors ----------------------------------------------------------------
        labels = ['Seizure','LPD','GPD','LRDA','GRDA','Other']
        num_preds_score_greater3 = np.zeros(6)
        num_ideal = np.zeros(6)
        num_preds_sure = np.zeros(6)
        num_preds_sure_but_wrong = np.zeros(6)

        for i, (col, label) in enumerate(list(zip(oof.columns[:6], labels))):
            # print(f'{TXT_ACC} {label} {TXT_RESET}')
            df_tmp = df_full.loc[(df_full['target_status'] == 'ideal') & (df_full['expert_consensus'] == label)]
            num_preds_score_greater3[i] = df_tmp.loc[df_tmp['score'] > 3].count()[0]
            num_ideal[i] = df_tmp.shape[0]

            df_tmp = df_full.loc[(np.exp(df_full[col + '_pred']) > 0.9)]
            num_preds_sure[i] = df_tmp.shape[0]
            num_preds_sure_but_wrong[i] = df_full.loc[(np.exp(df_full[col + '_pred']) > 0.9) \
                                                      & (df_full['expert_consensus'] != label)].count()[0]
            
        # diplay counts of incorrect predictions
        print(f'\n\n{TXT_ACC} Number of incorrect predictions {TXT_RESET}')
        df_analysis = pd.DataFrame([num_preds_score_greater3, num_preds_score_greater3 / num_ideal, 
                                    num_preds_sure, num_preds_sure_but_wrong, num_preds_sure_but_wrong/num_preds_sure], 
                                   index=['score >3', '%% score >3', 'predicted vote >90%', 'wrongly predicted vote >90%', '%% wrongly predicted vote >90%'],
                                   columns=labels).T.fillna(0)
        formats = ['{:.0f}', '{:.2%}', '{:.0f}', '{:.0f}', '{:.2%}']
        display(df_analysis.style \
                           .format({df_analysis.columns[i]: formats[i] for i in range(5)}) \
                           .set_table_styles([{'selector': 'th.col_heading', 'props': 'text-align: center; width:5em;'}]))
            
        # display best predictions ------------------------------------------------------
        print(f'\n\n{TXT_ACC} Best predictions {TXT_RESET}')
        _, axes = plt.subplots(nrows=1, ncols=n_examples, figsize=(n_examples*4,2))
        axes = axes.ravel()
        for i, row in enumerate(best):
            OOF.display_distributions(df, oof, row, axes[i])
        plt.tight_layout()
        plt.show()

        # display worst predictions -----------------------------------------------------
        print(f'\n{TXT_ACC} Worst predictions {TXT_RESET}')
        _, axes = plt.subplots(nrows=1, ncols=n_examples, figsize=(n_examples*4,2))
        axes = axes.ravel()
        for i, row in enumerate(worst):
            OOF.display_distributions(df, oof, row, axes[i])
        plt.tight_layout()
        plt.show()

        # display confusion matrix ------------------------------------------------------
        print(f'\n{TXT_ACC} Confusion matrix for ideal cases {TXT_RESET}')
        OOF.display_confusion_matrix(df, oof)
        print('\n\n\n')


    
    @staticmethod
    def _get_palette():
        palette  = ('#f1f1f1', '#d3d3c5', '#8d8d7c', '#5f5f47') #, '#ffffff')
        boundaries = [0, 0.01, 0.2, 1]  
        custom_color_map = LinearSegmentedColormap.from_list(boundaries, palette)
        return custom_color_map

    @staticmethod
    def _show_confusion_matrix(Y, preds):
        labels = ['Seizure','LPD','GPD','LRDA','GRDA','Other']
        cm = confusion_matrix(Y, preds, labels=labels)
        
        N = Y.nunique()
        plt.figure(figsize=(0.7*N, 0.35*N))
        heatmap = sns.heatmap(cm, vmin=0, annot_kws={'alpha': 0.5}, fmt='.5g',
                            annot=True, cmap=OOF._get_palette(), cbar=False)
        heatmap.set_xlabel('predicted')    
        heatmap.set_ylabel('true')
        heatmap.set_yticklabels(labels)
        heatmap.set_xticklabels(labels)
        for label in heatmap.get_yticklabels():
            label.set_rotation(0)
        
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{preds.name}.png')
        plt.show()



    @staticmethod
    def display_confusion_matrix(df, oof):

        cols = ['Seizure','LPD','GPD','LRDA','GRDA','Other']

        df_preds = df[['target_status', 'expert_consensus']]
        df_preds['pred'] = pd.Series(np.argmax(oof.iloc[:, :6], axis=1)).map({i:cols[i]  for i in range(6)})
        
        df_preds = df_preds.loc[df_preds['target_status'] == 'ideal']

        OOF._show_confusion_matrix(df_preds['expert_consensus'], df_preds['pred'])

    
    @staticmethod
    def display_distributions(df, oof, row, ax=None):

        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(5,1.5))

        data1 = df.iloc[row, -6:].values.astype('float32')
        data2 = np.exp(oof.iloc[row, :6].values.astype('float32'))

        eeg_id = df.iloc[row]['eeg_id']
        score = torch.nn.KLDivLoss(reduction='sum')(torch.Tensor(oof.iloc[row, :6].values),
                                                    torch.Tensor(data1))

        ax.bar(height=data1, x=np.arange(6)-0.2, width=0.3, label='true')
        ax.bar(height=data2, x=np.arange(6)+0.2, width=0.3, label='pred')

        ax.set_xticks(np.arange(6), ['seizure','lpd','gpd','lrda','grda','other'])

        ax.legend(bbox_to_anchor=(1, 0.9, 0, 0), loc='upper left')
        ax.set_title(f'ID {eeg_id}:   score {score:.2f}', fontsize=10)

        ax.yaxis.grid(True)













class SolutionOnFolds:

    def __init__(self, model_class, model_params, model_paths, dataset_class, dataset_params={}):
        self.solutions = []
        for fold in range(len(model_paths)):
            model = model_class(**model_params)
            self.solutions.append( Solution(model, model_paths[fold]) )

        self.dataset_class = dataset_class
        self.dataset_params = dataset_params
        self.oofs = None

    def predict_on_fold(self, df_fold, fold):

        dataset = self.dataset_class( df_fold , **self.dataset_params)
        df_pred = self.solutions[fold].predict_validation(dataset, batch_size=32)

        df_pred['fold'] = fold
        df_pred['eeg_id'] = df_fold['eeg_id']
        return df_pred
    
    def predict_OOF(self, df, fold_idx, path_save=None):
        oofs = []
        for fold in range(len(fold_idx)):
            print(f'{TXT_ACC} Fold {fold} {TXT_RESET}')
            df_fold = df.loc[ fold_idx[fold][1] ].reset_index(drop=True)
            oof = self.predict_on_fold(df_fold, fold)            
            oofs.append(oof)

        if path_save is not None:
            pd.concat(oofs, axis=0).to_csv(path_save, index=False)

        self.oofs = oofs
