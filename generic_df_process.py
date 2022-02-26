# read libraries
from concurrent.futures import process
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image
import pydotplus
from graphviz import Digraph
from dtreeviz.trees import *
import graphviz
import seaborn as sns
from six import StringIO
import sklearn
from sklearn.datasets import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn import model_selection, preprocessing,linear_model,tree,metrics
from sklearn.model_selection import train_test_split, KFold,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score,roc_curve, auc, accuracy_score,f1_score,mean_squared_error

# read data
class processDataFrame:
    'read excel and create a data frame'
    def __init__(self,df):
        'initialize parameters'
        self.data = df
    
    def drop(self,del_col_lst):
        'drop columns'
        self.data = self.data.drop(del_col_lst,axis = 1)

    def filter(self,filter_cond):
        'filter the df with the condition'
        self.data = self.data.query(filter_cond)
    
    def change_type(self,col,data_type):
        'change data type'
        self.data[col] = self.data[col].astype(data_type)

    def set_index(self,index_col):
        'set a column index'
        self.data = self.data.set_index(index_col)

    def pick_cols(self,new_cols_lst):
        'pick columns'
        self.data = self.data.reindex(columns = new_cols_lst)

    def drop_duplicates(self,col_lst):
        'drop_duplicates'
        self.data = self.data.drop_duplicates(col_lst)

    def dropna(self,col_lst):
        'drop na from the cols'
        self.data = self.data.dropna(subset = col_lst)

    def change_col_names(self,cols_dict):
        'change column names'
        self.data = self.data.rename(cols_dict,axis = 1)

    def fill_na(self,col,fill):
        'fill na with a specific value'
        self.data[col] = self.data.loc[:,(col)].fillna(fill)

    def print_shape(self):
        'print the shape of the data frame'
        print('(行,列) = ',self.data.shape)
        
    def dummyCols(self,colLst):
        'return the dataframe with dummies variables'
        return pd.get_dummies(data = self.data,columns = colLst)

class excelDataFrame(processDataFrame):
    'read excel and create a data frame'
    def __init__(self,file_name,sheet_name = 0):
        'initialize parameters'
        self.data = pd.read_excel(file_name,sheet_name = sheet_name)

class ML:
    'build a tree ensemble model'

    def __init__(self,dataset,independent_list,dependent_col,testSize = .3):
        'init params'
        self.dataset = dataset    
        self.features = self.dataset.loc[:,independent_list]
        self.dependent = self.dataset[dependent_col]
        self.feature_names = independent_list
        (self.train_X, self.test_X ,self.train_y, self.test_y) = train_test_split(self.features, self.dependent, test_size = testSize, random_state = 665)

    def GS(self,clf,params_dict,fold =10): 
        '''
        grid search
        (how to use) 
        GS(
            RandomForestClassifier(),
            {'n_estimators':[1,2,3,4,5],'max_depth':[2,3,4,5],'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10],'random_state':[0]}
            )
        '''
        self.grid_search = GridSearchCV(clf,  # 分類器を渡す
                                param_grid=params_dict,  # 試行してほしいパラメータを渡す
                                cv=fold,  # 10-Fold CV で汎化性能を調べる
                                )
        self.grid_search.fit(self.train_X, self.train_y)
        print(self.grid_search.best_score_)  # 最も良かったスコア
        print(self.grid_search.best_params_)  # 上記を記録したパラメータの組み合わせ
        self.clf = self.grid_search.best_estimator_ # update the clf by the best model
        return pd.DataFrame(self.grid_search.cv_results_).T

    def getClf(self):
        'return clf'
        return self.clf

    def fit(self):
        'fit the model'
        self.clf.fit(self.train_X,self.train_y)

    def CV(self,cvN):
        'cross validation'
        scores = model_selection.cross_val_score(self.clf, self.test_X, self.test_y, cv=cvN)
        print('cv_score:{}'.format(scores.mean()))
        print('Train score: {}'.format(self.clf.score(self.train_X, self.train_y)))
        print('Test score: {}'.format(self.clf.score(self.test_X,self.test_y)))
        print('Confusion matrix:\n{}'.format(confusion_matrix(self.test_y, self.clf.predict(self.test_X))))
        print (classification_report(self.test_y, self.clf.predict(self.test_X))) 
        return scores

    def AUC(self):
        '''
        AUC
        FPR, TPR(, しきい値) を算出
        '''
        fpr, tpr, thresholds = metrics.roc_curve(self.test_y, self.clf.predict(self.test_X))
        auc = metrics.auc(fpr, tpr)
        #プロット
        plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)

    def predNew(self,datasetNew,independent_list,dependent_col,proba = False):
        'predict new records using the model'
        self.datasetNew = datasetNew    
        self.featuresNew = self.datasetNew.loc[:,independent_list]
        self.dependentNew = self.datasetNew[dependent_col]
        self.feature_namesNew = independent_list

        print(self.featuresNew.shape)
        pred_count = sum(self.clf.predict(self.featuresNew))
        pred_time = len(clf.predict(self.featuresNew))
        if proba == True:
            print (clf.predict_proba(self.featuresNew))
        print('pred result:\n   {}'.format(clf.predict(self.featuresNew)))
        print('pred score:\n   {}'.format(1-(pred_count/pred_time)))
        print('------------')

    def recursive_elim(self,N):
        '''
        RECURSIVE FEATURE ELIMINATION
        This function can be only applied to the clf 
        to have 'coef_' or 'feature_importances_'
        '''
        selector = RFE(estimator = self.clf, n_features_to_select = N, verbose = 1)
        selector.fit(self.train_X,self.train_y)
        mask = selector.get_support()
        print(feature_names)
        print(mask)
        print(selector.ranking_)

        # 選択した特徴量の列のみ取得
        X_selected = selector.transform(self.train_X)
        print("X.shape={}, X_selected.shape={}".format(self.train_X.shape, X_selected.shape))
        rfe = pd.DataFrame(selector.transform(self.train_X), columns=self.train_X.columns.values[selector.support_])
        return self.dataset.reindex(columns=rfe.columns)

    def vizDTC(self,depth):
        'vizualize the decision tree classifier'
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(self.train_X, self.train_y)
        pred = clf.predict(self.test_X)
        dot_data = StringIO()
        dot = Digraph(format='png')
        # フォント設定
        dot.attr('node', fontname="IPAGothic")
        tree.export_graphviz(clf, out_file=dot_data,feature_names=self.train_X.columns, max_depth=depth)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # try:
        graph.write_pdf("graph.pdf")
        print(pred)
        return Image(graph.create_png())
        # except:
        #     env_path = input('You need to set the environment PATH of dot.exe. Please enter the absolute PATH here: ')
        #     graph.progs = {'dot':env_path}
        #     print('Now you completed to set the PATH. Please run this function again.')

    def simulation(self,N):
        'motecalro simulation'    
        N = 5000
        # シミュレーションデータの作成
        # initialize the np.array with zeros
        simulation_data = np.zeros(
            (N,self.train_X.shape[1])
            )

        # update the array with the values following normal dist.
        for i in range(self.train_X.shape[1]):
            simulation_data[:,i]= np.random.normal(
                np.array(self.train_X)[:,i].mean()
                ,np.array(self.train_X)[:,i].std()
                ,N)
                
        # 上記のシミュレーションデータを使って予測モデルに投入します。
        y_pred_proba = self.clf.predict_proba(simulation_data)

        # 1列目の変動が目的変数に与える影響を理解するためにグラフにします。
        for i in range(self.train_X.shape[1]):
            plt.plot(simulation_data[:, i], y_pred_proba[:,i])

    def stepwise_elim(self):
        'forward-backward stepwise feature selection'
        
        #どの特徴量が何列目にあるかを辞書型で保持 features は列名を保持したnp.arrayの一次配列
        feature_indices = {feature: idx for idx, feature in enumerate(self.features.columns)}
    
        # 特徴量をユニークにする。仮に同じ名前の列名があった場合、重複を削除している
        features = set(self.features.columns)
        
        # 評価（MSE）の初期化
        last_mse = np.inf
        
        #選ばれた特徴量を保存するための空集合を用意
        chosen_features = set()

        while len(chosen_features) < len(features):
            mse_features = []
            
            # 集合は引き算すると差集合
            for feature in (features - chosen_features): # 最初は、chosen_featuresは０だから、featuresの数だけ繰り返される
                candidates = chosen_features.union(set([feature])) #union()は「または」を示す。故にcandidatesは、featurekかまたはchose_features
                indices = [feature_indices[feature] for feature in candidates] #candidatesの中にある列名を一つずつ取り出し、列番号を取得
                
                self.clf.fit(self.train_X.iloc[:, indices], self.train_y)
                y_pred = self.clf.predict(self.test_X.iloc[:, indices])
                mse = mean_squared_error(self.test_y, y_pred)
                mse_features += [(mse, feature)]

            mse, feature = min(mse_features, key= lambda x:x[0])
            
            if mse >= last_mse:
                break
            last_mse = mse
            print('Newly Added Feature: {},\tRMSE Score: {}'.format(feature, np.sqrt(mse)))
            chosen_features.add(feature)
        return [feature_indices[feature] for feature in chosen_features]

# sample data -- for debugging
# import pydataset
# dataset = pydataset.data('Caschool')
# datasetA = dataset.iloc[:100,]
# datasetB = dataset.iloc[101:,]

if __name__ == "__main__":
    from concurrent.futures import process
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython.display import Image
    import pydotplus
    from graphviz import Digraph
    from dtreeviz.trees import *
    import graphviz
    import seaborn as sns
    from six import StringIO
    import sklearn
    from sklearn.datasets import *
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE
    from sklearn import model_selection, preprocessing,linear_model,tree,metrics
    from sklearn.model_selection import train_test_split, KFold,GridSearchCV
    from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score,roc_curve, auc, accuracy_score,f1_score,mean_squared_error
