# 多样性建模 ---- 三个数据不足的因素，缺失值补足

from heamy.dataset import Dataset
from heamy.estimator import Regressor
from heamy.pipeline import ModelsPipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from itertools import combinations
from sklearn import metrics
import numpy as np
import pandas as pd
import itertools
import math
import random
import joblib


# create dataset
def DataCleanRA(data):
	#data = data.drop('time', axis=1)
	#RA anslysis
	data = data.dropna()  #delete all rows with NaN
	data = data.drop(data[(data.times <= 5)].index)
	data = data.drop(data[(data.rate <= 1.5)].index)
	data = data[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T', 'diversity']]
	print(data)
	#data['lat'] = data['lat'].map(lambda x: math.sin(math.radians(x)))
	#data['lon'] = data['lon'].map(lambda x: math.cos(math.radians(x)))
	X_train, X_test, y_train, y_test = Train_Test(data)
	return X_train, X_test, y_train, y_test


def Train_Test(data):
	n = range(len(data))
	data.index = range(len(data))
	test_size = 0.1
	random.seed(1)
	test_n = random.sample(n,int(test_size*len(data)))
	test = data.loc[test_n]
	train_n = []
	for i in range(len(data)):
		if i in test_n:
			i = i
		else: train_n.append(i)
	train = data.loc[train_n]
	X_train = train.values[:,:-1]
	y_train = train.values[:,-1]
	X_test = test.values[:,:-1]
	y_test = test.values[:,-1]
	return X_train, X_test, y_train, y_test

def StackModel(X_train, y_train, X_test):
	dataset = Dataset(X_train,y_train,X_test)
	#build models
	model_rf = Regressor(dataset=dataset, estimator=RandomForestRegressor, parameters={'n_estimators': 1000, 'random_state' : 1, 'n_jobs':8, 'max_depth':26},name='rf')
	model_et = Regressor(dataset=dataset, estimator=ExtraTreesRegressor, parameters={'n_estimators': 1000, 'max_depth':26, 'random_state':10},name='et')
	model_knn = Regressor(dataset=dataset, estimator=KNeighborsRegressor, parameters={'n_neighbors': 25},name='knn')
	model_gb = Regressor(dataset=dataset, estimator=GradientBoostingRegressor, parameters={'n_estimators': 1000, 'learning_rate':0.10, 'max_depth':26,'random_state':4},name='gb')
	model_bg = Regressor(dataset=dataset, estimator=BaggingRegressor, parameters={'n_estimators':1000, 'max_samples':500, 'max_features': 1.0,'random_state' : 1}, name='bg')
	# Stack three models
	# Returns new dataset with out-of-fold predictions
	pipeline = ModelsPipeline(model_rf,model_et,model_knn, model_gb, model_bg)
	stack_ds = pipeline.stack(k=10,seed=11)
	#stacking
	stacker = Regressor(dataset=stack_ds, estimator=BaggingRegressor, parameters={'n_estimators':500, 'max_samples':500, 'max_features': 1.0,'random_state' : 2})
	results = stacker.predict()
	#joblib.dump(stacker, 'stacker_total_PRED_ra.model')
	return results

def Validate(results):
	#results_rmse = stacker.validate(k=10,scorer=mean_absolute_error)
	#results_r2 = stacker.validate(k=10,scorer=r2_score)
	r2 = (pd.Series(y_test)).corr(pd.Series(results))
	rmse = np.sqrt(metrics.mean_squared_error(y_test, results))
	return r2, rmse

def permutation_importances(X_train, y_train, X_test, metric):
	baseline = metric.copy()
	X_test0 = pd.DataFrame(X_test)
	save = X_test0.copy() 
	dd = {}
	for col in X_test0.columns: 
		imp_rmse = []
		for n in range(3): 
			print(n)
			np.random.seed(n)
			X_test0[col] = np.random.permutation(X_test0[col])
			r20, rmse0 = Validate(StackModel(X_train, y_train, X_test0))
			X_test0 = save.copy() 
			imp_rmse.append((rmse0-baseline)/baseline)
			print(imp_rmse)
		dd[col] = imp_rmse
	dd = pd.DataFrame(dd)
	return dd


def ICE_caculate_save(X_train, X_test, save_name):
	col_names = ('silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T')
	X_test = pd.DataFrame(X_test)
	X_train = pd.DataFrame(X_train)
	X_data = pd.concat([X_train, X_test], axis=0)
	X_data.columns = col_names
	#X_data.to_csv('D:/%s_ice_1112_data_%s.csv'%(save_name, 'X_data'))
	col_names = list(col_names)
	for col_i in col_names:
		save = X_data.copy()
		col_du = X_data[col_i]
		col_du = pd.DataFrame(col_du)
		col_du = col_du.drop_duplicates()
		col_du = col_du[col_i].tolist()
		save_i = pd.DataFrame()
		if len(col_du) < 20:
			for x in col_du:
				print('===============original X_data===================')
				print(X_data)
				print('================changed X_data==================')
				X_data[col_i].loc[X_data[col_i] > -1000] =x
				print(X_data)
				print('==================================')
				X_data = X_data.values
				pred = StackModel(X_train, y_train, X_data)
				pred = pd.DataFrame(pred)
				save_i = pd.concat([save_i, pred], axis=1)
				X_data = save.copy()
			save_i.columns = col_du
		else:
			col_du = np.array(col_du)
			col_du_perc = []
			for i in [5,15,25,35,45,65,75,85,95,100]:
				col_du_perc.append(np.percentile(col_du, i))
			for m in col_du_perc:
				print('===============original X_data===================')
				print(X_data)
				print('================changed X_data==================')
				X_data[col_i].loc[X_data[col_i] > -1000] = m
				print(X_data)
				X_data = X_data.values
				pred = StackModel(X_train, y_train, X_data)
				pred = pd.DataFrame(pred)
				save_i = pd.concat([save_i, pred], axis=1)
				X_data = save.copy()
			save_i.columns = col_du_perc
			print(save_i)
		save_i.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/%s_ice_%s.csv'%(save_name, col_i))
		X_data = save.copy()


def ICE_caculate(X_train, X_test, save_name):
	col_names = ('silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T')
	X_test = pd.DataFrame(X_test)
	X_train = pd.DataFrame(X_train)
	X_data = pd.concat([X_train, X_test], axis=0)
	X_data.columns = col_names
	X_data.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/%s.ice.Xdata_%s.csv'%(save_name, 'X_data'))
	col_names = list(col_names)
	ana_v = ['CO2','T']
	for col_i in list(combinations(ana_v,2)):
		save = X_data.copy()
		col_1 = X_data[col_i[0]]
		col_2 = X_data[col_i[1]]
		col_1 = pd.DataFrame(col_1)
		col_2 = pd.DataFrame(col_2)
		col_1 = col_1.drop_duplicates()
		col_2 = col_2.drop_duplicates()
		col_1 = col_1[col_i[0]].tolist()
		col_2 = col_2[col_i[1]].tolist()
		col_1 = np.array(col_1)
		col_2 = np.array(col_2)
		col_1_perc = []
		col_2_perc = []
		for i in range(20):
			col_1_perc.append(np.percentile(col_1, (i+1)*5))
			col_2_perc.append(np.percentile(col_2, (i+1)*5))
		m_save = {}
		for m in col_1_perc:
			save_i = pd.DataFrame()
			n_save = []
			for n in col_2_perc:
				print('===============original X_data===================')
				print(X_data)
				print('================changed X_data==================')
				X_data[col_i[0]].loc[X_data[col_i[0]] > -1000] = m
				X_data[col_i[1]].loc[X_data[col_i[1]] > -1000] = n
				print(X_data)
				print("====CO2, T====")
				print(X_data[['CO2','T']])
				X_data = X_data.values
				pred = StackModel(X_train, y_train, X_data)
				pred = pd.DataFrame(pred)
				pred.columns = [n]
				save_i = pd.concat([save_i, pred], axis=1)
				n_save.append(n)
				X_data = save.copy()
			m_save[m] = n_save
			save_i.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/%s_%s_ice_0112_%s.csv'%(save_name,col_i,m))
			print(save_i)
		m_save.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/%s_ice_mn0112_%s.csv'%(save_name, col_i))


def shap(data,num):
	cols = ['silicate', 'nitrate', 'phosphate','NPP', 'salinity']
	total_col = ['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T']
	for i in itertools.permutations(cols, 5):
		print(i)
		p1 = data[total_col].copy()
		save1 = 0
		shap_v = []
		shap_get = pd.DataFrame()
		for ii in i:
			save2 = save1
			col_ii = ii+num
			print('original',p1[ii])
			print('changed',data[col_ii])
			p1[ii] = data[col_ii]
			print(p1)
			pv = p1.values
			p_result = StackModel(X_train, y_train, pv)
			save1 = ave_div(p_result,data)
			shap_v.append(save1-save2)
			print(shap_v)
		p1 = data[total_col].copy()
		shap_get[i]=shap_v
	shap_get = pd.DataFrame(shap_get)
	return shap_get

def ave_div(p_result,data):
	ave_d = pd.DataFrame()
	ave_d['latlon'] = data['latitude'].map(str)+'|'+data['longitude'].map(str)
	ave_d['pred'] = p_result
	a = ave_d.groupby('latlon')['pred'].mean()
	get = pd.DataFrame(a)
	return get['pred'].mean()







print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("=======================div_data_class==============================")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")

#data = pd.read_csv("E:/project/div_all/div_all.csv", encoding='unicode_escape')
#data = pd.read_csv("E:/project/div_multi/div_class_model.csv", encoding='unicode_escape')
data = pd.read_csv("E:/project/div_all/div_all_times_rate/new_div_all/div_all_new.csv", encoding='unicode_escape')
data = data[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T', 'diversity', 'times','rate']]
print(data)
X_train, X_test, y_train, y_test = DataCleanRA(data)
results = StackModel(X_train, y_train, X_test)
#data_pt = pd.DataFrame({'y_test':y_test, 'results':results})  #save the prediction with y test
#data_pt.to_csv('C:/python/total_PRED_pt_ra.csv')
r2, rmse = Validate(results)
print("====diversity R2====")
print(r2)
print("====diversity RMSE====")
print(rmse) 
#ICE_caculate_save(X_train, X_test, 'div_all_new')
#imp_test = permutation_importances(X_train, y_train, X_test, rmse)
#imp_test.to_csv('E:/project/div_all/div_all_time_rate_importance_test.csv')
'''
p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')
p1 = p.dropna()
print(p1.columns)
shap_85 = shap(p1,'8.5')
shap_85.to_csv('E:/project/div_all/div_all_times_rate/div_time_rate_shap_8.5.csv')
shap_26 = shap(p1,'2.6')
shap_26.to_csv('E:/project/div_all/div_all_times_rate/div_time_rate_shap_2.6.csv')
'''

#ICE_caculate_save(X_train, X_test, 'div_all_times_rate')


'''
p0 = pd.read_csv("E:/project/div_all/div_all_times_rate/new_div_all/div_new_data_creating_for_2D_plot.csv", encoding='unicode_escape')
p = p0[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T']]
pred = p.values
results_p = StackModel(X_train, y_train, pred)
print(results_p)
p0['diversity'] = results_p
p0.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/多因素/div_代表数据生成_for_2D.csv')



#单因素多因素RCP2.6 RCP8.5
p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity8.5', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_salinity8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity2.6', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_salinity2.6.csv')


p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_T8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_T2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO28.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO22.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO2Tsaliniy8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO2Tsalinity2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO2TNPP8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO2TNPP2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO2Tphosphate8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO2Tphosphate2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO2Tnitrate8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO2Tnitrate2.6.csv')

p1 = p[['silicate8.5', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO2Tsilicate8.5.csv')

p1 = p[['silicate2.6', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_CO2Tsilicate2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity8.5', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_silicate8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity2.6', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_silicate2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP8.5', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_NPP8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP2.6', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_NPP2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate8.5', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_phosphate8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate2.6', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_phosphate2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_nitrate8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_nitrate2.6.csv')

p1 = p[['silicate8.5', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_silicate8.5.csv')

p1 = p[['silicate2.6', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_0821_silicate2.6.csv')
'''

#SHAP
'''
#T, CO2, OTHER
p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')

p1 = p[['silicate2.6', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_onlysili2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_onlysili_ni2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_onlysili_ni_pho2.6.csv')

p1 = p[['silicate8.5', 'nitrate2.6', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_onlyni2.6.csv')

p1 = p[['silicate8.5', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_onlyni_pho2.6.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate2.6', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_onlypho2.6.csv')

p1 = p[['silicate2.6', 'nitrate8.5', 'phosphate2.6', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_onlypho_sili2.6.csv')
'''
'''
p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_shap_T8.5.csv')

#T, OTHER, CO2

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO2', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_shap_Tother2.6.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO2', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_shap_Tother8.5.csv')

#CO2, T, OTHER

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_shap_CO22.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_shap_CO28.5.csv')

#CO2, OTHER, T
p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_shap_CO2other2.6.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_shap_CO2other8.5.csv')

#OTHER, T, CO2

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO2', 'NPP2.6', 'PAR', 'salinity2.6', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_shap_other2.6.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO2', 'NPP8.5', 'PAR', 'salinity8.5', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_shap_other8.5.csv')
'''

'''
p000 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2T_change_CO2.csv",encoding="unicode_escape")
p00 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')
p01 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2Tother_两端10.csv", encoding='unicode_escape')
p02 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2Tother_两端10_1.csv", encoding='unicode_escape')
p03 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2Tother_两端10_2.csv", encoding='unicode_escape')
p04 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2Tother_两端10_3.csv", encoding='unicode_escape')
p100 = p000[['silicate','nitrate','phosphate','NPP','salinity']]
p10 = p00[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CO28.5', 'NPP8.5', 'salinity8.5', 'T8.5','CO22.6','NPP2.6','salinity2.6','T2.6']]
p11 = p01[['CHL','PAR','nitrate02','nitrate04','nitrate06','nitrate08','nitrate10','nitrate15','nitrate20','nitrate30','nitrate40']]
p12 = p02[['NPP02','NPP04','NPP06','NPP08','NPP10','NPP15','NPP20','NPP30','NPP40','phosphate02','phosphate04','phosphate06','phosphate08','phosphate10','phosphate15','phosphate20','phosphate30','phosphate40']]
p13 = p03[['silicate02','silicate04','silicate06','silicate08','silicate10','silicate15','silicate20','silicate30','silicate40','salinity02','salinity04','salinity06','salinity08','salinity10','salinity15','salinity20','salinity30','salinity40']]
p14 = p04[['CO202','CO204','CO206','CO208','CO210','CO215','CO220','CO230','CO240','T02','T04','T06','T08','T10','T15','T20','T30','T40']]
p = pd.concat([p100,p10,p11,p12,p13,p14],axis=1)

print("CO2Tother_10")



p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO202', 'NPP', 'PAR', 'salinity', 'T02']]
print(p1)
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T02other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO204', 'NPP', 'PAR', 'salinity', 'T04']]
p1 = p1.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T04other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO206', 'NPP', 'PAR', 'salinity', 'T06']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T06other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO208', 'NPP', 'PAR', 'salinity', 'T08']]
print(p1)
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T08other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO210', 'NPP', 'PAR', 'salinity', 'T10']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T10other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO215', 'NPP', 'PAR', 'salinity', 'T15']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T15other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO220', 'NPP', 'PAR', 'salinity', 'T20']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T20other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO230', 'NPP', 'PAR', 'salinity', 'T30']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T30other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO240', 'NPP', 'PAR', 'salinity', 'T40']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T40other2018.csv')
'''




'''
p01 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2T_change_CO2.csv", encoding='unicode_escape')
p02 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/change_T.csv", encoding='unicode_escape')
p03 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')
p11 = p01[['CO21','CO22','CO23','CO24','CO25','CO2_6','CO27','CO28','CO29','T2.6','T4.5','T6','T8.5']]
p12 = p02[['T1','T2','T3','T4','T5','T_6','T7','T8','T9','CO22.6','CO24.5','CO26','CO28.5']]
p13 = p03[['CHL','NPP','nitrate','PAR','phosphate','salinity','silicate','nitrate2.6','nitrate8.5','NPP2.6','NPP8.5','phosphate2.6','phosphate8.5','salinity2.6','salinity8.5','silicate2.6','silicate8.5']]
p = pd.concat([p11,p12,p13],axis=1)

print("CO2Tother2018")
p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T2.6']]
print(p1)
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T2.6other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO21', 'NPP', 'PAR', 'salinity', 'T1']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T1other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22', 'NPP', 'PAR', 'salinity', 'T2']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T2other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO23', 'NPP', 'PAR', 'salinity', 'T3']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T3other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO24', 'NPP', 'PAR', 'salinity', 'T4']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T4other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO25', 'NPP', 'PAR', 'salinity', 'T5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T5other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2_6', 'NPP', 'PAR', 'salinity', 'T_6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T_6other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO27', 'NPP', 'PAR', 'salinity', 'T7']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T7other2018.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28', 'NPP', 'PAR', 'salinity', 'T8']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T8other2018.csv')


p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO29', 'NPP', 'PAR', 'salinity', 'T9']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T9other2018.csv')



print("CO2Tother2.6")

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO21', 'NPP2.6', 'PAR', 'salinity2.6', 'T1']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T1other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO22', 'NPP2.6', 'PAR', 'salinity2.6', 'T2']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T2other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO23', 'NPP2.6', 'PAR', 'salinity2.6', 'T3']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T3other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO24', 'NPP2.6', 'PAR', 'salinity2.6', 'T4']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T4other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO25', 'NPP2.6', 'PAR', 'salinity2.6', 'T5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T5other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO2_6', 'NPP2.6', 'PAR', 'salinity2.6', 'T_6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T_6other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO27', 'NPP2.6', 'PAR', 'salinity2.6', 'T7']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T7other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO28', 'NPP2.6', 'PAR', 'salinity2.6', 'T8']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T8other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO29', 'NPP2.6', 'PAR', 'salinity2.6', 'T9']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T9other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO24.5', 'NPP2.6', 'PAR', 'salinity2.6', 'T4.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T4.5other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO26', 'NPP2.6', 'PAR', 'salinity2.6', 'T6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T6other2.6.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO28.5', 'NPP2.6', 'PAR', 'salinity2.6', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T8.5other2.6.csv')



print("CO2Tother8.5")
p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO22.6', 'NPP8.5', 'PAR', 'salinity8.5', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T2.6other8.5.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO21', 'NPP8.5', 'PAR', 'salinity8.5', 'T1']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_CO2T1other8.5.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO22', 'NPP8.5', 'PAR', 'salinity8.5', 'T2']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T2other8.5.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO23', 'NPP8.5', 'PAR', 'salinity8.5', 'T3']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T3other8.5.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO24', 'NPP8.5', 'PAR', 'salinity8.5', 'T4']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T4other8.5.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO25', 'NPP8.5', 'PAR', 'salinity8.5', 'T5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T5other8.5.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO2_6', 'NPP8.5', 'PAR', 'salinity8.5', 'T_6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T_6other8.5.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO27', 'NPP8.5', 'PAR', 'salinity8.5', 'T7']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T7other8.5.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO28', 'NPP8.5', 'PAR', 'salinity8.5', 'T8']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T8other8.5.csv')


p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO29', 'NPP8.5', 'PAR', 'salinity8.5', 'T9']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T9other8.5.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO24.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T4.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T4.5other8.5.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO26', 'NPP8.5', 'PAR', 'salinity8.5', 'T6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T6other8.5.csv')
'''



'''
p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2T_1222.csv", encoding='unicode_escape')
p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_2018.csv')
'''

'''
p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T4.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_T4.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_T6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO24.5', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO24.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO26', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO26.csv')
'''



'''
p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2T_1222.csv", encoding='unicode_escape')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO24.5', 'NPP', 'PAR', 'salinity', 'T4.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T4.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO26', 'NPP', 'PAR', 'salinity', 'T6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2T8.5.csv')
'''






'''
p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsilinitratephoNPPsal2.6.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsilinitratephoNPPsal8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
print(p1)
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TnitratephoNPPsal2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TnitratephoNPPsal8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TphoNPPsal2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TphoNPPsal8.5.csv')


p1 = p[['silicate', 'nitrate', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TphoNPPsal2.6.csv')


p1 = p[['silicate', 'nitrate', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TphoNPPsal8.5.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsalinitrateNPPsal2.6.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsalinitrateNPPsal8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TnitrateNPPsal2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TnitrateNPPsal8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TNPPsal2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TNPPsal8.5.csv')

p1 = p[['silicate2.6', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsiliNPPsal2.6.csv')

p1 = p[['silicate8.5', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsiliNPPsal8.5.csv')

p1 = p[['silicate2.6', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsiliNPPsal2.6.csv')

p1 = p[['silicate8.5', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsiliNPPsal8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TNPPsal2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TNPPsal8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TNPPsal2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TNPPsal8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TnitratephoNPP2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TnitratephoNPP8.5.csv')

p1 = p[['silicate2.6', 'nitrate', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsiliphoNPP2.6.csv')

p1 = p[['silicate8.5', 'nitrate', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsiliphoNPP8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TphoNPP2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TphoNPP8.5.csv')

p1 = p[['silicate2.6', 'nitrate', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsiliphoNPP2.6.csv')

p1 = p[['silicate8.5', 'nitrate', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsiliphoNPP8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TphoNPP2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TphoNPP8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TnitrateNPP2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TnitrateNPP8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TnitrateNPP2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TnitrateNPP8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TNPP2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TNPP8.5.csv')

p1 = p[['silicate2.6', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsiliNPP2.6.csv')

p1 = p[['silicate8.5', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP8.5', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2TsiliNPP8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tnitratephosal2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tnitratephosal8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tphosal2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tphosal8.5.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsilinitratesal2.6.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsilinitratesal8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tnitratesal2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tnitratesal8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsal2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tnitratesal8.5.csv')

p1 = p[['silicate2.6', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsilisal2.6.csv')

p1 = p[['silicate8.5', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsilisal8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity2.6', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsal2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity8.5', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsal8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tnitratepho2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tnitratepho8.5.csv')

p1 = p[['silicate', 'nitrate', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tpho2.6.csv')

p1 = p[['silicate', 'nitrate', 'phosphate8.5', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tpho8.5.csv')

p1 = p[['silicate2.6', 'nitrate2.6', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsilinitrate2.6.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsilinitrate8.5.csv')

p1 = p[['silicate', 'nitrate2.6', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tnitrate2.6.csv')

p1 = p[['silicate', 'nitrate8.5', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tnitrate8.5.csv')

p1 = p[['silicate2.6', 'nitrate', 'phosphate', 'CHL', 'CO22.6', 'NPP', 'PAR', 'salinity', 'T2.6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsili2.6.csv')

p1 = p[['silicate8.5', 'nitrate', 'phosphate', 'CHL', 'CO28.5', 'NPP', 'PAR', 'salinity', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_ano_CO2Tsili8.5.csv')
'''


#排列组合寻找最小可能
'''
from itertools import combinations
from sklearn import metrics
import pandas as pd
import numpy as np
import itertools
import random
import math 


a1 = [393.5221258,388.1702249,382.818324,377.4664231,372.1145222,366.7626213] #NPP
a2 = [6.605893444,6.496930253,6.387967062,6.279003871,6.17004068,6.061077489] #nitrate
a3 = [0.597951414,0.588088298,0.578225182,0.568362065,0.558498949,0.548635833] #phosphate
a4 = [34.67824583,34.75824583,34.83824583,34.91824583,34.99824583,35.07824583] #salinity
a5 = [9.545117713,9.387672472,9.230227232,9.072781991,8.91533675,8.75789151] #silicate
a6 = [387.2690968,497.3750164,622.6679595,854.2700664] #CO2

#print(a2)
c = itertools.product(a1,a2,a3,a4,a5,a6)
save = []
for i in c:
    save.append(i)
save1 = pd.DataFrame(save)
save1.columns=('NPP','nitrate','phosphate','salinity','silicate','CO2')
save1['T'] = 0
save1['T'].loc[save1['CO2'] == 387.2690968] = 16.56627215
save1['T'].loc[save1['CO2'] == 497.3750164] = 17.37627215
save1['T'].loc[save1['CO2'] == 622.6679595] = 17.83627215
save1['T'].loc[save1['CO2'] == 854.2700664] = 19.32627215
save1['CHL'] = 0.331205863
save1['PAR'] = 36.12750252
save1
p1 = save1[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO2', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
save1['diversity'] = pred_result
save1.to_csv('E:/project/div_all/div_all_times_rate/div_time_rate_多因素排列组合寻找最小变化办法.csv')
'''



'''
pp1 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/div_最优组合.csv", encoding='unicode_escape')
pp2 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/div_最优组合_2.csv", encoding='unicode_escape')
p = pd.concat([pp1,pp2],axis=1)
print(p.columns)
p1 = p[['silicate0.6', 'nitrate0.8', 'phosphate8.5', 'CHL', 'CO22.6', 'NPP0.8', 'PAR', 'salinity0.2', 'T2.6']] #rcp2.6 min
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_multi/div_best_2.6min.csv')

p1 = p[['silicate8.5', 'nitrate0.4', 'phosphate2.6', 'CHL', 'CO22.6', 'NPP2.6', 'PAR', 'salinity8.5', 'T2.6']] #RCP2.6 max
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_multi/div_best_2.6max.csv')

p1 = p[['silicate8.5', 'nitrate2.6', 'phosphate8.5', 'CHL', 'CO24.5', 'NPP0.2', 'PAR', 'salinity0.8', 'T4.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_multi/div_best_4.5min.csv')

p1 = p[['silicate2.6', 'nitrate0.6', 'phosphate2.6', 'CHL', 'CO24.5', 'NPP0.4', 'PAR', 'salinity2.6', 'T4.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_multi/div_best_4.5max.csv')

p1 = p[['silicate8.5', 'nitrate8.5', 'phosphate8.5', 'CHL', 'CO26', 'NPP0.8', 'PAR', 'salinity8.5', 'T6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_multi/div_best_6min.csv')

p1 = p[['silicate2.6', 'nitrate0.4', 'phosphate0.2', 'CHL', 'CO26', 'NPP2.6', 'PAR', 'salinity2.6', 'T6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_multi/div_best_6max.csv')

p1 = p[['silicate8.5', 'nitrate0.2', 'phosphate0.6', 'CHL', 'CO28.5', 'NPP0.2', 'PAR', 'salinity0.8', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_multi/div_best_8.5min.csv')

p1 = p[['silicate2.6', 'nitrate0.8', 'phosphate2.6', 'CHL', 'CO28.5', 'NPP0.8', 'PAR', 'salinity2.6', 'T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_multi/div_best_8.5max.csv')
'''

'''
print("=====feature_importance diversity====")
imp = permutation_importances(X_train, y_train, X_test, rmse)
imp.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_new_importance.csv')

p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2T_change_CO2.csv", encoding='unicode_escape')
p1 = p[['silicate', 'nitrate', 'phosphate', 'CHL', 'CO26', 'NPP', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/div_all_CO2_6.csv')
'''
#ICE_caculate(X_train, X_test, 'div_all_new')
