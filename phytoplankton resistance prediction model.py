from heamy.dataset import Dataset
from heamy.estimator import Regressor
from heamy.pipeline import ModelsPipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from scipy.special import comb
from itertools import combinations
from sklearn import metrics
import pandas as pd
import numpy as np
import itertools
import random
import math




# create dataset
# create dataset
def DataCleanRA(data):
	#data = data.drop('class', axis=1)
	data = data.drop('RE', axis=1)
	data = data.drop('mon', axis =1)
	data = data.drop('year', axis =1)
	data = data.drop('latitude',axis = 1)
	data = data.drop('longitude',axis = 1)
	data = data.drop('ext-salinity',axis = 1)
	data = data.drop('ext-T',axis = 1)
	data = data.drop('ext-CHL',axis = 1)
	data = data.drop('ext-PAR',axis = 1)
	data = data.drop('ext-CO2',axis = 1)
	#RA anslysis
	data = data[['nitrate','phosphate', 'silicate', 'CHL', 'CO2', 'PAR','salinity','T','RA','class']]
	data = data.dropna()  #delete all rows with NaN
	data['RA'].iloc[data['RA'] > 20] = 20
	data['RA'] = data['RA'].apply(np.log)  #log(RA)
	data = data[['nitrate','phosphate', 'silicate', 'CHL', 'CO2', 'PAR','salinity','T','RA','class']]
	print(data.columns)
	#data['latitude'] = data['latitude'].map(lambda x: math.sin(math.radians(x)))
	#data['longitude'] = data['longitude'].map(lambda x: math.cos(math.radians(x)))
	#data.replace({
	#	'm':0, 's.ext4':-4, 's.ext10':-10, 's.ext20':-20, 's.ext50':-50, 's.ext100':-100,
	#	'b.ext4':4, 'b.ext10':10, 'b.ext20':20, 'b.ext50':50, 'b.ext100':100, 
	select_d = stra_samp(data)
	select_d = select_d[['nitrate','phosphate', 'silicate', 'CHL', 'CO2', 'PAR','salinity','T','pred']]
	print(select_d)
	#	}, inplace=True)  #replace categorical feature
	X_train, X_test, y_train, y_test = Train_Test(select_d)
	X_train1 = pd.DataFrame(X_train)
	return X_train, X_test, y_train, y_test

def DataCleanRE(data):
	#data = data.drop('class', axis=1)
	data = data.drop('RA', axis=1)
	data = data.drop('mon', axis =1)
	data = data.drop('year', axis =1)
	#ata = data.drop('latitude',axis = 1)
	#data = data.drop('longitude',axis = 1)
	data = data.drop('ext-salinity',axis = 1)
	data = data.drop('ext-T',axis = 1)
	data = data.drop('ext-CHL',axis = 1)
	data = data.drop('ext-PAR',axis = 1)
	data = data.drop('ext-CO2',axis = 1)
	#RA anslysis
	data = data.dropna()  #delete all rows with NaN
	data['RE'].iloc[data['RE'] > 20] = 20
	data['RE'].iloc[data['RE'] < 0.05] = 0.05
	data['RE'] = data['RE'].apply(np.log)  #log(RE)
	data = data[['nitrate','phosphate', 'silicate', 'CHL', 'CO2', 'PAR','salinity','T','RE','class']]
	print(data.columns)
	"""
	data['latitude'] = data['latitude'].map(lambda x: math.sin(math.radians(x)))
	data['longitude'] = data['longitude'].map(lambda x: math.cos(math.radians(x)))
	data.replace({
		'm':0, 's.ext4':-4, 's.ext10':-10, 's.ext20':-20, 's.ext50':-50, 's.ext100':-100,
		'b.ext4':4, 'b.ext10':10, 'b.ext20':20, 'b.ext50':50, 'b.ext100':100, 
		}, inplace=True)  #replace categorical feature
	"""
	select_d = stra_samp(data)
	#	}, inplace=True)  #replace categorical feature
	X_train, X_test, y_train, y_test = Train_Test(select_d)
	return X_train, X_test, y_train, y_test

def stra_samp(data):
	data.columns= ['nitrate','phosphate', 'silicate', 'CHL', 'CO2', 'PAR','salinity','T','pred','cla']
	d_other = data.drop(data[(data.cla == 'northp') | (data.cla == 'southp')].index)
	d_pp = data.drop(data[(data.cla != 'southp') & (data.cla != 'northp')].index)
	len_o = len(d_other)
	print(len_o)
	len_pp = len(d_pp)
	print("stra_samp random seed 3")
	random.seed(3)
	select_index = random.sample(list(d_other.index.values),len_pp*1)
	select_d = data.loc[select_index]
	select_save = pd.concat([select_d, d_pp], axis=0)
	select_save = select_save.sort_index()
	select_save = select_save[['nitrate','phosphate', 'silicate', 'CHL', 'CO2', 'PAR','salinity','T','pred']]
	return select_save

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
	X_train1 = pd.DataFrame(X_train)
	X_test1 = pd.DataFrame(X_test)
	y_train1 = pd.DataFrame(y_train)
	#X_test1.to_csv('E:/project/data_stack/test/southp_re_X_test.csv')
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
	stack_ds = pipeline.stack(k=10,seed=1)
	#stacking
	stacker = Regressor(dataset=stack_ds, estimator=BaggingRegressor, parameters={'n_estimators':500, 'max_samples':500, 'max_features': 1.0,'random_state' : 1})
	results = stacker.predict()
	return results

def Validate(results):
	#results_rmse = stacker.validate(k=10,scorer=mean_absolute_error)
	#results_r2 = stacker.validate(k=10,scorer=r2_score)
	r2 = (pd.Series(y_test)).corr(pd.Series(results))
	rmse = np.sqrt(metrics.mean_squared_error(y_test, results))
	return r2, rmse

def ie(data):
	data = data.values
	d_2f = np.round(data, 3)
	d_2f = pd.DataFrame(d_2f)
	d_2f.columns = ['x']
	print(pd.value_counts(d_2f['x']))
	a = pd.value_counts(d_2f['x'])/len(d_2f['x'])
	print(a)
	return sum(np.log2(a) * a * (-1))

def ie_gr(data):
	data.columns = ['str_x','str_y']
	data = data[[str_x,str_y]]
	data = data.values
	d_2f = np.round(data, 3)
	d_2f = pd.DataFrame(d_2f)
	d_2f.columns = [str_x, str_y]
	e1 = d_2f.groupby(str_x).apply(lambda x:infor(x[str_y]))
	p1 = pd.value_counts(d_2f[str_x]) / len(d_2f[str_x])
	e2 = sum(e1 * p1)
	return (ie(d_2f[[str_y]])-e2)/ie(d_2f[[str_x]])

'''original permutation importance calculation
def permutation_importances(X_train, y_train, y_test, metric):
        baseline = metric.copy()
        X_train = pd.DataFrame(X_train)
        save = X_train.copy()
        dd = {}
        for col in X_train.columns:
                imp_rmse = []
                for n in range(3):
                        print(n)
                        np.random.seed(n)
                        X_train[col] = np.random.permutation(X_train[col])
                        r2, rmse = Validate(StackModel(X_train, y_train, X_test))
                        X_train = save.copy()
                        imp_rmse.append((rmse - baseline)/baseline)
                        print(imp_rmse)
                dd[col] = imp_rmse
        dd = pd.DataFrame(dd)
        return dd
'''

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

def combinations_all():
	d = ['CHL','CO2','PAR','salinity','T']
	a = []
	for i in range(1,6,1):
		a1 = list(combinations(d,i))
		a.append(a1)
	return a

def ie_importances(X_train, y_train, X_test, y_test, r20,rmse0,name):
	col_names = ('latitude','longitude','CHL','ext-CHL','CO2','ext-CO2','PAR','ext-PAR','salinity','ext-salinity','T','ext-T')
	X_test1 = pd.DataFrame(X_test)
	X_train1 = pd.DataFrame(X_train)
	X_data = pd.concat([X_train1, X_test1], axis=0)
	X_data.columns = col_names
	y_test1 = pd.DataFrame(y_test)
	y_train1 = pd.DataFrame(y_train)
	y_data = pd.concat([y_train1, y_test1], axis=0)
	y_data.columns = ['y']
	save_data = X_data.copy()
	for i in range(5):
		a = combinations_all()
		b = a[i]
		for ii in range(int(comb(5,i+1))):
			col = b[ii]
			col = list(col)
			for n in range(6):
				np.random.seed(n)
				for m in col:
					X_data[m] = np.random.permutation(X_data[m])
				X_data = X_data.values
				result = StackModel(X_train, y_train, X_data)
				result_1 = pd.DataFrame(result)
				X_data = pd.DataFrame(X_data)
				X_data.index = range(len(X_data))
				X_data.columns = col_names
				result_1.index = range(len(result_1))
				y_data.index = range(len(y_data))
				y_ie = pd.concat([X_data,y_data, result_1], axis=1)
				y_ie.to_csv('D:/%s_%s_ie1117_ieimp_%s.csv'%(name, col, n))
				X_data = save_data.copy() 
	return save

def ICE_caculate_save(X_train, X_test, save_name):
	col_names = ('nitrate','phosphate', 'silicate', 'CHL', 'CO2', 'PAR','salinity','T')
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
		if len(col_du) < 5:
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
			for i in range(5):
				col_du_perc.append(np.percentile(col_du, i*20))
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
		save_i.to_csv('E:/project/data_stack/total_model/%s_ice_data_%s.csv'%(save_name, col_i))
		X_data = save.copy()

def ICE_caculate(X_train, X_test, save_name):
	col_names = ('latitude','longitude','CHL','ext-CHL','CO2','ext-CO2','PAR','ext-PAR','salinity','ext-salinity','T','ext-T')
	X_test = pd.DataFrame(X_test)
	X_train = pd.DataFrame(X_train)
	X_data = pd.concat([X_train, X_test], axis=0)
	X_data.columns = col_names
	#X_data.to_csv('E:/project/data_stack/test/1111test/%s_ice_1117_data_%s.csv'%(save_name, 'X_data'))
	col_names = list(col_names)
	ana_v = ['CHL','CO2','PAR','salinity','T']
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
			col_1_perc.append(np.percentile(col_1, i*5))
			col_2_perc.append(np.percentile(col_2, i*5))
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
				X_data = X_data.values
				pred = StackModel(X_train, y_train, X_data)
				pred = pd.DataFrame(pred)
				pred.columns = [n]
				save_i = pd.concat([save_i, pred], axis=1)
				n_save.append(n)
				X_data = save.copy()
			m_save[m] = n_save
			save_i.to_csv('D:/%s_%s_ice_1117_%s.csv'%(save_name,col_i,m))
			print(save_i)
		m_save.to_csv('D:/%s_ice_mn1117_%s.csv'%(save_name, col_i))

def Pred(X_train, X_test, y_train, y_test,save_name):
	col_names = ('latitude','longitude','CHL','ext-CHL','CO2','ext-CO2','PAR','ext-PAR','salinity','ext-salinity','T','ext-T')
	X_test = pd.DataFrame(X_test)
	X_train = pd.DataFrame(X_train)
	X_data = pd.concat([X_train, X_test], axis=0)
	X_data.columns = col_names
	X_data.index = range(len(X_data))
	y_test = pd.DataFrame(y_test)
	y_train = pd.DataFrame(y_train)
	y_data = pd.concat([y_train, y_test], axis=0)
	y_data.index = range(len(y_data))
	y_data.columns = ['orig_y']
	save_i = []
	X_data1 = X_data.values
	pred = StackModel(X_train, y_train, X_data1)
	pred = pd.DataFrame(pred)
	pred.index = range(len(pred))
	print(len(pred))
	print(len(X_data))
	save_i = pd.concat([X_data, y_data, pred], axis=1)
	save_i.to_csv('D:/%s_ice_all_data_predtion.csv'%(save_name))



print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("=====================total del ext addfeatures RA=======================")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")

data = pd.read_csv("E:/project/data_stack/total-addf.csv", encoding='unicode_escape')
X_train, X_test, y_train, y_test = DataCleanRA(data)
results = StackModel(X_train, y_train, X_test)
#data_pt = pd.DataFrame({'y_test':y_test, 'results':results})  #save the prediction with y test
#data_pt.to_csv('D:/northp_ra_pt.csv')
r2, rmse = Validate(results)
print("====northp RA R2====")
print(r2)
print("====northp RA RMSE====")
print(rmse)
#ICE_caculate_save(X_train, X_test, 'RA_total_')

'''
#imp = permutation_importances(X_train, y_train, X_test, rmse)
#imp.to_csv('E:/project/data_stack/total_model/total_addf_imoportance_20210306.csv')
allx = np.concatenate((X_train,X_test),axis=0)
ally = np.concatenate((y_train,y_test),axis=0)
predy_all = StackModel(X_train, y_train, allx)
p_y = pd.DataFrame(predy_all)
o_y = pd.DataFrame(ally)
pall = pd.concat([p_y,o_y],axis=1)
pall.to_csv('E:/project/data_stack/total_model/totalall_CHL_T_nitrate_silicate_CO2_PAR_salinity_phosphate_0404.csv')

p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2T_1222.csv", encoding='unicode_escape')
#p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO2', 'PAR', 'salinity','T']]
p1 = p[['nitrate','phosphate','silicate','CHL','CO2','PAR','salinity','T']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_2018_CHL_T_nitrate_silicate_T_PAR_salinity_phoaphate.csv')
'''




'''
p = pd.read_csv("E:/project/div_all/div_all_times_rate/new_div_all/ra_new_data_creating_for_2D_plot.csv", encoding='unicode_escape')
p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO2', 'PAR', 'salinity','T']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
p['pred'] = pred_result
p.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/多因素/RA_代表数据生成_for_2D.csv')

'''

'''
p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')
p1 = p[['nitrate8.5', 'phosphate8.5', 'silicate2.6', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/多因素/ra_onlysili2.6.csv')

p1 = p[['nitrate2.6', 'phosphate8.5', 'silicate2.6', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/多因素/ra_onlysili_ni2.6.csv')

p1 = p[['nitrate2.6', 'phosphate2.6', 'silicate2.6', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/多因素/ra_onlysili_ni_pho2.6.csv')

p1 = p[['nitrate2.6', 'phosphate8.5', 'silicate8.5', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/多因素/ra_onlyni2.6.csv')

p1 = p[['nitrate2.6', 'phosphate2.6', 'silicate8.5', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/多因素/ra_onlyni_pho2.6.csv')

p1 = p[['nitrate8.5', 'phosphate2.6', 'silicate8.5', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/多因素/ra_onlypho2.6.csv')

p1 = p[['nitrate8.5', 'phosphate2.6', 'silicate2.6', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/div_all/div_all_times_rate/new_div_all/多因素/ra_onlypho_sili2.6.csv')
'''


'''
p000 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2T_change_CO2.csv",encoding="unicode_escape")
p00 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')
p01 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2Tother_两端10.csv", encoding='unicode_escape')
p02 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2Tother_两端10_1.csv", encoding='unicode_escape')
p03 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2Tother_两端10_2.csv", encoding='unicode_escape')
p04 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2Tother_两端10_3.csv", encoding='unicode_escape')
p100 = p000[['silicate','nitrate','phosphate','NPP','salinity']]
p10 = p00[['silicate2.6', 'nitrate2.6', 'phosphate2.6']]
p11 = p01[['CHL','PAR','nitrate02','nitrate04','nitrate06','nitrate08','nitrate10','nitrate15','nitrate20','nitrate30','nitrate40']]
p12 = p02[['NPP02','NPP04','NPP06','NPP08','NPP10','NPP15','NPP20','NPP30','NPP40','phosphate02','phosphate04','phosphate06','phosphate08','phosphate10','phosphate15','phosphate20','phosphate30','phosphate40']]
p13 = p03[['silicate02','silicate04','silicate06','silicate08','silicate10','silicate15','silicate20','silicate30','silicate40','salinity02','salinity04','salinity06','salinity08','salinity10','salinity15','salinity20','salinity30','salinity40']]
p14 = p04[['CO202','CO204','CO206','CO208','CO210','CO215','CO220','CO230','CO240','T02','T04','T06','T08','T10','T15','T20','T30','T40']]
p = pd.concat([p100,p10,p11,p12,p13,p14],axis=1)

print("CO2Tother_10")

p1 = p[['nitrate', 'phosphate','silicate', 'CHL','CO202','PAR', 'salinity', 'T02']]
print(p1)

p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2T02other0518.csv')



p1 = p[['nitrate', 'phosphate','silicate',  'CHL', 'CO204', 'PAR', 'salinity', 'T04']]
p1 = p1.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2T04other.csv')

p1 = p[['nitrate', 'phosphate','silicate','CHL', 'CO206', 'PAR', 'salinity', 'T06']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2T06other.csv')

p1 = p[['nitrate', 'phosphate','silicate', 'CHL', 'CO208', 'PAR', 'salinity', 'T08']]
print(p1)
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2T08other.csv')

p1 = p[['nitrate', 'phosphate','silicate', 'CHL', 'CO210', 'PAR', 'salinity', 'T10']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2T10other.csv')

p1 = p[['nitrate', 'phosphate','silicate', 'CHL', 'CO215', 'PAR', 'salinity', 'T15']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2T15other.csv')

p1 = p[['nitrate', 'phosphate','silicate', 'CHL', 'CO220', 'PAR', 'salinity', 'T20']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2T20other.csv')

p1 = p[['nitrate', 'phosphate','silicate', 'CHL', 'CO230', 'PAR', 'salinity', 'T30']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2T30other.csv')

p1 = p[['nitrate', 'phosphate','silicate', 'CHL', 'CO240', 'PAR', 'salinity', 'T40']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2T40other.csv')
'''

'''
p00 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')

p10 = p00[['nitrate', 'phosphate','silicate', 'CHL','CO2','PAR', 'salinity', 'T8.5']]
p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_T8.5.csv')

p10 = p00[['nitrate', 'phosphate','silicate', 'CHL','CO2','PAR', 'salinity', 'T2.6']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_T2.6.csv')

p10 = p00[['nitrate', 'phosphate','silicate', 'CHL','CO28.5','PAR', 'salinity', 'T']]
p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_CO28.5.csv')

p10 = p00[['nitrate', 'phosphate','silicate', 'CHL','CO22.6','PAR', 'salinity', 'T']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_CO22.6.csv')

p10 = p00[['nitrate', 'phosphate','silicate', 'CHL','CO28.5','PAR', 'salinity8.5', 'T8.5']]
p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_CO2Tsalinity8.5.csv')

p10 = p00[['nitrate', 'phosphate','silicate', 'CHL','CO22.6','PAR', 'salinity2.6', 'T2.6']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_CO2Tsalinity2.6.csv')

p10 = p00[['nitrate', 'phosphate','silicate8.5', 'CHL','CO28.5','PAR', 'salinity', 'T8.5']]
p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_CO2Tsilicate8.5.csv')

p10 = p00[['nitrate', 'phosphate','silicate2.6', 'CHL','CO22.6','PAR', 'salinity', 'T2.6']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_CO2Tsilicate2.6.csv')

p10 = p00[['nitrate', 'phosphate8.5','silicate', 'CHL','CO28.5','PAR', 'salinity', 'T8.5']]
p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_CO2Tphosphate8.5.csv')

p10 = p00[['nitrate', 'phosphate2.6','silicate', 'CHL','CO22.6','PAR', 'salinity', 'T2.6']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_CO2Tphosphate2.6.csv')

p10 = p00[['nitrate8.5', 'phosphate','silicate', 'CHL','CO28.5','PAR', 'salinity', 'T8.5']]
p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_CO2Tnitrate8.5.csv')

p10 = p00[['nitrate2.6', 'phosphate','silicate', 'CHL','CO22.6','PAR', 'salinity', 'T2.6']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_CO2Tnitrate2.6.csv')

p10 = p00[['nitrate', 'phosphate','silicate', 'CHL','CO2','PAR', 'salinity8.5', 'T']]
p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_salinity8.5.csv')

p10 = p00[['nitrate', 'phosphate','silicate', 'CHL','CO2','PAR', 'salinity2.6', 'T']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_salinity2.6.csv')

p10 = p00[['nitrate', 'phosphate','silicate8.5', 'CHL','CO2','PAR', 'salinity', 'T']]
p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_silicate8.5.csv')

p10 = p00[['nitrate', 'phosphate','silicate2.6', 'CHL','CO2','PAR', 'salinity', 'T']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_silicate2.6.csv')

p10 = p00[['nitrate', 'phosphate8.5','silicate', 'CHL','CO2','PAR', 'salinity', 'T']]
p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_phosphate8.5.csv')

p10 = p00[['nitrate', 'phosphate2.6','silicate', 'CHL','CO2','PAR', 'salinity', 'T']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_phosphate2.6.csv')

p10 = p00[['nitrate8.5', 'phosphate','silicate', 'CHL','CO2','PAR', 'salinity', 'T']]
p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_nitrate8.5.csv')

p10 = p00[['nitrate2.6', 'phosphate','silicate', 'CHL','CO2','PAR', 'salinity', 'T']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_0821_nitrate2.6.csv')
'''



'''
p00 = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')
p10 = p00[['nitrate', 'phosphate','silicate', 'CHL','CO28.5','PAR', 'salinity', 'T8.5']]

p1 = p10.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2Tother8.5_2018.csv')

p10 = p00[['nitrate', 'phosphate','silicate',  'CHL', 'CO22.6', 'PAR', 'salinity', 'T2.6']]
p1 = p10.dropna()  #delete all rows with NaN
print(p1)
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/RA_CO2Tother2.6_2018.csv')






p = pd.read_csv("D:/total2/total_ra_data_creating.csv", encoding='unicode_escape')
p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO2', 'PAR', 'salinity','T']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
p1['pred'] = pred_result
p1.to_csv('D:/total2/total_ra_data_creating_pred.csv')
'''



'''
p = pd.read_csv("D:/ext-data-predict/CO2T_1222.csv", encoding='unicode_escape')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO2', 'PAR', 'salinity','T']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_2018.csv')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2T2.6.csv')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO24.5', 'PAR', 'salinity','T4.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2T4.5.csv')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO26', 'PAR', 'salinity','T6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2T6.csv')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO28.5', 'PAR', 'salinity','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2T8.5.csv')
'''

'''
p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')

p1 = p[['nitrate8.5', 'phosphate', 'silicate8.5', 'CHL', 'CO28.5', 'PAR', 'salinity','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_CO2Tnitratesilicate8.5.csv')

p1 = p[['nitrate2.6', 'phosphate', 'silicate2.6', 'CHL', 'CO22.6', 'PAR', 'salinity','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_CO2Tnitratesilicate2.6.csv')
'''

'''


p1 = p[['nitrate2.6', 'phosphate', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tnitrate2.6.csv')

p1 = p[['nitrate8.5', 'phosphate', 'silicate', 'CHL', 'CO28.5', 'PAR', 'salinity','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tnitrate8.5.csv')



p1 = p[['nitrate', 'phosphate', 'silicate2.6', 'CHL', 'CO22.6', 'PAR', 'salinity','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsilicate2.6.csv')
'''




'''
p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/TCO2_others_1222.csv", encoding='unicode_escape')
p1 = p[['nitrate2.6', 'phosphate2.6', 'silicate2.6', 'CHL', 'CO2', 'PAR', 'salinity2.6','T']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_shap_other2.6.csv')

p1 = p[['nitrate8.5', 'phosphate8.5', 'silicate8.5', 'CHL', 'CO2', 'PAR', 'salinity8.5','T']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_shap_other8.5.csv')

p1 = p[['nitrate2.6', 'phosphate2.6', 'silicate2.6', 'CHL', 'CO2', 'PAR', 'salinity2.6','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_shap_Tother2.6.csv')

p1 = p[['nitrate8.5', 'phosphate8.5', 'silicate8.5', 'CHL', 'CO2', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_shap_Tother8.5.csv')

p1 = p[['nitrate2.6', 'phosphate2.6', 'silicate2.6', 'CHL', 'CO22.6', 'PAR', 'salinity2.6','T']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_shap_CO2other2.6.csv')

p1 = p[['nitrate8.5', 'phosphate8.5', 'silicate8.5', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_shap_CO2other8.5.csv')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO2', 'PAR', 'salinity','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_shap_T2.6.csv')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO2', 'PAR', 'salinity','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_shap_T8.5.csv')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity','T']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_shap_CO22.6.csv')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO28.5', 'PAR', 'salinity','T']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('E:/project/data_stack/total_model/total_shap_CO28.5.csv')

p = pd.read_csv("E:/project/ext-总结/巴黎协定_温度变化/CO2T_1222.csv", encoding='unicode_escape')

p1 = p[['nitrate', 'phosphate', 'silicate','CHL', 'CO2', 'PAR', 'salinity', 'T4.5']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/total_shap_T4.5.csv')

p1 = p[[ 'nitrate', 'phosphate', 'silicate','CHL', 'CO2',  'PAR', 'salinity', 'T6']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/total_shap_T6.csv')

p1 = p[['nitrate', 'phosphate', 'silicate','CHL', 'CO24.5', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/total_shap_CO24.5.csv')

p1 = p[['nitrate', 'phosphate', 'silicate','CHL', 'CO26', 'PAR', 'salinity', 'T']]
p1 = p1.dropna()  #delete all rows with NaN
p_v = p1.values
pred_result = StackModel(X_train, y_train, p_v)
pred_result = pd.DataFrame(pred_result)
pred_result.to_csv('E:/project/data_stack/total_model/total_shap_CO26.csv')
'''




'''
p1 = p[['nitrate2.6', 'phosphate', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tnitrate2.6.csv')

p1 = p[['nitrate8.5', 'phosphate', 'silicate', 'CHL', 'CO28.5', 'PAR', 'salinity','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tnitrate8.5.csv')

p1 = p[['nitrate', 'phosphate', 'silicate8.5', 'CHL', 'CO28.5', 'PAR', 'salinity','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsilicate8.5.csv')

p1 = p[['nitrate', 'phosphate2.6', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tphos2.6.csv')

p1 = p[['nitrate', 'phosphate8.5', 'silicate', 'CHL', 'CO28.5', 'PAR', 'salinity','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tphos8.5.csv')

p1 = p[['nitrate2.6', 'phosphate2.6', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tphosnitrate2.6.csv')

p1 = p[['nitrate8.5', 'phosphate8.5', 'silicate', 'CHL', 'CO28.5', 'PAR', 'salinity','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tphosnitrate8.5.csv')

p1 = p[['nitrate', 'phosphate2.6', 'silicate2.6', 'CHL', 'CO22.6', 'PAR', 'salinity','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tphossilicate2.6.csv')

p1 = p[['nitrate', 'phosphate8.5', 'silicate8.5', 'CHL', 'CO28.5', 'PAR', 'salinity','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tphossilicate8.5.csv')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity2.6','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsal2.6.csv')

p1 = p[['nitrate', 'phosphate', 'silicate', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsal8.5.csv')

p1 = p[['nitrate', 'phosphate', 'silicate2.6', 'CHL', 'CO22.6', 'PAR', 'salinity2.6','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsalsili2.6.csv')

p1 = p[['nitrate', 'phosphate', 'silicate8.5', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsalsili8.5.csv')

p1 = p[['nitrate2.6', 'phosphate', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity2.6','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsalnitra2.6.csv')

p1 = p[['nitrate8.5', 'phosphate', 'silicate', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsalnitra8.5.csv')

p1 = p[['nitrate2.6', 'phosphate', 'silicate2.6', 'CHL', 'CO22.6', 'PAR', 'salinity2.6','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsalnitrasili2.6.csv')

p1 = p[['nitrate8.5', 'phosphate', 'silicate8.5', 'CHL', 'CO28.5', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsalnitrasili8.5.csv')

p1 = p[['nitrate', 'phosphate2.6', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity2.6','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsalphos2.6.csv')

p1 = p[['nitrate', 'phosphate8.5', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsalphos8.5.csv')


p1 = p[['nitrate2.6', 'phosphate2.6', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity2.6','T2.6']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsalphosnitrate2.6.csv')

p1 = p[['nitrate8.5', 'phosphate8.5', 'silicate', 'CHL', 'CO22.6', 'PAR', 'salinity8.5','T8.5']]
p1 = p1.dropna()
p_v = p1.values
pred_result = StackModel(X_train, y_train,p_v)
pred = pd.DataFrame(pred_result)
pred.to_csv('D:/total2/total_CO2Tsalphosnitrate8.5.csv')


ICE_caculate_save(X_train, X_test, 'total_ra')
'''
