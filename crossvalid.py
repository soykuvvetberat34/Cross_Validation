#veriyi istenilen kadar parçaya ayırıp içinden bir parçayı çıkarıp
#geri kalanalrı ile o parçayı bulmaya çalışır bunu ayırılan
#parça kadar yapar ve her birinin hata ortalamasını verir
from sklearn.model_selection import cross_val_score as cv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\maas.csv")
x=datas.iloc[:,2].values.reshape(-1,1)
y=datas.iloc[:,-1].values
lin_reg=LinearRegression()
lin_reg.fit(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.22,random_state=99)

#cross validation
error_list=cv(lin_reg,x_train,y_train,cv=10,scoring="neg_mean_squared_error")
#10 parçaya ayırıp her biri için hata tahminleri yapıp ortalama alacak
#şimdi biz hepsinin ortalamasını alacaz
mean_cv=np.mean(-error_list)
print(np.sqrt(mean_cv))




















