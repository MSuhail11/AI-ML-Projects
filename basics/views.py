#from django.shortcuts import render

# Create your views here.

from django.shortcuts import render

import pandas as pd
import sklearn 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def src(request):
     if(request.method=="POST"):
        data=request.POST

        fLength=float(data.get('fLength'))
        fWidth=float(data.get('fWidth'))
        fSize=float(data.get('fSize'))
        fConc=float(data.get('fConc'))
        fConc1=float(data.get('fConc1'))
        fAsym=float(data.get('fAsym'))
        fM3Long=float(data.get('fM3Long'))
        fM3Trans=float(data.get('fM3Trans'))
        fAlpha=float(data.get('fAlpha'))
        fDist=float(data.get('fDist'))
        path="C:\\Users\\DELL\\Desktop\\Karunadu Project\\ML\\train_dataset.csv"
        data1=pd.read_csv(path)
        
        
        le_class=LabelEncoder()
        data1['class_new']=le_class.fit_transform(data1['class'])
        x= data1.drop(['class','class_new'],'columns')
        y = data1['class_new']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        k = 71  
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred= knn.predict([[fLength,fWidth,fSize,fConc,fConc1,fAsym,fM3Long,fM3Trans,fAlpha,fDist]])
        if y_pred==1:
            info="hadron (background)"
        else:
            info="gamma (signal)"
        return render(request,"src.html",context={'info':info})

     return render(request,'src.html')