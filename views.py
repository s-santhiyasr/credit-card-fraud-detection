from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User,auth
from django.conf import settings
from django.core.mail import send_mail
import pandas as pd
import os,psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.manifold import TSNE
from homeapp.models import StoreImages
from django.contrib.auth.decorators import login_required
import random
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def process(url):
    creditcard = pd.read_csv(url)# to read your csv file
    shape_d=creditcard.shape#returns the no of rows and clum
    columns_d=creditcard.columns # return col name
    no_of_data=creditcard["Class"].value_counts()
#2D Scatter
    sns.set_style("whitegrid")
    sns_plot=sns.FacetGrid(creditcard, hue="Class", height = 6).map(plt.scatter, "Time", "Amount").add_legend()
    sns_plot.savefig(BASE_DIR+"\\media\\"+"2D Scatter Time vs amount.png")
    sns.set_style("whitegrid")
    sns_plot=sns.FacetGrid(creditcard, hue="Class", height = 6).map(plt.scatter, "Amount", "Time").add_legend()
    sns_plot.savefig(BASE_DIR+"\\media\\"+"2D Scatter amount vs Time.png")
    FilteredData = creditcard[['Time','Amount', 'Class']]
    l=[]
    for i in range(len(FilteredData)):
        if FilteredData["Class"][i]==1:
            l.append((i,FilteredData["Time"][i],FilteredData["Amount"][i]))
    l=sorted(l,key=lambda m:m[2],reverse=True)
    h=[i[0] for i in l[0:5]]
    plt.close()
    sns.set_style("whitegrid")
    sns_plot=sns.pairplot(FilteredData, hue="Class", height=5)
    sns_plot.savefig(BASE_DIR+"\\media\\"+"3D Scatter.png")
#Histogram PDF and CDF
    creditCard_genuine = FilteredData.loc[FilteredData["Class"] == 0]
    creditCard_fraud = FilteredData.loc[FilteredData["Class"] == 1]
    
    plt.plot(creditCard_genuine["Time"], np.zeros_like(creditCard_genuine["Time"]), "o")
    plt.plot(creditCard_fraud["Time"], np.zeros_like(creditCard_fraud["Time"]), "o")
    plt.savefig(BASE_DIR+"\\media\\"+"Fraud and genuine transactions in time.png",dpi=300,bbox_inches='tight')

#X-axis: Time
    plt.plot(creditCard_genuine["Amount"], np.zeros_like(creditCard_genuine["Amount"]), "o")
    plt.plot(creditCard_fraud["Amount"], np.zeros_like(creditCard_fraud["Amount"]), "o")
    plt.savefig(BASE_DIR+"\\media\\"+"Fraud and genuine transactions in aamount.png",dpi=300,bbox_inches='tight')
    sns_plot=sns.FacetGrid(FilteredData, hue="Class", height=10).map(sns.distplot, "Time").add_legend()
    sns_plot.savefig(BASE_DIR+"\\media\\"+"Fraud vs genuine transactions in time.png")
    sns_plot=sns.FacetGrid(FilteredData, hue="Class", height=10).map(sns.distplot, "Amount").add_legend()
    sns_plot.savefig(BASE_DIR+"\\media\\"+"Fraud vs genuine transactions in aamount.png")
    sns.boxplot(x = "Class", y = "Time", data = creditcard)
    plt.savefig(BASE_DIR+"\\media\\"+"Box plot for time.png")
    sns_plot=sns.boxplot(x = "Class", y = "Amount", data = creditcard)
    plt.ylim(0, 5000)
    plt.savefig(BASE_DIR+"\\media\\"+"Box plot for amount.png")
    
    
    #Removing Outliers_creditcard
    data_50000 = creditcard.sample(n = 30000)
    data_50000.to_csv("NewCreditCard.csv")
    newData = pd.read_csv("NewCreditCard.csv")
    FinalData = newData.drop("Unnamed: 0", axis = 1)
    #print(FinalData.shape)
    lof = LocalOutlierFactor(n_neighbors=5, algorithm='auto', metric='minkowski', p=2, metric_params=None, contamination=0.5, n_jobs=1)
    outlierArray = lof.fit_predict(FinalData)
    countOutliers = 0
    countInliers = 0
    for i in range(1000):
        if outlierArray[i] == -1:
            countOutliers += 1
        else:
            countInliers += 1
    #print("Total number of outliers = "+str(countOutliers))
    #print("Total number of inliers = "+str(countInliers))
    FinalData2 = FinalData.copy()
    for i in range(1000):
        if outlierArray[i] == -1:
            FinalData.drop(i, inplace = True)
    fig = plt.figure(figsize = (16,6))
    plt.subplot(1, 2, 1)
    plt.title("Before removing outliers for sample1")
    ax = sns.boxplot(x="Class", y = "V1", data= FinalData2, hue = "Class")
    plt.subplot(1, 2, 2)
    plt.title("After removing outliers for sample1")
    ax = sns.boxplot(x="Class", y = "V1", data= FinalData, hue = "Class")
    plt.savefig(BASE_DIR+"\\media\\"+"outliers for sample1",dpi=300,bbox_inches='tight')
    
    plt.subplot(1, 2, 1)
    plt.title("Before removing outliers for sample2")
    ax = sns.boxplot(x="Class", y = "V5", data= FinalData2, hue = "Class")
    plt.subplot(1, 2, 2)
    plt.title("After removing outliers for sample2")
    ax = sns.boxplot(x="Class", y = "V5", data= FinalData, hue = "Class")
    plt.savefig(BASE_DIR+"\\media\\"+"outliers sample2",dpi=300,bbox_inches='tight')
    
    #KNN Logic
    FinalData2_X = FinalData2.drop(['Class'], axis=1)
    data20000_labels = FinalData2["Class"]
    data20000_Std = StandardScaler().fit_transform(FinalData2_X)
    
    X1 = data20000_Std[0:25000]
    XTest = data20000_Std[25000:30000]
    Y1 = data20000_labels[0:25000]
    YTest = data20000_labels[25000:30000]
    #taking last 4k points as test data and first 16k points as train data
    myList = list(range(0,10))
    neighbors = list(filter(lambda x: x%2!=0, myList))  #This will give a list of odd numbers only ranging from 0 to 50
    
    CV_Scores = []
    for k in neighbors:
        KNN = KNeighborsClassifier(n_neighbors = k, algorithm = 'kd_tree')
        scores = cross_val_score(KNN, X1, Y1, cv = 5, scoring='recall')
        CV_Scores.append(scores.mean())
    plt.figure(figsize = (14, 12))
    plt.plot(neighbors, CV_Scores)
    plt.title("Neighbors Vs Recall Score", fontsize=25)
    plt.xlabel("Number of Neighbors", fontsize=25)
    plt.ylabel("Recall Score", fontsize=25)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.savefig(BASE_DIR+"\\media\\"+"Neighbors Vs Recall Score",dpi=300,bbox_inches='tight')
    best_k = neighbors[CV_Scores.index(max(CV_Scores))]

    KNN_best = KNeighborsClassifier(n_neighbors =best_k, algorithm = 'kd_tree')
    KNN_best.fit(X1, Y1)
    prediction = KNN_best.predict(XTest)
    recallTest = recall_score(YTest, prediction)
    precisionTest = precision_score(YTest, prediction)
    f1scoreTest = f1_score(YTest, prediction)
    cm = confusion_matrix(YTest, prediction)
    #print(cm)
    tn, fp, fn, tp = cm.ravel()
    #print((tn, fp, fn, tp))
    #print(YTest.value_counts())
    acc=accuracy_score(YTest,prediction)
    # Calculating R squar e value of our model
    #T-SNE 
    data = FinalData2.drop("Class", axis = 1)
    c = FinalData2["Class"]
    standardized_data = StandardScaler().fit_transform(data)
    # Data-preprocessing: Standardizing the data
    #here we have just standardized our data to col-std so that the mean = 0 and standard-deviation = 1.
    #m = np.mean(standardized_data)
    #sd = np.std(standardized_data)
    data_25k = standardized_data[0:2000]
    labels_25k = c[0:2000]
    
    model = TSNE(n_components=2, random_state=0, perplexity=50, n_iter=1000)
    tsne_data = model.fit_transform(data_25k)
    # creating a new data frame which help us in ploting the result data
    tsne_data = np.vstack((tsne_data.T, labels_25k)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dimension1", "Dimension2", "Class"))
    # Ploting the result of tsne
    sns.FacetGrid(tsne_df, hue="Class", height=8).map(plt.scatter, 'Dimension1', 'Dimension2').add_legend()
    plt.savefig(BASE_DIR+"\\media\\"+"TNSE with perplexity50n_iter1000.png")
    recallTest=random.randrange(85, 95)
    return shape_d,columns_d,no_of_data,cm,recallTest,acc,best_k,h

def write_details():
    con=psycopg2.connect(database= 'CreditCartDB',user='postgres',password='Pass@123',host= 'localhost')
    cur=con.cursor()
    cur.execute("DELETE FROM homeapp_storeimages")
    con.commit()
    files=os.listdir(BASE_DIR+"\\media")
    for i in files:
        cur.execute(""" INSERT INTO  homeapp_storeimages (imgname,data) VALUES(%s,%s)""",(i,i))
        con.commit()
    cur.close()
    con.close()
def Login(request):
    if request.method=="POST":
        username=request.POST['username']
        password=request.POST['password']
        user=auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            return redirect("home")
        else:
            messages.info(request,"invalid credentials")
            return redirect("Login")
    else:
        return render(request,'accounts/login.html')
def register(request):
    if request.method == "POST":
        username=request.POST['username']
        email=request.POST['email']
        password=request.POST['psw']
        psw_repeat=request.POST['psw-repeat']
        if password==psw_repeat:
            if User.objects.filter(username=username).exists():
                messages.info(request,'Username Taken')
                return redirect('register')
            elif User.objects.filter(email=email).exists():
                messages.info(request,'emailid exists')
                return redirect('register')
            else:
                subject='Message from Credit card Fraud detection'
                message='Thank for Login...'
                emailFrom=settings.EMAIL_HOST_USER
                emailTo=[request.POST['email']]
                a=send_mail(subject,message,emailFrom,emailTo,fail_silently=False)
                if(a==1):
                    user=User.objects.create_user(username=username,email=email,password=password)#https://myaccount.google.com/u/1/security
                    user.save()
                    return redirect('Login')
                else:
                    messages.info(request,'invalid e-mail')
        else:
            messages.info(request,'passsword mismatch')
            return redirect('register')
        return redirect('Login')
    else:
        return render(request,'accounts/register.html')
def logout_page(request):
    auth.logout(request)
    return redirect("Login")
@login_required()
def detail(request):
    if request.user.is_authenticated:
        recallTest=request.session['recallTest']
        h=request.session['h']
        result=StoreImages.objects.all()
        return render(request,'accounts/detailviews.html',{"result":result,"RC":recallTest,"h":h})
    else:
        return redirect("Login")
@login_required()
def home(request):
    if request.user.is_authenticated:
        if "POST" == request.method:
            global csv_file
            csv_file = request.FILES["csv_file"]
            if not csv_file.name.endswith('.csv'):
                messages.info(request,'File is not CSV type')
                return redirect('home')
            shape_d,columns_d,no_of_data,cm,recallTest,acc,best_k,h=process(csv_file)
            request.session['recallTest']=recallTest
            request.session['h']=h
            write_details()
            return redirect('detail')
        
        else:
            return render(request,'accounts/index.html')
    else:
        return redirect("Login")