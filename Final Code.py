#!/usr/bin/env python
# coding: utf-8


import pickle
from scipy.sparse import lil_matrix, csr_matrix
import numpy as np
import pandas as pd
import random
from sklearn.naive_bayes import GaussianNB
import xml.etree.ElementTree as ET
import getopt,sys
import os

INPUT='/home/mila/teaching/user22/Public_Test/'
OUTPUT='/home/mila/teaching/user22/Outputs2/'

try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
except getopt.GetoptError:
    print('ift6758 -i <inputfile> -o <outputfile>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('Please follow the below line command, with a complete path of the test data directory as input\n and the complete directory for the outputs\n ift6758 -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        INPUT = arg
    elif opt in ("-o", "--ofile"):
        OUTPUT = arg
print('Input file is "', INPUT)
print('Output file is "', OUTPUT)

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

data_profile=pd.read_csv('/home/mila/teaching/user22/Train/Profile/Profile.csv')
data_gender=pd.read_csv('/home/mila/teaching/user22/Train/Profile/Profile.csv',usecols=['userid','gender'])
image=pd.read_csv('/home/mila/teaching/user22/Train/Image/oxford.csv')



#Tidying the Data


def age_class(ages):
    """ 
    age_class function transforms age into 4 categories variable
    
    """
    result=[]
    for age in ages:
        if age<=24.:
            result.append('xx-24')
        elif age<=34.:
            result.append('25-34')
        elif age<=49.:
            result.append('35-49')
        else:
            result.append('50-xx')
    return pd.Series(result)

data_profile = (data_profile.assign(
        age=lambda df: age_class(df["age"])
    ).dropna())

data_profile = data_profile.set_index('userid').sort_index()
data_profile = data_profile.drop(columns='Unnamed: 0')

# Age Baseline Prediction


def majority(self) :
    """
    The majority function takes a series and return the more frequent value or label.
    If there are many such labels, the function returns a randomly chosen one.
    """
    y = self.value_counts()
    MajorList = list(y[y==max(y)].index)
    MajorValueIndex = 1
    if len(MajorList) >=2 :
        import random
        MajorValueIndex = random.randint(1,len(MajorList))  # if there are many categories, randomly choose among them
    MajorValue = MajorList[MajorValueIndex-1]
    return MajorValue

AgePredict = majority(data_profile.age)

"""
Gender Baseline Prediction  (considering 0 = 'male', 1 = 'female')

"""
GenderPredictCode = majority(data_profile.gender)
if GenderPredictCode == 0:
    GenderPredictCode = 'male'
elif GenderPredictCode == 1 :
    GenderPredictCode = 'female'
GenderPredictCode


""" Personalities Baseline Prediction """
BL_ope=round(data_profile.ope.mean(), 2)
BL_con=round(data_profile.con.mean(), 2)
BL_ext=round(data_profile.ext.mean(), 2)
BL_agr=round(data_profile.agr.mean(), 2)
BL_neu=round(data_profile.neu.mean(), 2)


""" New class of users """

class User:
    """
    Here we construct a class of users whose main attibutes are their userid, age,
    gender and personality traits
    """
    def __init__(self, userid, age=AgePredict, gender=GenderPredictCode, ope=str(BL_ope), con=str(BL_con), ext=str(BL_ext),
                 agr=str(BL_agr), neu=str(BL_neu)):
        self.userid=userid
        self.age=age
        if gender==0.: self.gender='male'
        else: self.gender='female'
        self.ope=ope
        self.con=con
        self.ext=ext
        self.agr=agr
        self.neu=neu
        
# create the file structure
        
    def output(self):
        data = ET.Element('user')
        item0 = ET.SubElement(data, 'id')
        item1 = ET.SubElement(data, 'age_group')
        item2 = ET.SubElement(data, 'gender')
        item3 = ET.SubElement(data, 'extrovert')
        item4 = ET.SubElement(data, 'neurotic')
        item5 = ET.SubElement(data, 'agreeable')
        item6 = ET.SubElement(data, 'conscientious')
        item7 = ET.SubElement(data, 'open')
        item0.set('id',self.userid)
        item1.set('age_group',self.age)
        item2.set('gender',self.gender)
        item3.set('extrovert',self.ext)
        item4.set('neurotic',self.neu)
        item5.set('agreeable',self.agr)
        item6.set('conscientious',self.con)
        item7.set('open',self.ope)

        """ create a new XML file with the results """
        mydata = ET.tostring(data, encoding='unicode')
        myfile = open(OUTPUT+self.userid+".xml", "w+")
        myfile.write(mydata)
        myfile.close()


########### GENDER PREDICTION #############


# Training Process

print("Preprocessing the training oxford file...\n")

def image_preprocess(image_set):
    """
    This is a feature extraction function that takes as argument the initial features on oxford data, 
    compute relevant distance features and facial hair feature, then compute ratios of distance features
    """
    
    """
    First, we can remove the height because it is equal to the width; and also headPose_pitch 
    because it's zero everywhere
    """
    image_set=image_set.drop(columns=['faceID','faceRectangle_height']) 
    image_set=image_set.rename(columns={"userId":"userid"})
    
    """ If multiple faces appear for a same user, we keep the biggest one """
    data_list=image_set['userid'].unique()
    max_faces=[]
    
    for user in data_list:
        user_image=image_set[image_set['userid']==user][['userid','faceRectangle_width']]
        max_faces.append(user_image['faceRectangle_width'].idxmax())
    
    image_set=image_set.iloc[max_faces]
    
    """ We define new more meaningful features : distances instead of points. The last computaton will be the facial hair
    feature, as the mean of facialHair_mustache, facialHair_beard, and facialHair_sideburns."""
    
    def dist(x_a,y_a,x_b,y_b):
        """ compute norm-2 distance between two 2-dimensional points."""
        return np.sqrt((x_b - x_a)**2 + (y_b - y_a)**2)
    
    image_dist=image_set.assign(
        eye_width=lambda df: 1/df['faceRectangle_width'] * 0.5 * (dist(df['eyeLeftInner_x'],df['eyeLeftInner_y'],df['eyeLeftOuter_x'],df['eyeLeftOuter_y'])+dist(df['eyeRightInner_x'],df['eyeRightInner_y'],df['eyeRightOuter_x'],df['eyeRightOuter_y'])),
        eye_height=lambda df: 1/df['faceRectangle_width'] * 0.5 * (dist(df['eyeLeftTop_x'],df['eyeLeftTop_y'],df['eyeLeftBottom_x'],df['eyeLeftBottom_y'])+dist(df['eyeRightTop_x'],df['eyeRightTop_y'],df['eyeRightBottom_x'],df['eyeRightBottom_y'])),
        pupil_dist=lambda df: 1/df['faceRectangle_width'] * (dist(df['pupilLeft_x'],df['pupilLeft_y'],df['pupilRight_x'],df['pupilRight_y'])),
        eyebrow_width=lambda df: 1/df['faceRectangle_width'] * 0.5 * (dist(df['eyebrowLeftOuter_x'],df['eyebrowLeftOuter_y'],df['eyebrowLeftInner_x'],df['eyebrowLeftInner_y'])+dist(df['eyebrowRightOuter_x'],df['eyebrowRightOuter_y'],df['eyebrowRightInner_x'],df['eyebrowRightInner_y'])),
        eyebrow_dist=lambda df: 1/df['faceRectangle_width'] * (dist(df['eyebrowLeftInner_x'],df['eyebrowLeftInner_y'],df['eyebrowRightInner_x'],df['eyebrowRightInner_y'])),
        noseRoot_dist=lambda df: 1/df['faceRectangle_width'] * dist(df['noseRootLeft_x'],df['noseRootLeft_y'],df['noseRootRight_x'],df['noseRootRight_y']),
        noseAlarTop_dist=lambda df: 1/df['faceRectangle_width'] * dist(df['noseLeftAlarTop_x'],df['noseLeftAlarTop_y'],df['noseRightAlarTop_x'],df['noseRightAlarTop_y']),
        noseAlarOut_dist=lambda df: 1/df['faceRectangle_width'] * dist(df['noseLeftAlarOutTip_x'],df['noseLeftAlarOutTip_y'],df['noseRightAlarOutTip_x'],df['noseRightAlarOutTip_y']),
        nose_height=lambda df: 1/df['faceRectangle_width'] * dist(0.5*(df['noseRootLeft_x']+df['noseRootRight_x']),0.5*( df['noseRootLeft_y']+df['noseRootRight_y']),df['noseTip_x'],df['noseTip_y']),
        upperLip_height=lambda df: 1/df['faceRectangle_width'] * dist(df['upperLipTop_x'],df['upperLipTop_y'],df['upperLipBottom_x'],df['upperLipBottom_y']),
        underLip_height=lambda df: 1/df['faceRectangle_width'] * dist(df['underLipTop_x'],df['underLipTop_y'],df['underLipBottom_x'],df['underLipBottom_y']),
        mouth_width=lambda df: 1/df['faceRectangle_width'] * dist(df['mouthLeft_x'],df['mouthLeft_y'],df['mouthRight_x'],df['mouthRight_y']),
        facialHair=lambda df: 1/3 * (df['facialHair_mustache']+df['facialHair_beard']+df['facialHair_sideburns'])
    )
    final_image=image_dist[['userid', 'eye_width', 'eye_height', 'pupil_dist', 'eyebrow_width', 'eyebrow_dist',
                            'noseRoot_dist','noseAlarTop_dist', 'noseAlarOut_dist', 'nose_height',
                            'upperLip_height', 'underLip_height','mouth_width', 'facialHair']]
    
    """ After reading some articles, it seems that the ratios play an important role in facial sexual dimorphism 
    That is why we compute now relevant ratios of distances. these will be our final extracted features, beside 
    the facial hair
    """
  
    final_image_ratio=final_image.assign(
        EyeNose_ratio=lambda df: df['pupil_dist']/df['nose_height'],
        Eyes_ratio=lambda df: df['eyebrow_width']/df['eye_height'],
        Nose_ratio=lambda df:df['noseAlarOut_dist']/df['nose_height'],
        NoseAlar_ratio=lambda df:df['noseAlarOut_dist']/df['nose_height'],
        Eyebrow_ratio=lambda df:(df['eyebrow_dist']+df['eyebrow_width'])/df['pupil_dist'],
        upperLip_ratio=lambda df:df['upperLip_height']/df['mouth_width'],
        underLip_ratio=lambda df:df['underLip_height']/df['mouth_width']
    )
    final_image_ratio=final_image_ratio.set_index('userid').sort_index()
    return final_image_ratio

# let's extract ratios and facial hair features from oxford data by using image_preprocess function
image=image_preprocess(image)
image=image.join(other= data_gender[['userid','gender']].set_index('userid')).sort_index()


# splitting Training set function

def SplitSet(data, crossSize=1000, seed=1):
    """
    This function takes as argument a dataframe, the desired size of validation /or test set and
    return 2 subsets of this dataframe, randomly chosen. The first one is the test set (or validation set) of 
    sample size equal to crossSize; and the second one is the train set. The use of the seed
    here is to allow possibility to recover the same split when entering the same data and crossSize 
    arguments
    """
    size=data.shape[0]
    cross_index= list(pd.Series(range(size)).sample(n=crossSize, replace=False, random_state=seed))
    cross_set=data.iloc[cross_index,:].copy()
    train_index=[i for i in range(size) if i not in cross_index]
    train_set=data.iloc[train_index,:].copy()
    return [cross_set, train_set]

# Now we split our image features
Valid_set , Train_set = SplitSet(image)

Train_set0=Train_set.copy()


def standardize(cross, train):
    """ This extraction function transforms features from both train and validation sets by 
    centering them with train set means and dividing them by train set standard deviations
    """
    val=cross.copy()
    for var in val.columns.values:
        val.loc[var] =(val[var] - train[var].mean())/(train[var].std())
    return val

def ReplaceOutlier(data, stat):
    """
    This function replace each feature (in data) outliers by the stat computed in for this feature. 
    stat is a statistical function, like, np.mean, np.median, etc. A value is considered as outlier if
    it is absolutely greater than 3.
    """
    data0=data.copy()
    feature_list = data.drop(columns=['facialHair','gender']).columns
    for feature in feature_list:
        stat_ = stat(data[feature])
        outlier= (np.abs(data[feature]) > 3)
        data0.loc[outlier,[feature]]=stat_
    return data0

print("Training...\n")

# Process the training set : Removing outliers

Train_set.loc[Train_set['gender']==1.,['facialHair']]=0. # remove facial hair of women in training

Valid_set.iloc[:,:-1]= standardize(Valid_set.iloc[:,:-1], Train_set0.iloc[:,:-1])
Train_set.iloc[:,:-1]= standardize(Train_set.iloc[:,:-1], Train_set0.iloc[:,:-1])

# Outlier processing on the rest of the training set
F=Train_set.gender==1; M=Train_set.gender==0
Train_set.loc[F]= ReplaceOutlier(Train_set[F], np.mean)
Train_set.loc[M]= ReplaceOutlier(Train_set[M], np.mean)


## Machine Learning prediction

modelnb=GaussianNB()
modelnb.fit(Train_set.iloc[:,:-1],Train_set.iloc[:,-1])


# Accuracy on the validation set (88%)
# modelnb.score(Valid_set.iloc[:,:-1],Valid_set.iloc[:,-1])


# Handling of the missing images

""" Old method (majority class on the profile that did not have any images """

# number_of_missing_men_image=sum(data_profile['gender']==0.)-sum(image['gender']==0.)
# number_of_missing_image=len(data_profile)-len(image)
# number_of_missing_women_image=number_of_missing_image-number_of_missing_men_image
# majority_missing_image=0.
# if number_of_missing_women_image>number_of_missing_men_image:majority_missing_image=1.

""" 
New method : we load a pretrained MLP on the page likes (relation) for gender prediction : 
84.73% on a validation set.
This alternative is performed for user without image features.
"""

genderMLPmodel = pickle.load(open('genderMLPmodel', 'rb'))

    

############# PERSONALITY ##############

# Loading the messages datas and personality traits

data_personality=pd.read_csv('/home/mila/teaching/user22/Train/Profile/Profile.csv',usecols=['userid', 'ope', 'con', 'ext', 'agr', 'neu'])
data_liwc=pd.read_csv('/home/mila/teaching/user22/Train/Text/liwc.csv')
data_nrc=pd.read_csv('/home/mila/teaching/user22/Train/Text/nrc.csv')

# Some column names issues...
data_nrc=data_nrc.rename(columns={"userId":"userid", 'anger':'Anger'})
data_liwc=data_liwc.rename(columns={'userId':'userid'})

# Merge the text (nrc and liwc) data
data_messages=data_liwc.set_index('userid').join(other=[data_nrc.set_index('userid'),data_personality.set_index('userid')])

Valid_set_messages , Train_set_messages = SplitSet(data_messages)

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

# we extract PCA features on the (merged) text data (excluding personality traits).

pca=PCA()
x_messages=pca.fit_transform(Train_set_messages.iloc[:,:-5])
x_valid_messages=pca.transform(Valid_set_messages.iloc[:,:-5])


# RMSE Score

def RMSE(predictions, labels):
    if (predictions.shape)!=(labels.shape): raise Exception('The dimensions of the predictions and the labels do not match')
    n= predictions.shape[0]
    if len(predictions.shape)==2: n*=predictions.shape[1]
    return np.sqrt(np.mean((predictions-labels)**2, axis=0))

# Personality Models/ Validation RMSE :(ope = 0.6075, con = 0.7288, ext = 0.7977, agr = 0.6415, neu = 0.7946)

ope_model=LinearRegression()
ope_model.fit(x_messages[:,0:37],Train_set_messages.iloc[:,-5])

con_model=LinearRegression()
con_model.fit(x_messages[:,0:20],Train_set_messages.iloc[:,-4])

""" Here are parameters for MLP performed on extroverion trait """

params={'activation': 'logistic',
 'alpha': 0.0001,
 'batch_size': 'auto',
 'beta_1': 0.9,
 'beta_2': 0.999,
 'early_stopping': True,
 'epsilon': 1e-08,
 'hidden_layer_sizes': (100, 100),
 'learning_rate': 'constant',
 'learning_rate_init': 0.001,
 'max_iter': 200,
 'momentum': 0.9,
 'n_iter_no_change': 10,
 'nesterovs_momentum': True,
 'power_t': 0.5,
 'random_state': 41,
 'shuffle': True,
 'solver': 'adam',
 'tol': 0.0001,
 'validation_fraction': 0.1,
 'verbose': False,
 'warm_start': False}


ext_model=MLPRegressor()
ext_model.set_params(**params)
ext_model.fit(Train_set_messages.iloc[:,:-5],Train_set_messages.iloc[:,-3])

agr_model=LinearRegression()
agr_model.fit(x_messages[:,0:20],Train_set_messages.iloc[:,-2])

neu_model=LinearRegression()
neu_model.fit(x_messages[:,0:20],Train_set_messages.iloc[:,-1])


########## AGE ##########

# Loading a pretrained model

"""
Because of heavy computation concern, we pretrained a MLP model on relation data, each user being represented
by a one hot encoding of pages. We start by loading this predicted model
"""

Age_model = pickle.load(open('model', 'rb'))

def relation_preprocess(test_relation):
    likes = pickle.load(open('likes', 'rb'))
    like_dict = pickle.load(open('like_dict', 'rb'))

    test_users=np.unique(test_relation.userid)
    test_user_dict=dict((user,i) for i,user in enumerate(test_users))

    # LIL is better for sparsed matrix assignments
    test_likes=lil_matrix((len(test_users),len(likes)),dtype=np.int32)

    for i in range(test_relation.shape[0]):
        if test_relation.like_id[i] in likes:
            test_likes[test_user_dict[test_relation.userid[i]],like_dict[test_relation.like_id[i]]]+=1

    # CSR is better for matrix operations
    test_likes=csr_matrix(test_likes)
    
    return test_likes


# Applying on the test set

print('Predicting...\n')
test_profile=pd.read_csv(INPUT+'Profile/Profile.csv')
test_profile = test_profile.set_index('userid').sort_index()
test_profile = test_profile.drop(columns='Unnamed: 0')

# Image

test_image=pd.read_csv(INPUT+'Image/oxford.csv')
test_image=image_preprocess(test_image)

# Text

test_liwc=pd.read_csv(INPUT+'Text/liwc.csv')
test_nrc=pd.read_csv(INPUT+'Text/nrc.csv')

test_nrc=test_nrc.rename(columns={"userId":"userid", 'anger':'Anger'})
test_liwc=test_liwc.rename(columns={'userId':'userid'})
test_messages=test_liwc.set_index('userid').join(other=[test_nrc.set_index('userid')])
test_messages=test_messages.sort_index()
x_test_messages=pca.transform(test_messages)

x_test_ope=x_test_messages

# Relations

test_relation=pd.read_csv(INPUT+'Relation/Relation.csv')
test_likes=relation_preprocess(test_relation)


test_image.iloc[:,:-1]= standardize(test_image.iloc[:,:-1], Train_set0.iloc[:,:-1])
for i,user in enumerate(test_messages.index):
#     user_gender=majority_missing_image (old method)
    if user in test_image.index: user_gender=modelnb.predict(test_image.loc[[user]])[0]
    else: user_gender=genderMLPmodel.predict(test_likes[i])[0]
    user_ope=ope_model.predict(x_test_messages[[i],:37])[0]
    user_con=con_model.predict(x_test_messages[[i],:20])[0]
    user_ext=ext_model.predict(test_messages.iloc[[i]])[0]
    user_agr=agr_model.predict(x_test_messages[[i],:20])[0]
    user_neu=neu_model.predict(x_test_messages[[i],:20])[0]
    user_age=Age_model.predict(test_likes[i])[0]
    
    pred=User(user, gender=user_gender, ope=str(user_ope), con=str(user_con), ext=str(user_ext), agr=str(user_agr), neu=str(user_neu),age=user_age)
    pred.output()

    
print("The outputs have been created....!!")
    
    

