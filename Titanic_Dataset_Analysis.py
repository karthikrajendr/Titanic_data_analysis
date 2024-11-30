#!/usr/bin/env python
# coding: utf-8

# In[139]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import pandas_profiling
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
from subprocess import check_output


# In[140]:


titanic_data=pd.read_csv("titanic_raw_data.csv")
titanic_data.shape


# In[141]:


titanic_data.head()


# In[142]:


type(titanic_data)


# In[143]:


titanic_data.info()


# In[144]:


#cabin, passenger ID and name are not important and can be dropped
titanic_data.columns


# In[145]:


titanic_data.tail()


# In[146]:


titanic_data.isnull().sum()


# In[147]:


#age null can be replaced by median if skewed and mean if normal distribution
#profile=ProfileReport(titanic_data)
#profile.to_file(output_file="titanic_before_processing.html")


# In[148]:


titanic_data[titanic_data['Embarked'].isnull()]


# In[149]:


titanic_data['Embarked'].mode()


# In[150]:


titanic_data.Embarked.fillna(titanic_data['Embarked'].mode()[0],inplace=True)
#titanic_data.Embarked.dropna(inline=True)


# In[151]:


titanic_data.info()


# In[152]:


#percentage survived in the age missing dataset.
(titanic_data[titanic_data['Age'].isnull()]['Survived'].sum()/titanic_data[titanic_data['Age'].isnull()].Survived.value_counts().sum())*100


# In[153]:


titanic_data[titanic_data['Age'].isnull()]['Survived'].sum()


# In[154]:


titanic_data[titanic_data['Age'].isnull()].Survived.value_counts().sum()


# In[155]:


(titanic_data.Survived.sum()/titanic_data.Survived.value_counts().sum())*100


# In[156]:


titanic_data.Age.median()


# In[157]:


titanic_data[titanic_data['Survived']==0]['Age'].median()


# In[158]:


median_age=titanic_data.Age.median()
titanic_data['Age']=titanic_data.Age.fillna(median_age)


# In[159]:


titanic_data[titanic_data['Age'].isnull()].Age


# In[160]:


titanic_data.drop('Cabin',axis=1,inplace=True)


# In[161]:


titanic_data.head(1)


# In[162]:


titanic_data.drop('Ticket', axis=1, inplace=True)
titanic_data.drop('PassengerId', axis=1, inplace=True)
titanic_data.head()


# In[163]:


titanic_data[titanic_data.Fare==0].Age


# In[164]:


titanic_data.Fare.median()


# In[165]:


titanic_data['Fare']=titanic_data['Fare'].replace(0,titanic_data.Fare.median())


# In[166]:


titanic_data[titanic_data.Fare==0].Age


# In[167]:


print(titanic_data.Fare.mean())
print(titanic_data.Fare.median())
print(titanic_data.groupby('Pclass')['Fare'].mean())


# In[168]:


titanic_data['GenderClass']=titanic_data.apply(lambda x: 'child' if x['Age']<15 else x['Sex'], axis=1)


# In[169]:


titanic_data.apply(lambda x: 'child' if x ['Age']<15 else x['Sex']>15 axis=1 ) 'child' if x['Age']<=15 else x['Sex'], axis=1


# In[170]:


titanic_data[titanic_data.Age>15].head()
#the GenerClass column value changes


# In[171]:


titanic_data.head()


# In[172]:


#ne2w col-Family Size
titanic_data['FamilySize']=titanic_data['SibSp']+titanic_data['Parch']+1


# In[173]:


titanic_data.head()


# In[174]:


sns.countplot(
    x='FamilySize',
    data=titanic_data)
plt.show()
#most of the passengers travelled solo.


# In[175]:


titanic_data.head()


# In[176]:


titanic_data.drop('Sex', axis=1, inplace=True)


# In[177]:


titanic_data.drop('Name', axis=1, inplace=True)


# In[178]:


titanic_data.head(2)


# In[179]:


titanic_data.drop_duplicates(inplace=True)


# In[180]:


titanic_data.head()


# In[181]:


titanic_data.info()


# In[182]:


pip install pandas-profiling


# In[183]:


import pandas as pd
from pandas_profiling import ProfileReport


# In[184]:


pip install --upgrade numpy


# In[185]:


pip install --upgrade pandas


# In[186]:


import pandas as pd
print(pd.__version__)


# In[187]:


import seaborn as sns
as_fig=sns.FacetGrid(titanic_data,hue='GenderClass', aspect=5)
as_fig.map(sns.kdeplot, 'Age', shade=True)
oldest=titanic_data['Age'].max()
as_fig.set(xlim=(0,oldest))
as_fig.add_legend()
plt.title('Age distribution using FacetGrid')


# In[ ]:


import numpy as np
print(np.__version__)


# In[188]:


titanic_data.groupby(['Survived','GenderClass','Pclass'])['Survived'].value_counts()


# In[190]:


titanic_data[titanic_data.GenderClass=='female']['Survived'].count()


# In[191]:


titanic_data[titanic_data.GenderClass=='female']['Survived'].sum()


# In[192]:


print("% of women survived: ", titanic_data[titanic_data.GenderClass=='female']['Survived'].sum())


# In[193]:


print("% of men survived: ", titanic_data[titanic_data.GenderClass=='male']['Survived'].sum())


# In[194]:


print("% of children survived: ", titanic_data[titanic_data.GenderClass=='child']['Survived'].sum())


# In[195]:


f,ax=plt.subplots(1,3,figsize=(20,7))
titanic_data['Survived'][titanic_data['GenderClass']=='male'].value_counts().plot.pie(explode=[0,0.2])
titanic_data['Survived'][titanic_data['GenderClass']=='female'].value_counts().plot.pie(explode=[0,0.2])
titanic_data['Survived'][titanic_data['GenderClass']=='child'].value_counts().plot.pie(explode=[0,0.2])
ax[0].set_title('Survived (male)')
ax[1].set_title('Survived (female)')
ax[2].set_title('Survived (child)')


# In[196]:


titanic_data['Survived'][titanic_data['GenderClass']=='male'].value_counts()


# In[197]:


titanic_data['Survived'][titanic_data['GenderClass']=='female'].value_counts()


# In[198]:


print("% of Survival in PClass=1: ", titanic_data[titanic_data.Pclass==1]['Survived'].sum()/titanic_data[titanic_data.Pclass==1]['Survived'].count()*100)
print("% of Survival in PClass=2: ", titanic_data[titanic_data.Pclass==2]['Survived'].sum()/titanic_data[titanic_data.Pclass==2]['Survived'].count()*100)
print("% of Survival in PClass=3: ", titanic_data[titanic_data.Pclass==3]['Survived'].sum()/titanic_data[titanic_data.Pclass==3]['Survived'].count()*100)


# In[199]:


sns.violinplot('Pclass','Survived',kind='point',data=titanic_data)
plt.title('Violin plot Pclass vs Survived')
plt.show()


# In[200]:


pd.crosstab([titanic_data.GenderClas, titanic_data.Survived], titanic_data.Pclass, margins=True).apply(lambda r: 100*r/len(titanic_data),axis=1).style.background_gradient(cmap='autumn_r')


# In[ ]:


titanic_data=sns.load_dataset("titanic")
titanic_data.head()
sns.violinplot(x="Pclass",y="age",data=titanic_data, hue="class",palette="muted", kind="point")
plt.title("Violin plot Pclass vs Survived")
plt.show()


# In[201]:


sns.catplot(x='Embarked',y='Survived',kind='point',data=titanic_data)
plt.title("Factorplot for embarked and survived")
plt.show()


# In[202]:


sns.countplot("Embarked", data=titanic_data, hue='GenderClass')
plt.show()


# In[203]:


sns.catplot(x='Embarked',y='Survived', kind='point', hue='GenderClass', data=titanic_data)
plt.title('Factorplot for embarked and sruvived')
plt.show()


# In[205]:


relation=pd.crosstab(titanic_data.Embarked, titanic_data.Pclass)
relation.plot.barh(figsize=(15,5))
plt.xticks(size=10)
plt.yticks(size=10)
plt.title('Relation between Pclass and Embarked', size=20)


# In[206]:


sns.set(style='whitegrid',palette='muted')
sns.swarmplot(x='Embarked',y='Age',hue='GenderClass',palette='gnuplot',data=titanic_data)


# In[209]:


sns.catplot(x='Embarked',y='Survived',col='Pclass',hue='GenderClass',kind='point',data=titanic_data)
plt.show()


# In[212]:


for i in range(4,0,-1):
    titanic_data.loc[titanic_data['Age']<=i*20, 'Age_bin']=i


# In[213]:


titanic_data.head()


# In[214]:


titanic_data.plot.hexbin(x='Age_bin',y='Survived',gridsize=12,legend=True)


# In[216]:


sns.barplot(x='Age_bin',y='Survived',hue='Pclass',data=titanic_data)
plt.show()


# In[217]:


titanic_data.groupby('Pclass')['Age_bin'].value_counts()


# In[219]:


titanic_data[(titanic_data.Age_bin==1)]['Pclass'].value_counts()


# In[220]:


titanic_data[(titanic_data.Age_bin==8)]['Age'].value_counts()


# In[221]:


sns.catplot(x='Age_bin', y='Survived', kind='point',data=titanic_data)
plt.show()


# In[222]:


sns.catplot(x='Age_bin', y='Survived', kind='GenderClass',data=titanic_data)
plt.show()


# In[224]:


sns.catplot(x='Age_bin', y='Survived', col='Pclass', row='GenderClass', hue='Embarked', kind='point', data=titanic_data)
plt.show()


# In[225]:


ax=sns.catplot(x='FamilySize', y='Survived', data=titanic_data, kind='violin', aspect=1.5, size=6, palette='Greens')
ax.set(ylabel='Percentage of Passengers')
plt.title('Survival by total family size')


# In[226]:


sns.distplot(titanic_data['Fare'], color='g')
plt.title('Distribution of Fare')
plt.show()


# In[229]:


for i in range(8,0,-1):
    titanic_data.loc[titanic_data['Fare']<=i*12, 'Fare_bin']=i
titanic_data.loc[titanic_data['Fare']>86, 'Fare_bin']=8


# In[230]:


titanic_data[['Fare','Fare_bin']].groupby('Fare_bin')['Fare'].count()


# In[231]:


sns.distplot(titanic_data['Fare_bin'],color='g')
plt.title('Distribution of Fare bin')
plt.show()


# In[234]:


print(titanic_data['Fare'].mean())
print(titanic_data['Fare'].median())


# In[235]:


fig, ax=plt.subplots(figsize=(8,8))
sns.barplot(x='Fare_bin',y='Survived',hue='Pclass',data=titanic_data, ax=ax)
plt.show()


# In[236]:


fig, ax=plt.subplots(figsize=(20,20))
sns.barplot(x='Fare_bin',y='Survived',hue='Pclass',data=titanic_data, ax=ax)
plt.show()


# In[237]:


sns.pairplot(titanic_data[['Fare','Age','Pclass','Survived']], vars=['Fare','Age','Pclass'],hue='Survived',dropna=True, markers=['o','s'])
plt.show()


# In[241]:


corr=titanic_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, vmax=1, linewidth=0.01, square=True, annot=True, cmap='YlGnBu', linecolor='black')
plt.title('Correlation between features')
plt.show()

