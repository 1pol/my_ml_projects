import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
df = pd.read_csv("Bengaluru_House_Data.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.isna())
print(df.isna().sum())
df.drop(['society','availability'],axis =1,inplace = True)
df["location"].fillna(df["location"].mode()[0],inplace = True)
df["bath"].fillna(df["bath"].median(),inplace = True)
df["balcony"].fillna(df["balcony"].median(),inplace = True)
print(df.isna().sum())
df.dropna(inplace=True)
print(df.isna().sum())
print(df)
print(df['size'].unique())
df["BHK"]=df["size"].apply(lambda x:int(x.split(' ')[0]))
print(df.head(10))
def isfloat(x):
  try:
    float(x)
  except:
    return False
  return True
df[~df['total_sqft'].apply(isfloat)].head(10)
df=df.drop(['area_type'],axis = 'columns')


def convert_sqft_tonum(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None
    df=df.copy() 
'''
def convert_sqft_tonum(x):
    if isinstance(x, str):
        # Check if the value is a string
        if '-' in x:
            # If it's a range, take the average
            tokens = x.split('-')
            try:
                return (float(tokens[0]) + float(tokens[1])) / 2
            except ValueError:
                # If the conversion fails, return NaN
                return np.nan
        try:
            # If it's a single value, convert it to float
            return float(x)
        except ValueError:
            # If the conversion fails, return NaN
            return np.nan
    # If it's already a number, return it as is
    return x

# Apply the modified function to the 'total_sqft' column
df['total_sqft'] = df['total_sqft'].apply(convert_sqft_tonum)'''

df['total_sqft']=df['total_sqft'].apply(convert_sqft_tonum)
print(df.head(10))
print(df.loc[30])
print(df.head())
df1=df.copy()
df1['price_per_sqft']=df1['price']*100000/df1['total_sqft']
print(df1.head())
print(len(df1.location.unique()))
df1.location=df1.location.apply(lambda x: x.strip())
location_stats=df1.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)
print(len(location_stats[location_stats<=10]))
locationlessthan10=location_stats[location_stats<=10]
print(locationlessthan10)
print(len(df1.location.unique()))
df1.location=df1.location.apply(lambda x: 'other' if x in locationlessthan10 else x)
print(len(df1.location.unique()))
print(df1.head(10))
print(df1[df1.total_sqft/df1.BHK<300].head())
df2=df1[~(df1.total_sqft/df1.BHK<300)]
print(df2.head(10))
print(df2.shape)
print(df2["price_per_sqft"].describe().apply(lambda x:format(x,'f')))
sns.boxplot(data=df)
print(plt.show())
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))& (subdf.price_per_sqft<(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df3=remove_pps_outliers(df2)
print(df3.shape)
import matplotlib.pyplot as plt
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location)&(df.BHK==2)]
    bhk3=df[(df.location==location)&(df.BHK==3)]
    plt.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='Blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color='green',marker='+',label='3 BHK',s=50)
    plt.xlabel('Total Square Foot')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
print(plot_scatter_chart(df3,"Rajaji Nagar"))
def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_sats={}
        for BHK,BHK_df in location_df.groupby('BHK'):
            bhk_sats[BHK]={
                'mean':np.mean(BHK_df.price_per_sqft),
                'std':np.std(BHK_df.price_per_sqft),
                'count':BHK_df.shape[0]
            }
        for BHK,BHK_df in location_df.groupby('BHK'):
            stats=bhk_sats.get(BHK-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df4=remove_bhk_outliers(df3)
print(df4.shape)
print(plot_scatter_chart(df4,"Rajaji Nagar"))
plt.rcParams['figure.figsize']=(15,8)
plt.hist(df4.price_per_sqft,rwidth=0.6)
plt.xlabel("Price Per Square Foor")
plt.ylabel("Count")
print(plt.show())
print(df4.bath.unique())
print(df4[df4.bath>10])
print(df3)
print(df4)
sns.boxplot(data=df['bath'])
print(plt.show())
plt.rcParams['figure.figsize']=(15,10)
plt.hist(df4.bath,rwidth=0.6)
plt.xlabel("Number Of Bathroom")
plt.ylabel("Count")
print(plt.show())
print(df4[df4.bath>df4.BHK+2])
df5=df4[df4.bath<df4.BHK+2]
print(df5.shape)
df6=df5.drop(['size','price_per_sqft','balcony'],axis='columns')
print(df6)
dummies=pd.get_dummies(df6.location)
dummies=dummies.astype(int)
print(dummies.head(10))
df7=pd.concat([df6,dummies.drop('other',axis='columns')],axis='columns')
print(df7.head())
df8=df7.drop('location',axis='columns')
print(df8.head())
print(df8.shape)
X=df8.drop('price',axis='columns')
print(X.head())
Y=df8.price
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test)*100)
def price_predict(location,sqft,bath,BHK):
    loc_index=np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=BHK
    if loc_index >=0:
        x[loc_index]=1
    return model.predict([x])[0]
print(price_predict('1st Phase JP Nagar',2000,2,4)*100000)
print(price_predict('Indira Nagar',1000,2,2)*100000)

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))



























































































































































































