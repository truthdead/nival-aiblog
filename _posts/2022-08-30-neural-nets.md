## Intro and Downloading data

Welcome to this exciting journey aboard Titanic Spaceship! 

This is tutorial is almost a complete copy of Jeremy Howard's excellent [Linear model and neural net from scratch](https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch) kaggle notebook which was part of the 2022 version of **Deep Learning for Coders** course by FastAI.

*This also borrows heavily from [Spaceship Titanic: A complete guide](https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide) notebook by Samuel Cortinhas which is based on the dataset I'm gonna use to build by neural network.*

Lastly and very importantly, Neural Network from Scratch series by sentdex youtuber channel's Harrison and Daniel was instrumental in broadening my understanding. I have burrowed their way of coding neural network layers as classes. You can access their video and the book [here](https://nnfs.io/)

My explanations/comments are going to be minimal, because everything is explained quite well in the resources mentioned above. For neural net foundations(which is the primary aim of this blog/notebook). please refer to course.fast.ai by Jeremy & Co. and the nnfs.io series.




- **Objective of dataset**: To predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly. 


- **My Aim** : is to replicate Jeremy's work on a different dataset(his work was based on original Titanic dataset), as part of my learning. 





*Let's go conquer a neural network then!*

Description of the features of the dataset, copied from the competition page:

- PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is   travelling with and pp is their number within the group. People in a group are often family members, but not always.

- HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.

- CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.

- Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.


- Destination - The planet the passenger will be debarking to.

- Age - The age of the passenger.

- VIP - Whether the passenger has paid for special VIP service during the voyage.

- RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.

- Name - The first and last names of the passenger.

- **Transported** - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.


```python
import numpy as np
import pandas as pd
import torch
import kaggle
import os
from pathlib import Path

```

Downloading the data from Titanic-spaceship competition via the kaggle API:


```python
path = Path('spaceship-titanic')
if not path.exists():
    import zipfile,kaggle
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)
```

Setting display options for numpy, pandas and pytorch to widen the output frames:


```python
np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)
```

## Cleaning the data

Looking at some samples from the dataset:


```python
df = pd.read_csv(path/'train.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Maham Ofracculy</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>Juanna Vines</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>Altark Susent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>Solam Susent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>Willy Santantines</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Exploring missing values and filling them with the **Mode** of the respective column:


```python
df.isna().sum()
```




    PassengerId       0
    HomePlanet      201
    CryoSleep       217
    Cabin           199
    Destination     182
    Age             179
    VIP             203
    RoomService     181
    FoodCourt       183
    ShoppingMall    208
    Spa             183
    VRDeck          188
    Name            200
    Transported       0
    dtype: int64




```python
modes = df.mode().iloc[0]
modes
```




    PassengerId                0001_01
    HomePlanet                   Earth
    CryoSleep                    False
    Cabin                      G/734/S
    Destination            TRAPPIST-1e
    Age                           24.0
    VIP                          False
    RoomService                    0.0
    FoodCourt                      0.0
    ShoppingMall                   0.0
    Spa                            0.0
    VRDeck                         0.0
    Name            Alraium Disivering
    Transported                   True
    Name: 0, dtype: object




```python
df.fillna(modes, inplace=True)
```


```python
df.isna().sum()
```




    PassengerId     0
    HomePlanet      0
    CryoSleep       0
    Cabin           0
    Destination     0
    Age             0
    VIP             0
    RoomService     0
    FoodCourt       0
    ShoppingMall    0
    Spa             0
    VRDeck          0
    Name            0
    Transported     0
    dtype: int64



## Exploratory Data Analysis & Feature Engineering


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('ggplot')
```


```python
# Figure size
plt.figure(figsize=(6,6))

# Pie plot
df['Transported'].value_counts().plot.pie(explode=[0.03,0.03], autopct='%1.1f%%', shadow=True, textprops={'fontsize':16}).set_title("Target distribution")
```




    Text(0.5, 1.0, 'Target distribution')




    
![png](output_22_1.png)
    


Target varaible is quite balanced, hence we don't need to perform over/undersampling.

Now let's describe categorical features:


```python
df.describe(include=[object])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8693</td>
      <td>8693</td>
      <td>8693</td>
      <td>8693</td>
      <td>8693</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>8693</td>
      <td>3</td>
      <td>6560</td>
      <td>3</td>
      <td>8473</td>
    </tr>
    <tr>
      <th>top</th>
      <td>0001_01</td>
      <td>Earth</td>
      <td>G/734/S</td>
      <td>TRAPPIST-1e</td>
      <td>Alraium Disivering</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>4803</td>
      <td>207</td>
      <td>6097</td>
      <td>202</td>
    </tr>
  </tbody>
</table>
</div>



Let's now replace the strings in these categorical features by numbers. Pandas offers a `get_dummies` method to convert these to numbers so that we can multiply them with weights. It's basically one-hot coding, letting the model know the unqiue levels available in a particular.

We only process HomePlanet and Destination via `get_dummies`because others simply have too many unique values (aka levels).


```python
df = pd.get_dummies(df, columns=["HomePlanet", "Destination"])
df.columns
```




    Index(['PassengerId', 'CryoSleep', 'Cabin', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name',
           'Transported', 'HomePlanet_Earth', 'HomePlanet_Europa', 'HomePlanet_Mars', 'Destination_55 Cancri e', 'Destination_PSO J318.5-22',
           'Destination_TRAPPIST-1e'],
          dtype='object')



Our dummy columns are visible at the end of the dataframe!

Looking at numerical features:


```python
df.describe(include=(np.number))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>HomePlanet_Earth</th>
      <th>HomePlanet_Europa</th>
      <th>HomePlanet_Mars</th>
      <th>Destination_55 Cancri e</th>
      <th>Destination_PSO J318.5-22</th>
      <th>Destination_TRAPPIST-1e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8693.000000</td>
      <td>8693.000000</td>
      <td>8693.000000</td>
      <td>8693.000000</td>
      <td>8693.000000</td>
      <td>8693.000000</td>
      <td>8693.000000</td>
      <td>8693.000000</td>
      <td>8693.000000</td>
      <td>8693.000000</td>
      <td>8693.000000</td>
      <td>8693.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28.728517</td>
      <td>220.009318</td>
      <td>448.434027</td>
      <td>169.572300</td>
      <td>304.588865</td>
      <td>298.261820</td>
      <td>0.552514</td>
      <td>0.245140</td>
      <td>0.202347</td>
      <td>0.207063</td>
      <td>0.091568</td>
      <td>0.701369</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.355438</td>
      <td>660.519050</td>
      <td>1595.790627</td>
      <td>598.007164</td>
      <td>1125.562559</td>
      <td>1134.126417</td>
      <td>0.497263</td>
      <td>0.430195</td>
      <td>0.401772</td>
      <td>0.405224</td>
      <td>0.288432</td>
      <td>0.457684</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>37.000000</td>
      <td>41.000000</td>
      <td>61.000000</td>
      <td>22.000000</td>
      <td>53.000000</td>
      <td>40.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>79.000000</td>
      <td>14327.000000</td>
      <td>29813.000000</td>
      <td>23492.000000</td>
      <td>22408.000000</td>
      <td>24133.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Samuel's notebook uncovered the following useful insight regarding `Age`. Let's visualize the feature first:


```python
plt.figure(figsize=(10,4))

# Histogram
sns.histplot(data=df, x='Age', hue='Transported', binwidth=1, kde=True)

# Aesthetics
plt.title('Age distribution')
plt.xlabel('Age (years)')
```




    Text(0.5, 0, 'Age (years)')




    
![png](output_32_1.png)
    


Notes and insights by Samuel:

*Notes:*
* 0-18 year olds were **more** likely to be transported than not.
* 18-25 year olds were **less** likely to be transported than not.
* Over 25 year olds were about **equally** likely to be transported than not.

*Insight:*
* Create a new feature that indicates whether the passanger is a child, adolescent or adult.



```python
p_groups = ['child', 'young', 'adult']

df['age_group'] = np.nan


    
        
```


```python
df.loc[df['Age'] < 18, 'age_group'] = p_groups[0]
df.loc[(df['Age'] >= 18) & (df['Age'] <= 25), 'age_group'] = p_groups[1]
df.loc[df['Age'] > 25, 'age_group'] = p_groups[2]
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
      <th>HomePlanet_Earth</th>
      <th>HomePlanet_Europa</th>
      <th>HomePlanet_Mars</th>
      <th>Destination_55 Cancri e</th>
      <th>Destination_PSO J318.5-22</th>
      <th>Destination_TRAPPIST-1e</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Maham Ofracculy</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>adult</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>Juanna Vines</td>
      <td>True</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>young</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>Altark Susent</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>adult</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>Solam Susent</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>adult</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>Willy Santantines</td>
      <td>True</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>child</td>
    </tr>
  </tbody>
</table>
</div>



Now we need dummies for `age_group` as well:


```python
df = pd.get_dummies(df, columns=["age_group"])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>...</th>
      <th>Transported</th>
      <th>HomePlanet_Earth</th>
      <th>HomePlanet_Europa</th>
      <th>HomePlanet_Mars</th>
      <th>Destination_55 Cancri e</th>
      <th>Destination_PSO J318.5-22</th>
      <th>Destination_TRAPPIST-1e</th>
      <th>age_group_adult</th>
      <th>age_group_child</th>
      <th>age_group_young</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>True</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>...</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>...</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>True</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Expenditure features
exp_feats=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Plot expenditure features
fig=plt.figure(figsize=(10,20))
for i, var_name in enumerate(exp_feats):
    # Left plot
    ax=fig.add_subplot(5,2,2*i+1)
    sns.histplot(data=df, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')
    ax.set_title(var_name)
    
    # Right plot (truncated)
    ax=fig.add_subplot(5,2,2*i+2)
    sns.histplot(data=df, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
    plt.ylim([0,100])
    ax.set_title(var_name)
fig.tight_layout()  # Improves appearance a bit
plt.show()
```


    
![png](output_39_0.png)
    


Insights:

- Luxuries such as VR decl and Spa were clearly more used by people who were NOT transported.
- Need to create a log form for all the above to reduce the skew of the distributions.


```python
for f in exp_feats:
    df[f'log{f}'] = np.log(df[f]+1)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>...</th>
      <th>Destination_PSO J318.5-22</th>
      <th>Destination_TRAPPIST-1e</th>
      <th>age_group_adult</th>
      <th>age_group_child</th>
      <th>age_group_young</th>
      <th>logRoomService</th>
      <th>logFoodCourt</th>
      <th>logShoppingMall</th>
      <th>logSpa</th>
      <th>logVRDeck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4.700480</td>
      <td>2.302585</td>
      <td>3.258097</td>
      <td>6.309918</td>
      <td>3.806662</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3.784190</td>
      <td>8.182280</td>
      <td>0.000000</td>
      <td>8.812248</td>
      <td>3.912023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>7.157735</td>
      <td>5.918894</td>
      <td>8.110728</td>
      <td>5.267858</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5.717028</td>
      <td>4.262680</td>
      <td>5.023881</td>
      <td>6.338594</td>
      <td>1.098612</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



Finally, we are splitting `PassengerId` as per its data description:


```python
# New feature - Group
df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
```


```python
df.columns
```




    Index(['PassengerId', 'CryoSleep', 'Cabin', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name',
           'Transported', 'HomePlanet_Earth', 'HomePlanet_Europa', 'HomePlanet_Mars', 'Destination_55 Cancri e', 'Destination_PSO J318.5-22',
           'Destination_TRAPPIST-1e', 'age_group_adult', 'age_group_child', 'age_group_young', 'logRoomService', 'logFoodCourt',
           'logShoppingMall', 'logSpa', 'logVRDeck', 'Group'],
          dtype='object')




```python
added_cols = ['HomePlanet_Earth', 'HomePlanet_Europa','HomePlanet_Mars', 'Destination_55 Cancri e', 'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e','age_group_adult', 'age_group_child', 'age_group_young', 'logRoomService', 'logFoodCourt',
       'logShoppingMall', 'logSpa', 'logVRDeck', 'Group']
```


```python
df[added_cols].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HomePlanet_Earth</th>
      <th>HomePlanet_Europa</th>
      <th>HomePlanet_Mars</th>
      <th>Destination_55 Cancri e</th>
      <th>Destination_PSO J318.5-22</th>
      <th>Destination_TRAPPIST-1e</th>
      <th>age_group_adult</th>
      <th>age_group_child</th>
      <th>age_group_young</th>
      <th>logRoomService</th>
      <th>logFoodCourt</th>
      <th>logShoppingMall</th>
      <th>logSpa</th>
      <th>logVRDeck</th>
      <th>Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4.700480</td>
      <td>2.302585</td>
      <td>3.258097</td>
      <td>6.309918</td>
      <td>3.806662</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3.784190</td>
      <td>8.182280</td>
      <td>0.000000</td>
      <td>8.812248</td>
      <td>3.912023</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>7.157735</td>
      <td>5.918894</td>
      <td>8.110728</td>
      <td>5.267858</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5.717028</td>
      <td>4.262680</td>
      <td>5.023881</td>
      <td>6.338594</td>
      <td>1.098612</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



What about `CryoSleep`?


```python
df.CryoSleep.unique()
```




    array([False,  True])



It's boolean, so we can mulitply with weights. VIP looks the same.


```python
indep_cols = ['Age', 'CryoSleep', 'VIP'] + added_cols
```

## Setting up a linear model

Single layer neural network with one neuron:


```python
np.random.seed(442)
weights = np.random.randn(len(indep_cols), 1)
bias = np.random.randn(1)
```


```python
preds = np.dot(df[indep_cols].values, weights)
```


```python
preds.shape
```




    (8693, 1)




```python
trn_indep = np.array(df[indep_cols], dtype='float32')
trn_dep = np.array(df['Transported'], dtype='float32')
```


```python
print(trn_indep.shape)
print(trn_dep.shape)
```

    (8693, 18)
    (8693,)


Single layer neural network with three neurons:


```python
np.random.seed(442)
weights = np.random.randn(len(indep_cols), 3) #three neurons in this layer
bias = np.random.randn(1, 3)
```


```python
bias
```




    array([[ 0.01176998, -0.85427749, -0.99987562]])




```python
preds = np.dot(df[indep_cols].values, weights) + bias
```


```python
preds.shape
```




    (8693, 3)




```python
preds[:10]
```




    array([[-48.06963390253141, -18.290557540354897, 25.89840533681499],
           [-37.91377002769904, -7.504076537856783, 19.929613327882404],
           [-77.52103870477566, -16.739210482170467, 48.30474706150288],
           [-49.956656061980745, -2.8342719146155777, 27.74823775227435],
           [-27.921019655098725, -0.8518280063674349, 7.996057055720733],
           [-58.52447309612352, -11.829816864262263, 30.01637869888577],
           [-43.57861207919503, -3.600338170392331, 14.143030534080323],
           [-42.93381590139605, -11.247805747916502, 13.576656755604825],
           [-53.425588340898884, -4.639261738916223, 17.59311661596172],
           [-25.380443498116176, -10.613520203166267, 1.9029395296299394]], dtype=object)



A deeper neural network with 2 hidden layers and an output layer:


```python
trn_dep = torch.tensor(trn_dep, dtype=torch.long)
trn_indep = torch.tensor(trn_indep, dtype=torch.float)
```


```python
trn_indep.shape
```




    torch.Size([8693, 18])




```python
trn_dep.shape
```




    torch.Size([8693])




```python
#collapse_output
trn_indep[:5]

```




    tensor([[39.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000,  0.0000,  1.0000],
            [24.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000,  4.7005,  2.3026,
              3.2581,  6.3099,  3.8067,  2.0000],
            [58.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  3.7842,  8.1823,
              0.0000,  8.8122,  3.9120,  3.0000],
            [33.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,  7.1577,
              5.9189,  8.1107,  5.2679,  3.0000],
            [16.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  5.7170,  4.2627,
              5.0239,  6.3386,  1.0986,  4.0000]])




```python
vals,indices = trn_indep.max(dim=0)
trn_indep = trn_indep / vals
```

Next let's define our layers, I'm going to define them as classes, so that we can reuse objects of each classes when needed. 


```python
#linear layer class
class linearlayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = ((torch.rand(n_inputs, n_neurons)-0.3)/n_neurons)*4
        self.weights = self.weights.requires_grad_()
        self.biases = ((torch.rand(1, n_neurons))-0.5)*0.1
        self.biases = self.biases.requires_grad_()
    def forward(self, inputs):
        self.output = inputs@self.weights + self.biases
    
        
        
```


```python
#ReLU activation function
class ReLU_act:
    def forward(self, inputs):
        self.output = torch.clip(inputs, 0.)
```


```python
#Softmax - for the last layer
class Softmax_act:
    def forward(self, inputs):
        exp_values = torch.exp(inputs)
        probs = exp_values/torch.sum(exp_values, axis=1, keepdims=True)
        self.output = probs
```


```python
n_inputs=len(indep_cols)
n_hidden=10
inputs=trn_indep
y_true=trn_dep
```


```python
n_inputs
```




    18



Initializing the parameters of our network:


```python
    #collapse_output
    layer1 = linearlayer(n_inputs, n_hidden)
    relu1 = ReLU_act()
    layer2 = linearlayer(n_hidden, n_hidden)
    relu2 = ReLU_act()
    layer3 = linearlayer(n_hidden, 2)
    wandbs = layer1.weights, layer2.weights, layer3.weights, layer1.biases, layer2.biases, layer3.biases
    print('These are our weights and biases:')
    print(wandbs)
    layer1.forward(inputs)
    print("\n")
    print('These are our l1 outputs:')
    print(layer1.output)
    relu1.forward(layer1.output)
    print("\n")
    print('These are our r1 outputs:')
    print(relu1.output)
    layer2.forward(relu1.output)
    print("\n")
    print('These are our l2 outputs:')
    print(layer2.output)
    relu2.forward(layer2.output)
    print("\n")
    print('These are our r2 outputs:')
    print(relu2.output)
    layer3.forward(relu2.output)
    print("\n")
    print('These are our l3 outputs:')
    print(layer3.output)
    softmax = Softmax_act()
    softmax.forward(layer3.output)
    
    print("\n")
    print('These are our softmax outputs:')
    print(softmax.output)
    
    
```

    These are our weights and biases:
    (tensor([[    -0.0002,      0.2331,      0.0117,      0.2600,      0.2774,      0.1307,      0.1950,      0.2057,      0.1399,     -0.0986],
            [     0.1946,     -0.0915,      0.0643,      0.0087,      0.2372,      0.2070,      0.2365,      0.2561,      0.2278,     -0.0841],
            [     0.0705,     -0.1123,     -0.0432,      0.2374,      0.1913,      0.1934,      0.0814,     -0.0839,     -0.1167,      0.2246],
            [     0.0705,      0.1724,      0.1248,      0.0990,      0.1931,     -0.0708,     -0.0022,      0.2181,      0.0299,      0.1734],
            [     0.1153,     -0.1066,     -0.0893,      0.1900,      0.2777,      0.1359,      0.1473,      0.1143,      0.2712,     -0.0556],
            [     0.1509,      0.1151,      0.0681,     -0.0028,      0.2353,      0.1948,      0.0870,      0.0006,     -0.0163,      0.0389],
            [    -0.0553,      0.2512,      0.0427,      0.2267,     -0.0093,      0.1136,     -0.1013,      0.1188,     -0.0013,      0.1935],
            [     0.1943,      0.2582,      0.1152,      0.2282,      0.1577,      0.0751,      0.0589,      0.0211,     -0.0506,      0.1875],
            [     0.0339,      0.1409,      0.0418,      0.1518,      0.0863,      0.0752,     -0.0287,      0.0064,     -0.0014,      0.0103],
            [    -0.1056,      0.2283,      0.1243,      0.1812,      0.1591,      0.1607,      0.1488,     -0.0419,     -0.0072,     -0.0633],
            [    -0.1127,     -0.0293,      0.1272,      0.0205,      0.0115,      0.2260,     -0.0251,      0.2198,     -0.0274,      0.0513],
            [     0.0951,      0.0732,      0.2631,      0.1341,     -0.0821,      0.0232,      0.1464,      0.0473,      0.1390,     -0.0458],
            [     0.1315,      0.0439,      0.1412,     -0.0664,      0.0940,      0.1332,     -0.0286,      0.0506,      0.0248,      0.0755],
            [     0.2530,      0.2420,      0.2765,      0.2048,     -0.0882,     -0.0035,      0.1577,      0.0605,      0.0726,      0.2733],
            [     0.2079,      0.0583,      0.0246,      0.1822,     -0.0031,      0.1719,     -0.0781,      0.2127,      0.2108,      0.0760],
            [    -0.0597,     -0.0759,      0.1393,      0.2165,      0.2028,      0.0627,     -0.0170,     -0.0651,      0.0579,      0.0773],
            [     0.1473,     -0.0495,      0.2317,      0.0157,     -0.1144,     -0.0419,      0.1788,      0.2388,      0.1428,      0.0085],
            [     0.0061,      0.2621,      0.2392,      0.0263,     -0.1017,     -0.0018,      0.1258,      0.1266,      0.1345,     -0.0650]],
           requires_grad=True), tensor([[-0.0621, -0.0691,  0.2171, -0.0623,  0.1145,  0.0525,  0.2408,  0.2170,  0.1916,  0.0629],
            [-0.0341, -0.0009, -0.0881,  0.1173,  0.0511, -0.0289,  0.1675,  0.2559,  0.0796, -0.0451],
            [ 0.1161,  0.0154, -0.0154, -0.0897,  0.1087, -0.0746, -0.0648, -0.0229,  0.1991,  0.2154],
            [ 0.0914,  0.0735, -0.0821, -0.0773,  0.0863,  0.0284, -0.0972,  0.1577,  0.1829, -0.0549],
            [ 0.0412,  0.1230,  0.2663,  0.0677,  0.2552, -0.0593,  0.0626,  0.1118, -0.0184,  0.1215],
            [-0.0723,  0.1951,  0.0814,  0.1907, -0.0819,  0.0893,  0.2131, -0.0612,  0.0357,  0.0940],
            [ 0.2410,  0.0710, -0.0211,  0.1043,  0.0940,  0.0236, -0.1108, -0.0611,  0.2035, -0.0306],
            [ 0.1431,  0.2022,  0.1448, -0.1078, -0.0829,  0.1917,  0.2769,  0.1859,  0.0531,  0.1962],
            [ 0.0767,  0.2175,  0.1521, -0.0540,  0.0504,  0.2353,  0.2760, -0.1082,  0.1134,  0.2455],
            [ 0.2477,  0.2034,  0.1353,  0.0091, -0.0367,  0.2536,  0.0904,  0.1440, -0.1072, -0.0786]], requires_grad=True), tensor([[ 0.6042,  0.9942],
            [ 1.1100,  0.6981],
            [ 0.2269, -0.0373],
            [ 0.8578,  0.3621],
            [ 1.1303,  0.1530],
            [ 1.2218,  0.6881],
            [ 0.4251,  0.7249],
            [-0.3872,  0.6244],
            [ 1.3207, -0.3660],
            [ 0.7619,  0.8244]], requires_grad=True), tensor([[ 0.0031,  0.0179, -0.0161, -0.0428, -0.0240, -0.0101, -0.0082,  0.0291,  0.0329, -0.0085]], requires_grad=True), tensor([[-0.0072,  0.0279, -0.0105, -0.0237, -0.0117,  0.0383,  0.0107, -0.0042,  0.0435,  0.0144]], requires_grad=True), tensor([[-0.0298, -0.0055]], requires_grad=True))
    
    
    These are our l1 outputs:
    tensor([[ 0.0465,  0.3955,  0.0665,  0.6085,  0.6360,  0.4262,  0.3555,  0.2094,  0.3645, -0.1657],
            [ 0.4090,  0.5032,  0.7315,  0.6355,  0.3676,  0.2011,  0.2193,  0.5196,  0.4299,  0.2742],
            [ 0.3745,  0.4630,  0.5139,  1.2414,  0.9952,  0.7398,  0.6521,  0.2783,  0.4553,  0.3533],
            [ 0.3732,  0.4930,  0.5060,  1.0217,  0.6563,  0.5438,  0.4839,  0.4328,  0.6498,  0.1433],
            [ 0.2600,  0.4513,  0.6046,  0.5558,  0.4570,  0.4457, -0.0068,  0.6613,  0.2647,  0.4525],
            [ 0.2802,  0.9090,  0.5997,  0.8559,  0.7023,  0.2610,  0.3909,  0.3403,  0.1594,  0.4420],
            [ 0.2624,  0.8340,  0.5346,  0.6196,  0.4794,  0.2714,  0.2643,  0.3717,  0.1908,  0.3143],
            ...,
            [ 0.2636,  0.7582,  0.6401,  0.8374,  0.4956,  0.4299,  0.6259,  0.4212,  0.6110,  0.0037],
            [ 0.3290,  0.7838,  0.6694,  0.7983,  0.3901,  0.3895,  0.6834,  0.4900,  0.6283,  0.0081],
            [ 0.2696,  0.7915,  0.7026,  1.2959,  0.6626,  0.6849,  0.6939,  0.4754,  0.5521,  0.4695],
            [ 0.5637,  0.7453,  0.7932,  0.5127,  0.4434,  0.2524,  0.6015,  0.7450,  0.5453,  0.1351],
            [ 0.1594,  0.9366,  0.5459,  0.6524,  0.4158,  0.3292,  0.2400,  0.5606,  0.3965,  0.0767],
            [ 0.2173,  0.8266,  0.7594,  0.9643,  0.3819,  0.4520,  0.6310,  0.6241,  0.6840,  0.1978],
            [ 0.3641,  0.8806,  0.6636,  0.7896,  0.4980,  0.4865,  0.6539,  0.4849,  0.6162,  0.0276]], grad_fn=<AddBackward0>)
    
    
    These are our r1 outputs:
    tensor([[0.0465, 0.3955, 0.0665, 0.6085, 0.6360, 0.4262, 0.3555, 0.2094, 0.3645, 0.0000],
            [0.4090, 0.5032, 0.7315, 0.6355, 0.3676, 0.2011, 0.2193, 0.5196, 0.4299, 0.2742],
            [0.3745, 0.4630, 0.5139, 1.2414, 0.9952, 0.7398, 0.6521, 0.2783, 0.4553, 0.3533],
            [0.3732, 0.4930, 0.5060, 1.0217, 0.6563, 0.5438, 0.4839, 0.4328, 0.6498, 0.1433],
            [0.2600, 0.4513, 0.6046, 0.5558, 0.4570, 0.4457, 0.0000, 0.6613, 0.2647, 0.4525],
            [0.2802, 0.9090, 0.5997, 0.8559, 0.7023, 0.2610, 0.3909, 0.3403, 0.1594, 0.4420],
            [0.2624, 0.8340, 0.5346, 0.6196, 0.4794, 0.2714, 0.2643, 0.3717, 0.1908, 0.3143],
            ...,
            [0.2636, 0.7582, 0.6401, 0.8374, 0.4956, 0.4299, 0.6259, 0.4212, 0.6110, 0.0037],
            [0.3290, 0.7838, 0.6694, 0.7983, 0.3901, 0.3895, 0.6834, 0.4900, 0.6283, 0.0081],
            [0.2696, 0.7915, 0.7026, 1.2959, 0.6626, 0.6849, 0.6939, 0.4754, 0.5521, 0.4695],
            [0.5637, 0.7453, 0.7932, 0.5127, 0.4434, 0.2524, 0.6015, 0.7450, 0.5453, 0.1351],
            [0.1594, 0.9366, 0.5459, 0.6524, 0.4158, 0.3292, 0.2400, 0.5606, 0.3965, 0.0767],
            [0.2173, 0.8266, 0.7594, 0.9643, 0.3819, 0.4520, 0.6310, 0.6241, 0.6840, 0.1978],
            [0.3641, 0.8806, 0.6636, 0.7896, 0.4980, 0.4865, 0.6539, 0.4849, 0.6162, 0.0276]], grad_fn=<ClampBackward1>)
    
    
    These are our l2 outputs:
    tensor([[ 0.1788,  0.3784,  0.1961,  0.0859,  0.2355,  0.1763,  0.2745,  0.2243,  0.3367,  0.2174],
            [ 0.3220,  0.4116,  0.2579, -0.0955,  0.2618,  0.2804,  0.4132,  0.4018,  0.5158,  0.3829],
            [ 0.4339,  0.6410,  0.3890,  0.0903,  0.4595,  0.3141,  0.4068,  0.4590,  0.6507,  0.3687],
            [ 0.3566,  0.5638,  0.3251,  0.0046,  0.3596,  0.3282,  0.4666,  0.3938,  0.6289,  0.4177],
            [ 0.2960,  0.4863,  0.3065, -0.0497,  0.1770,  0.3261,  0.4764,  0.4248,  0.3754,  0.3667],
            [ 0.3670,  0.4384,  0.2241,  0.0423,  0.3642,  0.2121,  0.3428,  0.5588,  0.5101,  0.2422],
            [ 0.2763,  0.3741,  0.1834,  0.0262,  0.2649,  0.2036,  0.3613,  0.4662,  0.4422,  0.2451],
            ...,
            [ 0.3498,  0.4886,  0.2095,  0.0291,  0.3451,  0.2555,  0.4172,  0.3680,  0.6598,  0.4025],
            [ 0.3693,  0.4834,  0.2048,  0.0112,  0.3305,  0.2773,  0.4418,  0.3806,  0.6904,  0.4129],
            [ 0.5209,  0.6908,  0.2949,  0.0601,  0.3974,  0.3924,  0.4738,  0.5311,  0.7260,  0.3913],
            [ 0.3983,  0.4815,  0.3269, -0.0498,  0.3313,  0.3176,  0.5540,  0.4679,  0.6727,  0.4875],
            [ 0.2548,  0.4201,  0.1639,  0.0116,  0.2426,  0.2281,  0.4484,  0.4445,  0.4960,  0.3393],
            [ 0.4534,  0.5900,  0.2193, -0.0076,  0.3166,  0.3599,  0.4884,  0.4361,  0.7055,  0.4458],
            [ 0.3559,  0.5106,  0.2420,  0.0457,  0.3540,  0.2792,  0.4954,  0.4229,  0.6938,  0.4276]], grad_fn=<AddBackward0>)
    
    
    These are our r2 outputs:
    tensor([[0.1788, 0.3784, 0.1961, 0.0859, 0.2355, 0.1763, 0.2745, 0.2243, 0.3367, 0.2174],
            [0.3220, 0.4116, 0.2579, 0.0000, 0.2618, 0.2804, 0.4132, 0.4018, 0.5158, 0.3829],
            [0.4339, 0.6410, 0.3890, 0.0903, 0.4595, 0.3141, 0.4068, 0.4590, 0.6507, 0.3687],
            [0.3566, 0.5638, 0.3251, 0.0046, 0.3596, 0.3282, 0.4666, 0.3938, 0.6289, 0.4177],
            [0.2960, 0.4863, 0.3065, 0.0000, 0.1770, 0.3261, 0.4764, 0.4248, 0.3754, 0.3667],
            [0.3670, 0.4384, 0.2241, 0.0423, 0.3642, 0.2121, 0.3428, 0.5588, 0.5101, 0.2422],
            [0.2763, 0.3741, 0.1834, 0.0262, 0.2649, 0.2036, 0.3613, 0.4662, 0.4422, 0.2451],
            ...,
            [0.3498, 0.4886, 0.2095, 0.0291, 0.3451, 0.2555, 0.4172, 0.3680, 0.6598, 0.4025],
            [0.3693, 0.4834, 0.2048, 0.0112, 0.3305, 0.2773, 0.4418, 0.3806, 0.6904, 0.4129],
            [0.5209, 0.6908, 0.2949, 0.0601, 0.3974, 0.3924, 0.4738, 0.5311, 0.7260, 0.3913],
            [0.3983, 0.4815, 0.3269, 0.0000, 0.3313, 0.3176, 0.5540, 0.4679, 0.6727, 0.4875],
            [0.2548, 0.4201, 0.1639, 0.0116, 0.2426, 0.2281, 0.4484, 0.4445, 0.4960, 0.3393],
            [0.4534, 0.5900, 0.2193, 0.0000, 0.3166, 0.3599, 0.4884, 0.4361, 0.7055, 0.4458],
            [0.3559, 0.5106, 0.2420, 0.0457, 0.3540, 0.2792, 0.4954, 0.4229, 0.6938, 0.4276]], grad_fn=<ClampBackward1>)
    
    
    These are our l3 outputs:
    tensor([[1.7382, 1.0127],
            [2.3116, 1.5027],
            [3.1483, 1.8254],
            [2.8914, 1.7114],
            [2.1701, 1.6438],
            [2.2242, 1.4845],
            [1.9086, 1.3069],
            ...,
            [2.7114, 1.5372],
            [2.7652, 1.5826],
            [3.3515, 2.0682],
            [2.8961, 1.8331],
            [2.1227, 1.4342],
            [3.0565, 1.8808],
            [2.8763, 1.6804]], grad_fn=<AddBackward0>)
    
    
    These are our softmax outputs:
    tensor([[0.6738, 0.3262],
            [0.6919, 0.3081],
            [0.7897, 0.2103],
            [0.7649, 0.2351],
            [0.6286, 0.3714],
            [0.6769, 0.3231],
            [0.6460, 0.3540],
            ...,
            [0.7639, 0.2361],
            [0.7654, 0.2346],
            [0.7830, 0.2170],
            [0.7433, 0.2567],
            [0.6656, 0.3344],
            [0.7642, 0.2358],
            [0.7678, 0.2322]], grad_fn=<DivBackward0>)



```python
y_preds = softmax.output

```


```python
#hide
wandbs
```




    (tensor([[    -0.0002,      0.2331,      0.0117,      0.2600,      0.2774,      0.1307,      0.1950,      0.2057,      0.1399,     -0.0986],
             [     0.1946,     -0.0915,      0.0643,      0.0087,      0.2372,      0.2070,      0.2365,      0.2561,      0.2278,     -0.0841],
             [     0.0705,     -0.1123,     -0.0432,      0.2374,      0.1913,      0.1934,      0.0814,     -0.0839,     -0.1167,      0.2246],
             [     0.0705,      0.1724,      0.1248,      0.0990,      0.1931,     -0.0708,     -0.0022,      0.2181,      0.0299,      0.1734],
             [     0.1153,     -0.1066,     -0.0893,      0.1900,      0.2777,      0.1359,      0.1473,      0.1143,      0.2712,     -0.0556],
             [     0.1509,      0.1151,      0.0681,     -0.0028,      0.2353,      0.1948,      0.0870,      0.0006,     -0.0163,      0.0389],
             [    -0.0553,      0.2512,      0.0427,      0.2267,     -0.0093,      0.1136,     -0.1013,      0.1188,     -0.0013,      0.1935],
             [     0.1943,      0.2582,      0.1152,      0.2282,      0.1577,      0.0751,      0.0589,      0.0211,     -0.0506,      0.1875],
             [     0.0339,      0.1409,      0.0418,      0.1518,      0.0863,      0.0752,     -0.0287,      0.0064,     -0.0014,      0.0103],
             [    -0.1056,      0.2283,      0.1243,      0.1812,      0.1591,      0.1607,      0.1488,     -0.0419,     -0.0072,     -0.0633],
             [    -0.1127,     -0.0293,      0.1272,      0.0205,      0.0115,      0.2260,     -0.0251,      0.2198,     -0.0274,      0.0513],
             [     0.0951,      0.0732,      0.2631,      0.1341,     -0.0821,      0.0232,      0.1464,      0.0473,      0.1390,     -0.0458],
             [     0.1315,      0.0439,      0.1412,     -0.0664,      0.0940,      0.1332,     -0.0286,      0.0506,      0.0248,      0.0755],
             [     0.2530,      0.2420,      0.2765,      0.2048,     -0.0882,     -0.0035,      0.1577,      0.0605,      0.0726,      0.2733],
             [     0.2079,      0.0583,      0.0246,      0.1822,     -0.0031,      0.1719,     -0.0781,      0.2127,      0.2108,      0.0760],
             [    -0.0597,     -0.0759,      0.1393,      0.2165,      0.2028,      0.0627,     -0.0170,     -0.0651,      0.0579,      0.0773],
             [     0.1473,     -0.0495,      0.2317,      0.0157,     -0.1144,     -0.0419,      0.1788,      0.2388,      0.1428,      0.0085],
             [     0.0061,      0.2621,      0.2392,      0.0263,     -0.1017,     -0.0018,      0.1258,      0.1266,      0.1345,     -0.0650]],
            requires_grad=True),
     tensor([[-0.0621, -0.0691,  0.2171, -0.0623,  0.1145,  0.0525,  0.2408,  0.2170,  0.1916,  0.0629],
             [-0.0341, -0.0009, -0.0881,  0.1173,  0.0511, -0.0289,  0.1675,  0.2559,  0.0796, -0.0451],
             [ 0.1161,  0.0154, -0.0154, -0.0897,  0.1087, -0.0746, -0.0648, -0.0229,  0.1991,  0.2154],
             [ 0.0914,  0.0735, -0.0821, -0.0773,  0.0863,  0.0284, -0.0972,  0.1577,  0.1829, -0.0549],
             [ 0.0412,  0.1230,  0.2663,  0.0677,  0.2552, -0.0593,  0.0626,  0.1118, -0.0184,  0.1215],
             [-0.0723,  0.1951,  0.0814,  0.1907, -0.0819,  0.0893,  0.2131, -0.0612,  0.0357,  0.0940],
             [ 0.2410,  0.0710, -0.0211,  0.1043,  0.0940,  0.0236, -0.1108, -0.0611,  0.2035, -0.0306],
             [ 0.1431,  0.2022,  0.1448, -0.1078, -0.0829,  0.1917,  0.2769,  0.1859,  0.0531,  0.1962],
             [ 0.0767,  0.2175,  0.1521, -0.0540,  0.0504,  0.2353,  0.2760, -0.1082,  0.1134,  0.2455],
             [ 0.2477,  0.2034,  0.1353,  0.0091, -0.0367,  0.2536,  0.0904,  0.1440, -0.1072, -0.0786]], requires_grad=True),
     tensor([[ 0.6042,  0.9942],
             [ 1.1100,  0.6981],
             [ 0.2269, -0.0373],
             [ 0.8578,  0.3621],
             [ 1.1303,  0.1530],
             [ 1.2218,  0.6881],
             [ 0.4251,  0.7249],
             [-0.3872,  0.6244],
             [ 1.3207, -0.3660],
             [ 0.7619,  0.8244]], requires_grad=True),
     tensor([[ 0.0031,  0.0179, -0.0161, -0.0428, -0.0240, -0.0101, -0.0082,  0.0291,  0.0329, -0.0085]], requires_grad=True),
     tensor([[-0.0072,  0.0279, -0.0105, -0.0237, -0.0117,  0.0383,  0.0107, -0.0042,  0.0435,  0.0144]], requires_grad=True),
     tensor([[-0.0298, -0.0055]], requires_grad=True))



## Defining our loss function:

Negative log loss is going to be our loss function. We use our predictions from softmax to calculate the loss. I won't bore youwith all the explanations. That is done in the resources I mentioned above, in a better way than I ever can.


```python
class negative_log_loss:
    def calculate(self, y_preds, y_true):
        samples = len(y_preds)
        y_pred_clipped = torch.clip(y_preds, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[torch.tensor(range(samples)), y_true]
           
        elif len(y_true.shape) == 2:
            correct_confidences = torch.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -torch.log(correct_confidences)
        return torch.mean(negative_log_likelihoods)
```

## 'Training' our neural network

First we are gonna write a function to update our weights and biases according to our loss(i.e. by reducing the product of the gradient and learning rate by our w&bs.)


```python
def update_wandbs(wandbs, lr):
    for layer in wandbs:
        layer.sub_(layer.grad * lr)
        #print(layer.grad)
        layer.grad.zero_()
```

Then we write a training loop for our network, to train for one 'epoch'.


```python
def one_epoch(wandbs, lr, inputs):
    layer1.weights, layer2.weights, layer3.weights, layer1.biases, layer2.biases, layer3.biases = wandbs
    #print('These are our weights and biases:')
    #print(wandbs)
    layer1.forward(inputs)
    #print("\n")
    #print('These are our l1 outputs:')
    #print(layer1.output)
    relu1.forward(layer1.output)
    #print("\n")
    #print('These are our r1 outputs:')
    #print(relu1.output)
    layer2.forward(relu1.output)
    #print("\n")
    #print('These are our l2 outputs:')
    #print(layer2.output)
    relu2.forward(layer2.output)
    #print("\n")
    #print('These are our r2 outputs:')
    #print(relu2.output)
    layer3.forward(relu2.output)
    #print("\n")
    #print('These are our l3 outputs:')
    #print(layer3.output)
    softmax = Softmax_act()
    softmax.forward(layer3.output)
    y_preds = softmax.output
    #print("\n")
    #print('These are our softmax outputs:')
    #print(softmax.output)
    nll = negative_log_loss()
    loss = nll.calculate(y_preds, y_true)
    loss.backward()
    with torch.no_grad(): update_wandbs(wandbs, 2)
    print(f"{loss:.3f}", end="; ")
```

Then, we define another function so that we can train the model easily for multiple epochs. Learning rate is an important hyperparameter here. Refer to Jeremy's notebooks/videos if you don't already know about it.


```python
def train_model(wandbs, lr, inputs, epochs=30):
    #torch.manual_seed(442)
    for i in range(epochs): one_epoch(wandbs, lr, inputs)
    return y_preds
```


```python
y_preds = train_model(wandbs, 0.5, inputs, epochs=40)
```

    0.801; 1.816; 0.827; 0.694; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 0.693; 


```python
y_preds
```




    tensor([[0.6738, 0.3262],
            [0.6919, 0.3081],
            [0.7897, 0.2103],
            [0.7649, 0.2351],
            [0.6286, 0.3714],
            [0.6769, 0.3231],
            [0.6460, 0.3540],
            ...,
            [0.7639, 0.2361],
            [0.7654, 0.2346],
            [0.7830, 0.2170],
            [0.7433, 0.2567],
            [0.6656, 0.3344],
            [0.7642, 0.2358],
            [0.7678, 0.2322]], grad_fn=<DivBackward0>)




```python
def accuracy(y_preds, y_true):
    samples = len(y_preds)
    return print(f'Accuracy is {(y_true.bool()==(y_preds[torch.tensor(range(samples)), y_true]>0.5)).float().mean()*100 :.3f} percent')
```


```python
accuracy(y_preds, y_true)
```

    Accuracy is 0.000 percent


Well, My network is still pretty crappy. I tried a lot, but couldn't get it to train yet. I'm gonna keep trying. But for now, I'm going to use [a framework](https://www.kaggle.com/code/jhoward/why-you-should-use-a-framework) to make my life easier. Afterall, purpose of this whole exercise was not to get an accurate model, but to understand the nuts and bolts of a neural network!


```python

```
