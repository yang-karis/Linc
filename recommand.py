import numpy as np
import pandas as pd


#read data
data = pd.read_table('data.txt')

#drop duplicate the same product
data = data.drop_duplicates(['Shopper ID','Product ID'])

#dummy the product
data = data.drop('Purchase datetime',1)
data = pd.get_dummies(data, columns=['Shopper ID']).groupby(['Product ID'], as_index=False).sum()


#build Jaccard index matrix according to Shopper
Jaccard_matrix = np.eye(data.shape[1]-1)

#calculate the Jaccard
for i in range(1,data.shape[1]):
    for j in range(1,data.shape[1]):
        if i != j:
            J1 = data.loc[data[data.columns[i]]==1,'Product ID']
            J2 = data.loc[data[data.columns[j]]==1,'Product ID']
            
            J_u = set(J1).union(J2)
            J_i = set(J1).intersection(J2)
            
            Jaccard_matrix[i-1,j-1] = len(J_i)/len(J_u)
            
'''
1.Find two shopper with highest Jaccard index
'''
J_max = np.ma.masked_equal(Jaccard_matrix, 1.0, copy=False).max()
position = np.where(Jaccard_matrix == J_max)[0]
print('Find the two shoppers with the highest Jaccard index:',data.columns[position[0]+1],data.columns[position[1]+1])


'''
2.Find top 3 products we should recommend to shopper “andrew”
'''
#drop the product Andrew buy already
product_other = data[data['Shopper ID_andrew'] != 1]
#build recommand array
r_product = np.zeros(len(product_other))
#Find the position of andrew in matrix 
position = np.where(data.columns == 'Shopper ID_andrew')[0]-1
related = Jaccard_matrix[0]

#fill up recommand index
for i in range(len(product_other)):
    rel = product_other.iloc[i,(product_other.columns != 'Product ID')].values
    r_product[i] = np.sum((rel*related))/rel.sum()

#choose top k
top_3 = np.argsort(r_product)[-3:]
print('Find the top 3 products we should recommend to shopper “andrew”:',product_other.iloc[top_3]['Product ID'].values)

















        