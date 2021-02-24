import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings("ignore")

dataset = [['Milk', 'Snacks', 'Bread', 'Jam', 'Eggs', 'Banana', 'Beer', 'Diaper'],
           ['Milk','Bread','Beer', 'Diaper', 'Snacks','Banana'],
           ['Milk','Bread','Jam','Eggs','Banana'],
           ['Milk', 'Bread','Jam' 'Yogurt', 'Eggs','Banana'],
           ['Bread', 'Eggs', 'Diaper', 'Beer', 'Snacks'],
           ['Beer','Diaper','Snacks' 'Eggs','Jam']]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df.head()

frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

frequent_itemsets

association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules

rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules

rules[ (rules['antecedent_len'] >= 0.5) &
       (rules['confidence'] > 0.75) &
       (rules['lift'] > 1.2) ]

rules[rules['antecedents'] == {'Milk', 'Bread'}]
