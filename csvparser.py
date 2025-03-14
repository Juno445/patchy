import  pandas as pd

df = pd.read_csv('input.csv')
#Breaks our values down to 100 per group
filtered = df.groupby('agent_group').head(100)

result = filtered[['agent_group','subject']]

result.to_csv('output.csv')
