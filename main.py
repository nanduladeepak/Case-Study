import pandas as pd
from pandas_profiling import ProfileReport
df = pd.read_csv("data_banknote_authentication.txt", sep=",", header=None)
df.rename({0:'Variance',1:'Skewness',2:'Kurtosis',3:'Entropy',4:'output'}, axis=1, inplace=True)
print(df)
# Generate a report
profile = ProfileReport(df)
profile.to_file(output_file="test.pdf")

# x = df[['Variance','Skewness','Kurtosis','Entropy']]
# y = df[['output']]
