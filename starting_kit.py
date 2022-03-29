import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


df = pd.read_csv("public_data.csv")
print(df.info())
print(df.head())


X = df['message']
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


vectorizer = CountVectorizer(min_df=.01, max_df=.8, ngram_range=[1,1], max_features=300, stop_words='english')

pipe = Pipeline([('vec', vectorizer),  ('clf', DecisionTreeClassifier(random_state=223))])

pipe.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix, classification_report

pred_val = pipe.predict(X_val)
print(confusion_matrix(y_val, pred_val))
print(classification_report(y_val, pred_val))


from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

ari = adjusted_rand_score(y_val, pred_val)
ami = adjusted_mutual_info_score(y_val, pred_val, average_method='arithmetic')

print("ARI: {}".format(ari))
print("AMI: {}".format(ami))


## Kaggle Predictions


df_test = pd.read_csv('input_data.csv')
df_test.info()
df_test.head()

pred_test = pipe.predict(df_test['message'])

my_submission = pd.DataFrame({'Id': df_test['id'], 'label': pred_test})
print(my_submission.head())

# NOTE: after saving the CSV file, be sure to zip the file before submitting to the competition website!
my_submission.to_csv('answers.csv', index=False)