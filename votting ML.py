from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


iris = load_iris()
X, y = iris.data, iris.target


data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target


print("First 5 rows of the dataset with target:")
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


decision_tree = DecisionTreeClassifier(random_state=42)
svm = SVC(kernel='linear', random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)


voting_clf = VotingClassifier(
    estimators=[('dt', decision_tree), ('svm', svm), ('rf', random_forest)],
    voting='hard'
)


decision_tree.fit(X_train, y_train)
svm.fit(X_train, y_train)
random_forest.fit(X_train, y_train)


y_pred_dt = decision_tree.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_rf = random_forest.predict(X_test)


voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)




accuracy_voting = accuracy_score(y_test, y_pred_voting)
precision_voting = precision_score(y_test, y_pred_voting, average='weighted')
recall_voting = recall_score(y_test, y_pred_voting, average='weighted')
f1_voting = f1_score(y_test, y_pred_voting, average='weighted')


print("Accuracy - Voting Classifier:", accuracy_voting)
print("Precision - Voting Classifier:", precision_voting)
print("Recall - Voting Classifier:", recall_voting)
print("F1-score - Voting Classifier:", f1_voting)

cm_voting = confusion_matrix(y_test, y_pred_voting)

print("Confusion Matrix - Voting Classifier:")
print(cm_voting)

# Plot confusion matrix for the voting classifier
plt.figure(figsize=(6, 4))
sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Voting Classifier')
plt.show()