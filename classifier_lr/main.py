import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load data
digits = load_digits()

print('Data description:')
print(' Shape of features: ',digits.data.shape)
print(' Shape of labels: ', digits.target.shape)

# Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.8)
print(' Train data:{}'.format(X_train.shape))
print(' Test data:{}\n'.format(X_train.shape))

# Create logistic Regression model
lr_model = LogisticRegression(solver='newton-cg', max_iter=100)

print('Training model...\n')
lr_model.fit(X_train, y_train) # fit train data

# Predict test data
y_pred = lr_model.predict(X_test)

# Summerize results
print('Prediction results:')
cm = confusion_matrix(y_test, y_pred)
print(' Confusion matrix:')
print(cm) # print confusion matix

print(' Accuracy = {}'.format(accuracy_score(y_test, y_pred))) # print accuracy

# Plot confusion matrix
sns.set(font_scale=1)  # for label size
color_map = sns.cubehelix_palette(dark=0, light=0.95, as_cmap=True)  # color_map for seaborn plot
plt.title('LINEAR CLASSIFIER: CONFUSION MATRIX (HEATMAP)', fontsize=10)
sns.heatmap(cm, cmap=color_map, annot=True, annot_kws={"size": 10}, fmt="d")  # plot confusion matrix heatmap
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Save the heatmap plot in local
plt.savefig('docker_classifier_lr.png', bbox_inches='tight')
plt.show()