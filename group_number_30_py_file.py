
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Load the datasets
productdf = pd.read_csv("olist_products_dataset.csv")
reviewdf = pd.read_csv("olist_order_reviews_dataset.csv")
ordersdf = pd.read_csv("olist_orders_dataset.csv")
customersdf = pd.read_csv("olist_customers_dataset.csv")
itemsdf = pd.read_csv("olist_order_items_dataset.csv")
sellerdf = pd.read_csv("olist_sellers_dataset.csv")

# Step 1: Merge ordersdf with customersdf to get customer information
orders_customers = pd.merge(ordersdf, customersdf[['customer_id', 'customer_unique_id', 'customer_city', 'customer_state']],
                            on='customer_id', how='left')

# Step 2: Merge the result with reviewdf to get review_score
orders_customers_reviews = pd.merge(orders_customers, reviewdf[['order_id', 'review_score']],
                                    on='order_id', how='left')

# Step 3: Merge the result with itemsdf to get product_id, seller_id, and price
orders_items = pd.merge(orders_customers_reviews, itemsdf[['order_id', 'product_id', 'seller_id', 'price']],
                        on='order_id', how='left')

# Step 4: Merge the result with productdf to get product_category_name and product_photos_qty
final_df = pd.merge(orders_items, productdf[['product_id', 'product_category_name', 'product_photos_qty']],
                    on='product_id', how='left')

# Step 5: Merge the result with sellerdf to get seller city and state
final_df = pd.merge(final_df, sellerdf[['seller_id', 'seller_city', 'seller_state']],
                    on='seller_id', how='left')

# Step 6: Select the relevant columns
final_df = final_df[['customer_unique_id', 'customer_city', 'customer_state', 'product_photos_qty',
                     'order_id', 'order_approved_at', 'order_delivered_carrier_date', 'order_estimated_delivery_date',
                     'review_score', 'seller_id', 'price', 'product_category_name', 'product_id']]

# Check the final DataFrame
print(final_df.head())

joined_df=final_df

#Convert the columns as a date time format

joined_df['order_approved_at'] = pd.to_datetime(joined_df['order_approved_at'], errors='coerce')
joined_df['order_delivered_carrier_date'] = pd.to_datetime(joined_df['order_delivered_carrier_date'], errors='coerce')
joined_df['order_estimated_delivery_date'] = pd.to_datetime(joined_df['order_estimated_delivery_date'], errors='coerce')
joined_df['days_pay_delivery'] = joined_df['order_delivered_carrier_date'] - joined_df['order_approved_at']

# Subtract order_estimate+
joined_df['days_expec_del_vs_actual_del'] = joined_df['order_estimated_delivery_date'] - joined_df['order_delivered_carrier_date']
joined_df['days_pay_expect'] = joined_df['order_approved_at']-joined_df['order_estimated_delivery_date']
#dropping observations where expected delivery date is before date of approval
joined_df=joined_df[joined_df.days_pay_expect<pd.Timedelta(0, unit='s')]
joined_df=joined_df[joined_df.days_pay_delivery>pd.Timedelta(0, unit='s')]
# Convert the timedelta to days as floats:
joined_df['days_pay_delivery'] = joined_df['days_pay_delivery'].dt.total_seconds() / 86400
joined_df['days_expec_del_vs_actual_del'] = joined_df['days_expec_del_vs_actual_del'].dt.total_seconds() / 86400
joined_df['days_pay_expect']=joined_df['days_pay_expect'].dt.total_seconds()/86400
#removing observations with outliers in the different time variables
joined_df=joined_df[joined_df['days_pay_expect']>-40]
joined_df=joined_df[joined_df['days_pay_delivery']<50]
joined_df=joined_df[joined_df['days_expec_del_vs_actual_del']>-80]


#remove na values and variables that are no longer needed
joined_df = joined_df.drop(['order_approved_at', 'order_delivered_carrier_date', 'order_estimated_delivery_date','order_id'], axis=1)
joined_df = joined_df.dropna(subset=['review_score', 'product_photos_qty', 'seller_id', 'price',
                                     'product_category_name', 'product_id',
                                     'days_pay_delivery', 'days_expec_del_vs_actual_del','days_pay_expect'])

joined_df = joined_df[~joined_df.duplicated(keep='first')] # keep the first instance only
#remove all instances where the unique customer id or customer state is a single observation
joined_df = joined_df[joined_df.duplicated(subset=['customer_state'], keep=False)]
joined_df = joined_df[joined_df.duplicated(subset=['customer_unique_id'], keep=False)]
#create the binary variable for positive review scores, where 1 is if review score is 4 or higher and 0 otherwise
joined_df['Positive'] = np.where(joined_df.review_score >= 4, 1, 0)
#check to see the number of observations in each class to ensure there is no huge imbalance
joined_df['Positive'].value_counts()
#not symmetrical, but relatively similar. to ensure there is no effect from class imbalance later on


from sklearn.preprocessing import RobustScaler

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(joined_df, test_size=0.2, random_state=4567)
# print the shapes to check everything is OK
print(X_train.shape)
print(X_test.shape)

#finding mean of each customer and state in the training set
train_customer_mean = X_train.groupby('customer_unique_id')['review_score'].mean()
train_state_mean = X_train.groupby('customer_state')['review_score'].mean()
#mapping the mean to the relevant customers and states in the training set
X_train['mean by customer']=X_train['customer_unique_id'].map(train_customer_mean)
X_train['mean by state']=X_train['customer_state'].map(train_state_mean)
#mapping that same prediction to the relevant customers and states in the test set
X_test['mean by customer']=X_test['customer_unique_id'].map(train_customer_mean)
X_test['mean by state']=X_test['customer_state'].map(train_state_mean)

#removing any observations with NA in case there was certain observations that lacked a mean in customer or state
X_test = X_test.dropna(subset=['review_score', 'product_photos_qty', 'seller_id', 'price',
                                     'product_category_name', 'product_id',
                                     'days_pay_delivery', 'days_expec_del_vs_actual_del','days_pay_expect',
                                     'Positive',"mean by customer", "mean by state"])
X_train = X_train.dropna(subset=['review_score', 'product_photos_qty', 'seller_id', 'price',
                                     'product_category_name', 'product_id',
                                     'days_pay_delivery', 'days_expec_del_vs_actual_del','days_pay_expect'
                                     ,'Positive'])
#remove variables that are no longer needed
X_train = X_train.drop(['customer_unique_id','seller_id',
                     'product_id', 'customer_city', 'customer_state', 'product_category_name'],axis=1)
X_test = X_test.drop(['customer_unique_id','seller_id',
                     'product_id', 'customer_city', 'customer_state', 'product_category_name'],axis=1)

#creating Ytrain and test datasets
Y_train = X_train['Positive']  # Set the target variable (Positive)
Y_train = np.ravel(Y_train)
Y_test= X_test['Positive']
Y_test=np.ravel(Y_test)


# Initialize the RobustScaler
scaler = RobustScaler()
cols = X_train.columns.difference(['Positive'])
# Apply RobustScaler to the numerical columns except for positive
X_train.loc[:,cols] = scaler.fit_transform(X_train[cols])
cols = X_test.columns.difference(['Positive'])
# Apply RobustScaler to the numerical columns except for positive
X_test.loc[:,cols] = scaler.fit_transform(X_test[cols])

#remove the target variable Positive and also review score to reduce data leakage
X_train=X_train.drop(['review_score','Positive'],axis=1)
X_test=X_test.drop(['review_score','Positive'],axis=1)


# Import necessary models (classification versions)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support

# Initialize the classification models
RF_algo = RandomForestClassifier()
GBDT_algo = GradientBoostingClassifier()


# Train the models
RF_model = RF_algo.fit(X_train, Y_train)
GBDT_model = GBDT_algo.fit(X_train, Y_train)


models = [RF_model, GBDT_model]
names = ['Unoptimised Random Forest', 'Unoptimised GBDT']

# Evaluate the models using classification metrics
for i in range(2):
    print(f"Model: {names[i]}")

    # Predict on test data (or training data, depending on your choice)
    predict = models[i].predict(X_test)  # Use X_test to evaluate on unseen data

    # Calculate precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict, average='macro')

    # Print the metrics
    print(f"Macro Precision: {precision}")
    print(f"Macro Recall: {recall}")
    print(f"Macro F1-score: {f1_score}")
    print("\n")

    # Calculate precision, recall, and F1-score for each class
    precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict, average=None)

    # Print the metrics
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1_score}")
    print("\n")


# Evaluate the models on training data using classification metrics
names1 = ['Unoptimised Random Forest on Training Data', 'Unoptimised GBDT on Training Data']
for i in range(2):
    print(f"Model: {names1[i]}")

    # Predict on test data (or training data, depending on your choice)
    predict = models[i].predict(X_train)  # Use X_test to evaluate on unseen data

    # Calculate precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict, average='macro')

    # Print the metrics
    print(f"Macro Precision: {precision}")
    print(f"Macro Recall: {recall}")
    print(f"Macro F1-score: {f1_score}")
    print("\n")
#small decrease when moving from training to test set, which is expected.


import warnings
from sklearn.metrics import precision_recall_fscore_support
# Import regression metrics
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
# Suppress warnings
warnings.filterwarnings("ignore")
# Create a hyperparameter search function for reusability
def random_search(algo, hyperparameters, X_train, Y_train):
    # Perform the search using 5 folds (cross-validation)
    clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                             scoring= 'precision_macro', n_iter=20, refit=True)

    # Fit/train the model
    clf.fit(X_train, Y_train)

    return clf.best_params_

# Define hyperparameters for RandomForest
RF_algo = RandomForestClassifier()
# Random Forest
RF_tuned_parameters = {
    'n_estimators': randint(1, 100),  # Number of trees
    'max_depth': randint(2, 40),  # Max depth of the trees
    'min_samples_split': randint(10, 30),  # Min samples for a split
    'max_features': ['sqrt', 'log2', None]  # Features to consider for each split
}


RF_best_params = random_search(RF_algo, RF_tuned_parameters, X_train, Y_train)
print("Random Forest Best Params:", RF_best_params)
RF_model = RandomForestClassifier(**RF_best_params).fit(X_train, Y_train)
models = [RF_model]
names = ['Optimised Random Forest']

# Evaluate the models using precision, recall, and F1 score
for i in range(1):
    print(f"Model: {names[i]}")

    # Predict based on test data
    predict = models[i].predict(X_test)  # Ensure you're using X_test for evaluation

    # Calculate precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict, average='macro')

    # Print the metrics
    print(f"Macro Precision: {precision}")
    print(f"Macro Recall: {recall}")
    print(f"Macro F1-score: {f1_score}")
    print("\n")

from sklearn.metrics import classification_report
#Getting the classification report to see the precision for each class
print("Classification Report:\n")
print(classification_report(Y_test, RF_model.predict(X_test)))

#Finding the most important feature in our model
importances = RF_model.feature_importances_
features = X_test.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay as CM
# Checking to see if class imbalance affects the Random Forest Model significantly
print("Optimised Random Forest Confusion Matrix")
predict = RF_model.predict(X_test)
CM.from_predictions(Y_test, predict)
plt.show()

#Checking the metrics when using it on the training set
RF_model = RandomForestClassifier(**RF_best_params).fit(X_train, Y_train)
models = [RF_model]
names = ['Optimised Random Forest on Training Dataset']

# Evaluate the models using precision, recall, and F1 score
for i in range(1):
    print(f"Model: {names[i]}")

    # Predict based on test data
    predict = models[i].predict(X_train)  # Ensure you're using X_test for evaluation

    # Calculate precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict, average='macro')

    # Print the metrics
    print(f"Macro Precision: {precision}")
    print(f"Macro Recall: {recall}")
    print(f"Macro F1-score: {f1_score}")
    print("\n")



# Gradient Boosting Decision Trees hyperparameter optimisation(GBDT)
GBDT_tuned_parameters = {
    'n_estimators': randint(20, 100),  # Number of boosting stages (trees)
    'learning_rate': uniform(loc=0.1, scale=0.1),  # Learning rate
    'criterion': ['friedman_mse', 'squared_error'],  # Split quality function
    'max_depth': randint(2, 50)  # Maximum depth of trees
}
GBDT_algo = GradientBoostingClassifier()
GBDT_best_params = random_search(GBDT_algo, GBDT_tuned_parameters, X_train, Y_train)
print("GBDT Best Params:", GBDT_best_params)
GBDT_model = GradientBoostingClassifier(**GBDT_best_params).fit(X_train, Y_train)
models = [GBDT_model]
name = ['Optimised GBDT']

# Evaluate the models using precision, recall, and F1 score
for i in range(1):
    print(f"Model: {name[i]}")

    # Predict based on test data
    predict = models[i].predict(X_test)  # Ensure you're using X_test for evaluation

    # Calculate precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict, average='macro')

    # Print the metrics
    print(f"Macro Precision: {precision}")
    print(f"Macro Recall: {recall}")
    print(f"Macro F1-score: {f1_score}")
    print("\n")

#Getting the classification report to see the precision for each class
print("Classification Report:\n")
print(classification_report(Y_test, GBDT_model.predict(X_test)))

#Finding the most important feature in our model
importances = GBDT_model.feature_importances_
features = X_test.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay as CM

#checking to see if class imbalance is affecting the precision the model
# GBDT confusion matrix
print("GBDT Confusion Matrix")
predict = GBDT_model.predict(X_test)
CM.from_predictions(Y_test, predict)
plt.show()



#Checking the metrics when running the model on training data
GBDT_model = GradientBoostingClassifier(**GBDT_best_params).fit(X_train, Y_train)
models = [GBDT_model]
names = ['Optimised GBDT on Training Data']

# Evaluate the models using precision, recall, and F1 score
for i in range(1):
    print(f"Model: {names[i]}")

    # Predict based on test data
    predict = models[i].predict(X_train)  # Ensure you're using X_test for evaluation

    # Calculate precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict, average='macro')

    # Print the metrics
    print(f"Macro Precision: {precision}")
    print(f"Macro Recall: {recall}")
    print(f"Macro F1-score: {f1_score}")
    print("\n")
