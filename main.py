import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load airbnb data
df = pd.read_csv('/content/sample_data/AB_NYC_2019.csv')

# find features that have correlation with price to be able to create a prediction
corr_matrix = df.corr()
# set median house value as target in box, then sort values according to it
corr_matrix["price"].sort_values(ascending=False)    # ascending=False means descending

# possible useful augmentations?
df["price_for_all_availabilities"] = df["price"]*df["availability_365"]
df["minimum_price_per_nights"] = df["price"]/df["minimum_nights"]

# prepare for regression
# drop target
df_unlabeled = df.drop(['price'], axis = 1)
# remove the categorical features
df_num_only = df_unlabeled.drop("neighbourhood_group", axis=1)
df_num_only = df_num_only.drop("neighbourhood", axis=1)
df_num_only = df_num_only.drop("room_type", axis=1)
# create pipeline
num_pipeline_1 = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
df_num_only_tr = num_pipeline_1.fit_transform(df_num_only)
numerical_features = list(df_num_only)
categorical_features = ["neighbourhood_group", "neighbourhood", "room_type"]
full_pipeline = ColumnTransformer([
        ("num", num_pipeline_1, numerical_features),
        ("cat", OneHotEncoder(), categorical_features), # cat means categorical
    ])

# result oipeline
df_prepared = full_pipeline.fit_transform(df_unlabeled)

# split into training and test data
from sklearn.model_selection import train_test_split
data_target = df['price']
train, test, target, target_test = train_test_split(df_prepared, data_target, test_size=0.8, random_state=0)

# fit model
lin_reg = LinearRegression()
lin_reg.fit(train, target)
data = test
labels = target_test

print("Predictions:", lin_reg.predict(data)[:5])
print("Actual labels:", list(labels)[:5])

# rmse error evaluation
preds = lin_reg.predict(test)
mse = mean_squared_error(target_test, preds)
rmse = np.sqrt(mse)
rmse



import tensorflow as tf
from tensorflow import keras

# change all train, target, etc to numpy or an array so TF can use
# check: I think these might be sparse as they are csr_matrix type, make dense?
# use .to_numpy() or .toarray()
train = train.toarray()
test = test.toarray()
target = target.toarray()
target_test = target_test.toarray()


# use same train and test split with linear regression
# try using relu to see if there is non-linear relations in the data
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape = [train.shape[1]]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train, target, epochs=100, batch_size=32)
test_loss = model.evaluate(test, target_test)
rsme = np.sqrt(test_loss)
rsme