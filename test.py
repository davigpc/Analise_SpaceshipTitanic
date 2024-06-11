import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 42


# Function to combine and separate dataframes
def combine_df(train_df, test_df):
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    return combined_df


def separate_df(combined_df, train_df_len, test_df_len):
    train_df = combined_df.iloc[:train_df_len]
    test_df = combined_df.iloc[train_df_len:train_df_len + test_df_len]
    return train_df, test_df


# Function to preprocess the data
def preprocess_data(train_df, test_df):
    full_df = combine_df(train_df, test_df)

    zero_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    false_columns = ['VIP']

    condition = (full_df['CryoSleep'] == True)
    full_df.loc[~condition, zero_columns] = full_df.loc[~condition, zero_columns].fillna(0)
    full_df.loc[~condition, false_columns] = full_df.loc[~condition, false_columns].fillna(False)

    cryosleep_column = ['CryoSleep']
    condition = (full_df[zero_columns].eq(0).all(axis=1))
    full_df.loc[~condition, cryosleep_column] = full_df.loc[~condition, cryosleep_column].fillna(True)
    full_df['CryoSleep'].fillna(False, inplace=True)

    cabin_split = full_df['Cabin'].str.split(pat='/', expand=True)
    cabin_split.columns = ['Deck', 'Room_Number', 'Side']
    full_df = pd.concat([full_df, cabin_split], axis=1)
    full_df.drop(columns=['Cabin'], inplace=True)

    group_means = full_df.groupby(['HomePlanet', 'CryoSleep'])['Age'].mean()
    group_means = full_df.groupby(['HomePlanet', 'CryoSleep'], group_keys=True)['Age'].mean().reset_index(
        name='Age_mean')
    full_df = full_df.merge(group_means, on=['HomePlanet', 'CryoSleep'], how='left')
    full_df['Age'] = full_df['Age'].fillna(full_df['Age_mean'])
    full_df.drop(columns=['Age_mean'], inplace=True)

    return full_df


# Main function to run the application
def main():
    st.title('Spaceship Titanic Prediction')

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    st.write("### Train Data Preview")
    st.write(train_df.head())

    st.write("### Test Data Preview")
    st.write(test_df.head())

    full_df = preprocess_data(train_df, test_df)
    train_df_len = len(train_df)
    test_df_len = len(test_df)

    train_df, test_df = separate_df(full_df, train_df_len, test_df_len)

    y = train_df['Transported']
    train_df.drop(columns=['Transported'], inplace=True)
    test_df.drop(columns=['Transported'], inplace=True)

    y = y.astype(bool)

    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    numerical_features = ['Room_Number', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    categorical_pipeline = Pipeline([
        ('inputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])

    transformed_data_train = preprocessor.fit_transform(train_df)
    transformed_data_test = preprocessor.transform(test_df)

    feature_names = preprocessor.named_transformers_['categorical'] \
        .named_steps['encoder'].get_feature_names_out(input_features=categorical_features)

    all_feature_names = numerical_features + list(feature_names)

    transformed_train_df = pd.DataFrame(transformed_data_train, columns=all_feature_names)
    transformed_test_df = pd.DataFrame(transformed_data_test, columns=all_feature_names)

    X_train, X_test, y_train, y_test = train_test_split(transformed_train_df, y, test_size=0.2, random_state=SEED)

    gbm_model_1 = GradientBoostingClassifier(n_estimators=70, learning_rate=0.1, max_features='sqrt', max_depth=5,
                                             random_state=SEED,
                                             min_samples_split=2, min_samples_leaf=3, loss='exponential', subsample=0.5)
    gbm_model_2 = GradientBoostingClassifier(n_estimators=70, learning_rate=0.1, max_features='log2', max_depth=5,
                                             random_state=SEED,
                                             min_samples_split=2, min_samples_leaf=3, subsample=0.5, loss='log_loss')
    gbm_model_3 = GradientBoostingClassifier(n_estimators=70, learning_rate=0.1, max_features='log2', max_depth=5,
                                             random_state=SEED,
                                             min_samples_split=2, min_samples_leaf=3, subsample=0.5, loss='exponential')

    gbm_model_1.fit(X_train, y_train)
    gbm_model_2.fit(X_train, y_train)
    gbm_model_3.fit(X_train, y_train)

    gbm_1_predictions = gbm_model_1.predict(X_test)
    gbm_2_predictions = gbm_model_2.predict(X_test)
    gbm_3_predictions = gbm_model_3.predict(X_test)

    stacked_features = np.column_stack((gbm_1_predictions, gbm_2_predictions, gbm_3_predictions))

    meta_model = LogisticRegression()
    meta_model.fit(stacked_features, y_test)

    gbm_1_base_preds = gbm_model_1.predict(X_test)
    gbm_2_base_preds = gbm_model_2.predict(X_test)
    gbm_3_base_preds = gbm_model_3.predict(X_test)

    stacked_base_preds = np.column_stack((gbm_1_base_preds, gbm_2_base_preds, gbm_3_base_preds))

    ensemble_predictions = meta_model.predict(stacked_base_preds)

    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    st.write(f"Ensemble Accuracy: {ensemble_accuracy}")

    gbm_1_base_preds = gbm_model_1.predict(transformed_test_df)
    gbm_2_base_preds = gbm_model_2.predict(transformed_test_df)
    gbm_3_base_preds = gbm_model_3.predict(transformed_test_df)

    stacked_base_preds = np.column_stack((gbm_1_base_preds, gbm_2_base_preds, gbm_3_base_preds))

    final_predictions = meta_model.predict(stacked_base_preds)

    output = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_predictions
    })

    st.write("### Final Predictions")
    st.write(output.head())

    if st.button('Save Submission'):
        output.to_csv('submission.csv', index=False)
        st.write('Submission Saved!')


if __name__ == "__main__":
    main()
