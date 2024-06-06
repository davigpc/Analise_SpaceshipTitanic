import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from ydata_profiling import ProfileReport
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
SEED = 42
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Obtendo os dados

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.head())
print(train_df.columns.tolist())

# Junção dos dataframes para facilitar o tratamento dos dados

train_df_len = len(train_df) # Verificação do tamanho para futura separação
test_df_len = len(test_df)
print(train_df_len, test_df_len)

def combine_df(train_df, test_df):
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    return combined_df

def separate_df(combined_df, train_df_len, test_df_len):
    train_df = combined_df.iloc[:train_df_len]
    test_df = combined_df.iloc[train_df_len:train_df_len + test_df_len]
    return train_df, test_df

full_df = combine_df(train_df, test_df)
print(len(full_df))

# Tratamento de colunas vazias
# Passageiros congelados nao irao gastar dinheiro e nem adquirir passe VIP durante a viagem,
# logo, pode-se encontrar valores NaN nessas colunas nesse caso.

# Colunas a serem alteradas:
zero_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
false_columns = ['VIP']

# Condition se o passageiro está em CryoSleep
condition = (full_df['CryoSleep'] == True)

# Substituição nas colunas
full_df.loc[~condition, zero_columns] = full_df.loc[~condition, zero_columns].fillna(0)
full_df.loc[~condition, false_columns] = full_df.loc[~condition, false_columns].fillna(False)

# O contrario também acontece: se o passageiro possui valor NaN na coluna CryoSleep, mas está gastando dinheiro,
# provavelmente não está congelado.

cryosleep_column = ['CryoSleep']

# Verifica se todas as colunas de dinheiro são iguais a zero.
condition = (full_df[zero_columns].eq(0).all(axis=1))

# Preenche valores NaN como True se a condição for verdadeira
full_df.loc[~condition, cryosleep_column] = full_df.loc[~condition, cryosleep_column].fillna(True)

# Agora, todos os valores NaN restantes devem ser de pessoas que não estão congeladas
full_df['CryoSleep'].fillna(False, inplace=True)

print(full_df.isna().sum())
print(full_df['Cabin'].head())

# As cabines estão em um formato que os modelos de machine learning não conseguem processar,
# portanto, deve-se separar em novas colunas: deck, number e side.

cabin_split = full_df['Cabin'].str.split(pat='/', expand=True)
cabin_split.columns = ['Deck', 'Room_Number', 'Side']
new_df = pd.concat([full_df, cabin_split], axis=1)
full_df = new_df
full_df.drop(columns=['Cabin'], inplace=True)

# Agora é necessário tratar os valores NaN em outras colunas, como a idade referente a cada planeta natal,
# checando se está ou não em CryoSleep

group_means = full_df.groupby(['HomePlanet', 'CryoSleep'])['Age'].mean()
print(group_means)

# Preenchendo os valores para a idade

group_means = full_df.groupby(['HomePlanet', 'CryoSleep'], group_keys=True)['Age'].mean().reset_index(name='Age_mean')
full_df = full_df.merge(group_means, on=['HomePlanet', 'CryoSleep'], how='left')
full_df['Age'] = full_df['Age'].fillna(full_df['Age_mean'])
full_df.drop(columns=['Age_mean'], inplace=True)

print(full_df.isna().sum())

# Para continuar o processamento dos dados, será utilizado o recurso Pipeline, para gerenciar os tratamentos de forma
# organizada e padronizada

# Separando em variaveis categoricas e numericas

categorical_features = ['HomePlanet','CryoSleep','Destination','VIP','Deck', 'Side']
numerical_features = ['Room_Number', 'Age', 'RoomService', 'FoodCourt','ShoppingMall','Spa','VRDeck']

# Criando os pipelines de pre processamento para variaveis categoricas e numericas

categorical_pipeline = Pipeline([
    ('inputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combinando os pipelines utilizando o ColumnTransformer
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# Separando novamente os dataframes em treino e teste

train_df, test_df = separate_df(full_df,train_df_len,test_df_len)

# Selecionando o objetivo

y = train_df['Transported']

# Retirando o objetivo de ambos os dataframes

train_df.drop(columns=['Transported'], inplace=True)
test_df.drop(columns=['Transported'], inplace=True)

# Y não pode ser string, deve ter valores binarios

y = y.astype(bool)
print('\n',y.head())

# Agora, o tratamento de dados será direcionado aos algoritmos de machine learning. Dados numericos serão escalados e categoricos utilizarão
# o one hot encoder

print('X_train shape: {}'.format(train_df.shape))
print('y_train shape: {}'.format(y.shape))
print('X_test shape: {}'.format(test_df.shape))

# Fit e limpeza dos dados, troca de nomes das colunas para correção
transformed_data_train = preprocessor.fit_transform(train_df)
transformed_data_test = preprocessor.transform(test_df)

feature_names = preprocessor.named_transformers_['categorical'] \
    .named_steps['encoder'].get_feature_names_out(input_features=categorical_features)

all_feature_names = numerical_features + list(feature_names)

# Transformando em dataframes
transformed_train_df = pd.DataFrame(transformed_data_train, columns = all_feature_names)
transformed_test_df = pd.DataFrame(transformed_data_test, columns = all_feature_names)

# Verificando se as colunas possuem a mesma quantidade de categorias
print('X_train shape: {}'.format(transformed_train_df.shape))
print('y_train shape: {}'.format(y.shape))
print('X_test shape: {}'.format(transformed_test_df.shape))

# Primeiro GBM Model
gbm_model_1 = GradientBoostingClassifier(n_estimators=70, learning_rate=0.1, max_features='sqrt', max_depth=5, random_state=SEED,
                                        min_samples_split=2, min_samples_leaf=3, loss='exponential', subsample=0.5)

# Segundo GBM Model
gbm_model_2 = GradientBoostingClassifier(n_estimators=70, learning_rate=0.1, max_features='log2', max_depth=5, random_state=SEED,
                                        min_samples_split=2, min_samples_leaf=3, subsample=0.5, loss='log_loss')

# Terceiro GBM Model
gbm_model_3 = GradientBoostingClassifier(n_estimators=70, learning_rate=0.1, max_features='log2', max_depth=5, random_state=SEED,
                                        min_samples_split=2, min_samples_leaf=3, subsample=0.5, loss='exponential')

X_train, X_test, y_train, y_test = train_test_split(transformed_train_df, y, test_size=0.2, random_state=SEED)

# Fit dos modelos
gbm_model_1.fit(X_train, y_train)
gbm_model_2.fit(X_train, y_train)
gbm_model_3.fit(X_train, y_train)

# Executando o treinamento para o Logistic Regression
gbm_1_predictions = gbm_model_1.predict(X_test)
gbm_2_predictions = gbm_model_2.predict(X_test)
gbm_3_predictions = gbm_model_3.predict(X_test)

# Combinando as predições para o Logistic Regression
stacked_features = np.column_stack((gbm_1_predictions, gbm_2_predictions, gbm_3_predictions))

# Fit do modelo
meta_model = LogisticRegression()
meta_model.fit(stacked_features, y_test)

# Prevendo novamente
gbm_1_base_preds = gbm_model_1.predict(X_test)
gbm_2_base_preds = gbm_model_2.predict(X_test)
gbm_3_base_preds = gbm_model_3.predict(X_test)

# Combinando novas previsões
stacked_base_preds = np.column_stack((gbm_1_base_preds, gbm_2_base_preds, gbm_3_base_preds))

# Obtendo a previsão final
ensemble_predictions = meta_model.predict(stacked_base_preds)

# Pontuando a previsão pela acurácia
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print("Ensemble Accuracy:", ensemble_accuracy)

# Obtendo novas bases de previsão
gbm_1_base_preds = gbm_model_1.predict(transformed_test_df)
gbm_2_base_preds = gbm_model_2.predict(transformed_test_df)
gbm_3_base_preds = gbm_model_3.predict(transformed_test_df)

# Combinando
stacked_base_preds = np.column_stack((gbm_1_base_preds, gbm_2_base_preds, gbm_3_base_preds))

# Previsão e resultado final
ensemble_predictions = meta_model.predict(stacked_base_preds)

output = pd.DataFrame({
    'PassengerId' : test_df['PassengerId'],
    'Transported' : ensemble_predictions
})
output.to_csv('submission.csv', index=False)
print('Submission Saved')