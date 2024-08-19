import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Perlu untuk mengaktifkan IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")
combined_df = pd.concat([train_df,test_df],ignore_index=True)

columns = train_df.columns.to_list()

# Group 
combined_df['group'] = combined_df['PassengerId'].apply(lambda x : x[:4])

# Bill Amount Statistic
combined_df['TotalSpendingMean'] = combined_df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].mean(axis=1)
combined_df['TotalSpending'] = combined_df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
combined_df['TotalSpendingMedian'] = combined_df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].median(axis=1)

# Add 'Family Name' column
combined_df['Family Name'] = combined_df['Name'].apply(lambda x: x.split()[1] if pd.notna(x) else np.nan)
# Menggabungkan kolom 'Family Name' dan 'group' dengan tanda pemisah '-'
combined_df['Related Family'] = combined_df['Family Name'] + '-' + combined_df['group']

# Ekstrak fitur dari 'Cabin'
combined_df[['CabinDeck', 'CabinNumber', 'CabinSide']] = combined_df['Cabin'].str.split('/', expand=True)
combined_df['CabinNumber'] = pd.to_numeric(combined_df['CabinNumber'], errors='coerce')  # Konversi ke numerik

# Kategorikan 'Age'
combined_df['AgeGroup'] = pd.cut(combined_df['Age'], bins=[0, 17, 65, 100], labels=['Child', 'Adult', 'Senior'])

# Mengisi nilai NaN pada kolom VIP dengan False jika Age < 18 atau Age > 75 dan yang berasal dari Bumi
combined_df.loc[(combined_df['VIP'].isna()) & ((combined_df['Age'] < 18) | (combined_df['Age'] > 75)), 'VIP'] = False
combined_df.loc[(combined_df['VIP'].isna()) & (combined_df['HomePlanet'] == 'Earth'), 'VIP'] = False


# Mengisi nilai 'Age' yang hilang pada 'TotalSpending' == 0 dengan 6
combined_df.loc[(combined_df['Age'].isna()) & (combined_df['TotalSpending'] == 0), 'Age'] = 6

# Menyiapkan kolom yang relevan untuk imputasi MICE
cols_for_imputation = ['Age']  # Tambahkan kolom lain yang relevan jika diperlukan

# Membuat salinan DataFrame untuk imputasi
df_for_imputation = combined_df[cols_for_imputation]

# Menggunakan IterativeImputer (MICE) untuk mengisi nilai yang hilang
mice_imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data = mice_imputer.fit_transform(df_for_imputation)

# Menggantikan nilai yang diimputasi pada DataFrame asli
combined_df['Age'] = imputed_data[:, 0]

# Kategorikan 'Age' setelah imputasi
combined_df['AgeGroup'] = pd.cut(combined_df['Age'], bins=[-1, 17, 65, 100], labels=['Child', 'Adult', 'Senior'])

# Fill Cycrosleep jadi True jika Total Spending adalah 0
combined_df['CryoSleep'] = combined_df['CryoSleep'].fillna(combined_df['TotalSpending'].apply(lambda x: x == 0))

# Kolom yang ingin diisi
columns_to_fill = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Mengisi NaN di kolom tertentu dengan 0 jika CryoSleep bernilai True
combined_df.loc[combined_df['CryoSleep'] == True, columns_to_fill] = combined_df.loc[combined_df['CryoSleep'] == True, columns_to_fill].fillna(0)

# Mengisi NaN di kolom tertentu dengan median jika CryoSleep bernilai False
for column in columns_to_fill:
    median_value = combined_df[column].median()
    combined_df.loc[combined_df['CryoSleep'] == False, column] = combined_df.loc[combined_df['CryoSleep'] == False, column].fillna(median_value)
    
# Mengelompokkan berdasarkan 'group' dan memfilter yang nilai 'group'-nya lebih dari satu dan urutkan
group_analysis = combined_df.groupby('group').filter(lambda x: len(x) > 1).sort_values(by=['group','PassengerId','Name'])

# Fill NaN Value (diisi berdasarkan nama keluarga dari group dia berada) hasil pengurutan di atas (keluarga biasanya idnya urut)
group_analysis['Name'].bfill(inplace=True)

# Add 'Family Name' column
group_analysis['Family Name'] = group_analysis['Name'].apply(lambda x: x.split()[1] if pd.notna(x) else np.nan)
# Menggabungkan kolom 'Family Name' dan 'group' dengan tanda pemisah '-'
group_analysis['Related Family'] = group_analysis['Family Name'] + '-' + group_analysis['group']

# Identifikasi dan hapus duplikat dalam group_analysis
group_analysis.drop_duplicates(subset='PassengerId', inplace=True)

# Create Rest NaN Name Value from All Data
group_analysis_id = group_analysis['PassengerId'].values.tolist()
nan_name_df = combined_df.loc[(combined_df['Name'].isna()) & (~combined_df['PassengerId'].isin(group_analysis_id))]

# Join nan data to group analysis data
group_analysis = pd.concat([group_analysis, nan_name_df]).drop_duplicates(subset='PassengerId')

# Fill NaN Name Data with Mode of Family Name
group_analysis['Name'].fillna(group_analysis['Family Name'].mode()[0], inplace=True)

# Add 'Family Name' column with the updated condition
group_analysis['Family Name'] = group_analysis['Name'].apply(
    lambda x: x.split()[1] if pd.notna(x) and len(x.split()) > 1 else (x.split()[0] if pd.notna(x) else np.nan)
)

# Menggabungkan kolom 'Family Name' dan 'group' dengan tanda pemisah '-'
group_analysis['Related Family'] = group_analysis['Family Name'] + '-' + group_analysis['group']

# Update data lama dengan data baru
combined_df.update(group_analysis)

# Mengisi nilai NaN pada Cabin berdasarkan Family Name karena terkadang keluarga yang sama berada pada cabin yang sama 
group_analysis['Cabin'] = group_analysis.groupby('Related Family')['Cabin'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

# Mengisi nilai 'HomePlanet' berdasarkan 'CabinDeck'
combined_df.loc[combined_df['CabinDeck'].isin(['C', 'B', 'A']) & combined_df['HomePlanet'].isna(), 'HomePlanet'] = 'Europa'

combined_df.loc[(combined_df['Destination'].isna()) & (combined_df['HomePlanet'] == 'Mars'), 'Destination'] = 'TRAPPIST-1e'

# Mengisi nilai NaN pada kolom kategorikal menggunakan modus
categorical_cols = ['Destination', 'HomePlanet', 'VIP']

for col in categorical_cols:
    mode = combined_df[col].mode()[0]
    combined_df[col].fillna(mode, inplace=True)
    
combined_df['Cabin'].ffill(inplace=True)

# Ekstrak fitur dari 'Cabin'
combined_df[['CabinDeck', 'CabinNumber', 'CabinSide']] = combined_df['Cabin'].str.split('/', expand=True)
combined_df['CabinNumber'] = pd.to_numeric(combined_df['CabinNumber'], errors='coerce')  # Konversi ke numerik

# Mengganti nilai True dan False ke 1 dan 0
combined_df[['VIP','CryoSleep']] = combined_df[['VIP','CryoSleep']].astype(int)

combined_df['CabinNumber'] = pd.to_numeric(combined_df['CabinNumber'], errors='coerce')  # Konversi ke numerik
combined_df['CabinDeck'] = combined_df['CabinDeck'].astype(str)  # Pastikan CabinDeck adalah string
combined_df['CabinSide'] = combined_df['CabinSide'].astype(str)  # Pastikan CabinSide adalah string

combined_df_copy = combined_df.copy()

# Membuat instance LabelEncoder
le = LabelEncoder()

# Label encoding untuk kolom 'CabinDeck'
combined_df_copy['CabinDeck Encoded'] = le.fit_transform(combined_df_copy['CabinDeck'])
# Label encoding untuk kolom 'Family Name'
combined_df_copy['Related Family Encoded'] = le.fit_transform(combined_df_copy['Related Family'])
combined_df_copy['AgeGroup Encoded'] = le.fit_transform(combined_df_copy['AgeGroup'])

# Label encoding untuk kolom 'CabinDeck'
combined_df_copy['CabinDeck Encoded'] = le.fit_transform(combined_df_copy['CabinDeck'])

# Label encoding untuk kolom 'CabinSide'
combined_df_copy['CabinSide Encoded'] = le.fit_transform(combined_df_copy['CabinSide'])

# Standarisasi kolom numerik
scaler = MinMaxScaler()
numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpendingMean','TotalSpending','TotalSpendingMedian']
combined_df_copy[numerical_columns] = scaler.fit_transform(combined_df_copy[numerical_columns])

# Label encoding untuk kolom 'CabinSide'
combined_df_copy = pd.get_dummies(combined_df_copy, columns=['CabinSide'])

# Hitung frekuensi label di kolom HomePlanet
homeplanet_counts = combined_df_copy['HomePlanet'].value_counts()
homeplanet_mapping = {label: idx for idx, label in enumerate(homeplanet_counts.index)}

# Hitung frekuensi label di kolom Destination
destination_counts = combined_df_copy['Destination'].value_counts()
destination_mapping = {label: idx for idx, label in enumerate(destination_counts.index)}

# Encode kolom berdasarkan mapping
combined_df_copy['HomePlanetEncoded'] = combined_df_copy['HomePlanet'].map(homeplanet_mapping)
combined_df_copy['DestinationEncoded'] = combined_df_copy['Destination'].map(destination_mapping)

# One-hot encoding untuk kolom 'HomePlanet' dan 'Destination'
combined_df_copy = pd.get_dummies(combined_df_copy, columns=['HomePlanet', 'Destination'])

train_df_id = train_df['PassengerId'].values
ready_to_train = combined_df_copy[combined_df_copy['Transported'].notnull()].copy()

# Mengganti nilai True dan False ke 1 dan 0
ready_to_train['Transported'] = ready_to_train['Transported'].astype(int)

# Memilih fitur-fitur yang memiliki korelasi tinggi dan moderat dengan 'Transported'
features_to_use = ['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall',
       'Spa', 'VRDeck', 'TotalSpending', 'CabinNumber',
       'CabinDeck Encoded', 'CabinSide Encoded']

features_set = set(features_to_use)


# Drop fitur-fitur yang tidak digunakan dan memastikan tipe data float
ready_to_train_df = ready_to_train[features_to_use + ['Transported']]

ready_to_train_df.dropna(inplace=True)

# Pisahkan fitur dan target
X = ready_to_train_df.drop('Transported', axis=1)  # Drop kolom target
y = ready_to_train_df['Transported']               # Kolom target

# Identifikasi kolom kategori (jika ada, meskipun data ini tampaknya sudah di-encode)
categorical_features = []  # Daftar kolom kategori jika ada

# Pisahkan data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Inisialisasi CatBoostClassifier
model = CatBoostClassifier(
    cat_features=categorical_features,  # Kolom kategori
    random_state=123           # Set seed untuk reproducibility
)

# Tentukan parameter grid untuk GridSearchCV
param_grid = {
    'iterations': [500, 1000, 1500],
    'learning_rate': [0.01, 0.1, 0.2],
    'depth': [4, 6, 8]
}

# Inisialisasi GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,  # 3-fold cross-validation
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Latih model menggunakan GridSearchCV
grid_search.fit(X_train, y_train)

# Cetak hyperparameter terbaik
print(f'Best Hyperparameters: {grid_search.best_params_}')

# Prediksi pada data uji menggunakan model terbaik
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

data_to_predict = combined_df_copy[combined_df_copy['PassengerId'].isin(test_df['PassengerId'].values)]

ready_to_predict_df = data_to_predict[features_to_use]

# Ubah semua kolom bertipe number dan bool menjadi float
num_bool_cols = ready_to_predict_df.select_dtypes(include=['number', 'bool']).columns
ready_to_predict_df[num_bool_cols] = ready_to_predict_df[num_bool_cols].astype(float)

# Gunakan best_model untuk memprediksi data baru
predictions = best_model.predict(ready_to_predict_df)

# Buat DataFrame untuk submission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],  # Menggunakan PassengerId dari data asli
})

submission['Transported'] = list(predictions)
submission['Transported'].replace(1.0,True,inplace=True)
submission['Transported'].replace(0.0,False,inplace=True)

submission.to_csv('submission_py.csv',index=False)