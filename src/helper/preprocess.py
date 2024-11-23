# IMPORTS
import numpy as np
import pandas as pd
import polars as pl
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper import float_to_time, time_to_float, float_time_range, float_time_minus
import random
import tqdm
import xgboost as xgb
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# Set seed for repeatability
def seed_everything(seed):
    np.random.seed(seed) # np random seed
    random.seed(seed) # py random seed
seed_everything(seed=1024)
import torch
print('torch version: ',torch.__version__)
print('Cuda available: ',torch.cuda.is_available())
print('Running on ',torch.cuda.get_device_name(torch.cuda.current_device()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CONFIG
load_train_test = False
generate_lstm = False
load_lstm = False
imputate_hr = False
load_lstm_hr = False
clean_bg = False
load_lstm_hr_clean = True
normalise = True



# SETUP SCHEMA AND LOAD DATA 
train_participants = ['p_num_p01','p_num_p02','p_num_p03','p_num_p04','p_num_p05','p_num_p06','p_num_p10','p_num_p11','p_num_p12']
test_participants = [ 'p_num_p01', 'p_num_p02', 'p_num_p04', 'p_num_p05', 'p_num_p06', 'p_num_p10', 'p_num_p11', 'p_num_p12', 'p_num_p15', 'p_num_p16', 'p_num_p18', 'p_num_p19', 'p_num_p21', 'p_num_p22', 'p_num_p24']
all_participants = sorted(set(train_participants + test_participants))

train_schema = {'id': pl.String(),'p_num': pl.String(),'time': pl.Time(),'bg+1:00': pl.Float64()}

for measurement_time in float_time_range(5.55,0.00,-0.05):
    train_schema[f'bg-{measurement_time}'] = pl.Float64()
    train_schema[f'insulin-{measurement_time}'] = pl.Float64()
    train_schema[f'carbs-{measurement_time}'] = pl.Float64()
    train_schema[f'hr-{measurement_time}'] = pl.Float64()
    train_schema[f'steps-{measurement_time}'] = pl.Float64()
    train_schema[f'cals-{measurement_time}'] = pl.Float64()
    train_schema[f'activity-{measurement_time}'] = pl.String()
    
test_schema = train_schema
del test_schema['bg+1:00']


def load_data(path, schema):
    df = pl.read_csv(path,
                schema_overrides=pl.Schema(schema),
                null_values = ['',' ','null','NaN','None']
    )
    
    step_columns = [f'steps-{t}' for t in float_time_range(5.55, 0.00, -0.05)]

    df = df.with_columns(pl.col(step_columns).cast(pl.UInt32))
    df = df.to_dummies(columns=['p_num'])
    
    df = df.with_columns(pl.lit(0).alias(col) for col in all_participants if col not in df.columns)
        

    for measurement in ['insulin', 'carbs', 'steps', 'cals']:
        for time in float_time_range(5.55, 0.00, -0.05):
            df = df.with_columns(
                [pl.col(f'{measurement}-{time}').fill_null(0)]
            )
    return df


if load_train_test:
    train_df = load_data('data/train.csv', train_schema)
    test_df = load_data('data/test.csv', test_schema)




# # # ESTABLISH USEFUL LISTS OF COLUMNS

lstm__train_features = [
    'id', 'p_num_p01', 'p_num_p02', 'p_num_p03', 'p_num_p04', 'p_num_p05', 'p_num_p06',
    'p_num_p10', 'p_num_p11', 'p_num_p12', 'p_num_p15', 'p_num_p16', 'p_num_p18', 'p_num_p19',
    'p_num_p21', 'p_num_p22', 'p_num_p24','bg', 'insulin', 'carbs', 'hr', 'steps', 
    'cals', 'Run', 'Strength training', 'Swim', 'Bike', 'Dancing', 'Stairclimber', 
    'Spinning', 'Walking', 'HIIT', 'Outdoor Bike', 'Walk', 'Aerobic Workout', 'Tennis', 
    'Workout', 'Hike', 'Zumba', 'Sport', 'Yoga', 'Swimming', 'Weights', 'Running'
]
lstm_target = ['bg+1:00']

lstm_train_schema = {
    'id': pl.String,
    'time_delta': pl.Float64,
    'time': pl.Float64,  
    'p_num_p01': pl.Boolean, 
    'p_num_p02': pl.Boolean,
    'p_num_p03': pl.Boolean,
    'p_num_p04': pl.Boolean,
    'p_num_p05': pl.Boolean,
    'p_num_p06': pl.Boolean,
    'p_num_p10': pl.Boolean,
    'p_num_p11': pl.Boolean,
    'p_num_p12': pl.Boolean,
    'p_num_p15': pl.Boolean,
    'p_num_p16': pl.Boolean,
    'p_num_p18': pl.Boolean,
    'p_num_p19': pl.Boolean,
    'p_num_p21': pl.Boolean,
    'p_num_p22': pl.Boolean,
    'p_num_p24': pl.Boolean,
    'bg': pl.Float64,  
    'insulin': pl.Float64,
    'carbs': pl.Float64,
    'hr': pl.Float64,  
    'steps': pl.Float64, 
    'cals': pl.Float64,
    'Run': pl.Boolean,
    'Strength training': pl.Boolean,
    'Swim': pl.Boolean,
    'Bike': pl.Boolean,
    'Dancing': pl.Boolean,
    'Stairclimber': pl.Boolean,
    'Spinning': pl.Boolean,
    'Walking': pl.Boolean,
    'HIIT': pl.Boolean,
    'Outdoor Bike': pl.Boolean,
    'Walk': pl.Boolean,
    'Aerobic Workout': pl.Boolean,
    'Tennis': pl.Boolean,
    'Workout': pl.Boolean,
    'Hike': pl.Boolean,
    'Zumba': pl.Boolean,
    'Sport': pl.Boolean,
    'Yoga': pl.Boolean,
    'Swimming': pl.Boolean,
    'Weights': pl.Boolean,
    'Running': pl.Boolean,
    'bg+1:00': pl.Float64
}
lstm_test_schema = lstm_train_schema.copy()
del lstm_test_schema['bg+1:00']





# # # MELT TO LSTM FORMAT i.e. 1 row per time point 

time_range = float_time_range(5.55, 0.00, -0.05)
float_minute_percent = 10/6
def melt_row(melter_row, is_train=True):
    rows = []
    for m_time in time_range:
        m_time_float = time_to_float(m_time)
        m_time_hours = int(m_time_float)
        m_time_minutes = (m_time_float - m_time_hours) * float_minute_percent
        current_activity = melter_row[f'activity-{m_time}'][0]
        row = {
            'id'               : melter_row['id'][0],
            'time_delta'       : 1.0 if m_time == '0:00' else 5/60,
            'time'             : (( melter_row['time'][0].hour + melter_row['time'][0].minute / 60 ) - (m_time_hours + m_time_minutes)) % 24, 
            'p_num_p01'        : melter_row['p_num_p01'][0], 
            'p_num_p02'        : melter_row['p_num_p02'][0],
            'p_num_p03'        : melter_row['p_num_p03'][0],
            'p_num_p04'        : melter_row['p_num_p04'][0], 
            'p_num_p05'        : melter_row['p_num_p05'][0],
            'p_num_p06'        : melter_row['p_num_p06'][0],
            'p_num_p10'        : melter_row['p_num_p10'][0],
            'p_num_p11'        : melter_row['p_num_p11'][0],
            'p_num_p12'        : melter_row['p_num_p12'][0],
            'p_num_p15'        : melter_row['p_num_p15'][0],
            'p_num_p16'        : melter_row['p_num_p16'][0],
            'p_num_p18'        : melter_row['p_num_p18'][0],
            'p_num_p19'        : melter_row['p_num_p19'][0],
            'p_num_p21'        : melter_row['p_num_p21'][0],
            'p_num_p22'        : melter_row['p_num_p22'][0],
            'p_num_p24'        : melter_row['p_num_p24'][0],
            'bg'               : melter_row[f'bg-{m_time}'][0], 
            'insulin'          : melter_row[f'insulin-{m_time}'][0], 
            'carbs'            : melter_row[f'carbs-{m_time}'][0],
            'hr'               : melter_row[f'hr-{m_time}'][0],
            'steps'            : melter_row[f'steps-{m_time}'][0], 
            'cals'             : melter_row[f'cals-{m_time}'][0],
            'Run'              : 1 if current_activity == 'Run' else 0,
            'Strength training': 1 if current_activity == 'Strength training' else 0,
            'Swim'             : 1 if current_activity == 'Swim' else 0,
            'Bike'             : 1 if current_activity == 'Bike' else 0, 
            'Dancing'          : 1 if current_activity == 'Dancing' else 0,
            'Stairclimber'     : 1 if current_activity == 'Stairclimber' else 0,
            'Spinning'         : 1 if current_activity == 'Spinning' else 0,
            'Walking'          : 1 if current_activity == 'Walking' else 0,
            'HIIT'             : 1 if current_activity == 'HIIT' else 0,
            'Outdoor Bike'     : 1 if current_activity == 'Outdoor Bike' else 0,
            'Walk'             : 1 if current_activity == 'Walk' else 0,
            'Aerobic Workout'  : 1 if current_activity == 'Aerobic Workout' else 0,
            'Tennis'           : 1 if current_activity == 'Tennis' else 0,
            'Workout'          : 1 if current_activity == 'Workout' else 0,
            'Hike'             : 1 if current_activity == 'Hike' else 0,
            'Zumba'            : 1 if current_activity == 'Zumba' else 0,
            'Sport'            : 1 if current_activity == 'Sport' else 0,
            'Yoga'             : 1 if current_activity == 'Yoga' else 0,
            'Swimming'         : 1 if current_activity == 'Swimming' else 0,
            'Weights'          : 1 if current_activity == 'Weights' else 0,
            'Running'          : 1 if current_activity == 'Running' else 0
        }
        if is_train:
            row['bg+1:00'] = melter_row[f'bg+1:00'][0]
        rows.append(row)
    return rows



def melt_to_lstm_format(df, path_to_csv, is_train=True):
    is_first_write = True
    all_rows = []
    for melter_row in tqdm.tqdm(range(df.shape[0]), desc="Processing rows"):
        all_rows.extend(melt_row(df[int(melter_row)], is_train))

        if melter_row % 100 == 0 or melter_row == df.shape[0] - 1:
            lstm_df = pl.DataFrame(all_rows, schema=lstm_train_schema if is_train else lstm_test_schema)
            all_rows = []

            if is_first_write:
                lstm_df.write_csv(path_to_csv, include_header=True)
                is_first_write = False
            else:
                with open(path_to_csv, 'a') as f:
                    lstm_df.write_csv(f, include_header=False)
                    
if generate_lstm:  
    melt_to_lstm_format(train_df, 'data/lstm_train.csv', is_train=True)
    melt_to_lstm_format(test_df, 'data/lstm_test.csv', is_train=False)           






# # Use XGBoost to predict missing hr values
if imputate_hr:
    lstm_train_df = pl.read_csv('data/lstm_train.csv', schema_overrides=lstm_train_schema)
    lstm_test_df = pl.read_csv('data/lstm_test.csv', schema_overrides=lstm_test_schema)


    best_params = { # obtained from hyperparameter tuning on google colab
        'subsample': 1.0,
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.2,
        'colsample_bytree': 0.8
    }

    final_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        device='cuda',
        **best_params
    )

    hr_pred_feature_cols = [
        'time', 'Run', 'Strength training', 'Swim', 'Bike', 'Dancing',
        'Stairclimber', 'Spinning', 'Walking', 'HIIT', 'Outdoor Bike',
        'Walk', 'Aerobic Workout', 'Tennis', 'Workout', 'Hike',
        'Zumba', 'Sport', 'Yoga', 'Swimming', 'Weights', 'Running'
    ]  + all_participants

    clean_train_hr_df = lstm_train_df.filter(pl.col('hr').is_not_null()).select(hr_pred_feature_cols + ['hr'])
    clean_test_hr_df = lstm_test_df.filter(pl.col('hr').is_not_null()).select(hr_pred_feature_cols + ['hr'])

    clean_hr_df = pl.concat([clean_train_hr_df, clean_test_hr_df])

    X = clean_hr_df.select(hr_pred_feature_cols)
    y = clean_hr_df.select('hr')
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()

    final_model.fit(X_np, y_np)

    def imputate_hr_values(df, model, output_path):

        missing_hr_df = df.filter(pl.col('hr').is_null())

        X_missing = missing_hr_df.select(hr_pred_feature_cols)
        X_missing_np = X_missing.to_numpy()

        predicted_hr = final_model.predict(X_missing_np)
        predicted_hr_lst = [hr for hr in predicted_hr]

        pd_df = df.to_pandas()

        predicted_hr_lst_copy = pd.Series(predicted_hr_lst)

        missing_mask = pd_df['hr'].isnull() 
        missing_count = missing_mask.sum()
        print(f"Number of missing 'hr' values: {missing_count}")

        if missing_count > 0:
            pd_df.loc[missing_mask, 'hr'] = predicted_hr_lst_copy[:missing_count].values

        pd_df.to_csv(output_path, index=False)
    
    imputate_hr_values(lstm_train_df, final_model, 'data/lstm_hr_train.csv')
    imputate_hr_values(lstm_test_df, final_model, 'data/lstm_hr_test.csv')

if load_lstm_hr:
    lstm_hr_train_df = pl.read_csv('data/lstm_hr_train.csv', schema_overrides=lstm_train_schema)
    lstm_hr_test_df = pl.read_csv('data/lstm_hr_test.csv', schema_overrides=lstm_test_schema)
    





# CLEAN BG VALUES

noisy_participants = ['p_num_p01', 'p_num_p05', 'p_num_p06', 'p_num_p21']
clean_participants = [p for p in all_participants if p not in noisy_participants]

def clean_bg_values(train_df: pl.DataFrame, test_df: pl.DataFrame,
                   clean_participants: list, noisy_participants: list):
    clean_train = train_df.clone()
    clean_test = test_df.clone()
    
    for p in clean_participants:
        p_trim = p.replace('p_num_', '')
        

        train_mask = clean_train['id'].str.contains(p_trim)
        id_list = (clean_train
                  .filter(train_mask)
                  .filter(pl.col('bg').null_count().over('id') > 10)
                  .select('id'))
        
        clean_train = (clean_train
                      .filter(~(pl.col('id').is_in(id_list['id']) & train_mask))
                      .with_columns(pl.col('bg').fill_null(strategy="forward")))
        
    for p in noisy_participants:
        p_trim = p.replace('p_num_', '')
        
        train_mask = clean_train['id'].str.contains(p_trim)
        clean_train = (clean_train
                      .filter(train_mask)
                      .with_columns(pl.col('bg').fill_null(strategy="forward"))
                      .vstack(clean_train.filter(~train_mask)))
        
        
    clean_test = (clean_test.with_columns(pl.col('bg').fill_null(strategy="forward")))
    
    # Just to get rid of any edge cases
    clean_train = clean_train.with_columns(pl.col('bg').fill_null(strategy="backward"))
    clean_test = clean_test.with_columns(pl.col('bg').fill_null(strategy="backward"))
    
    return clean_train, clean_test

if clean_bg:
    cleaned_train, cleaned_test = clean_bg_values(lstm_hr_train_df, lstm_hr_test_df, clean_participants, noisy_participants)

    cleaned_train.write_csv('data/lstm_hr_train_clean.csv', include_header=True)
    cleaned_test.write_csv('data/lstm_hr_test_clean.csv', include_header=True)


if load_lstm_hr_clean:
    lstm_hr_train_clean_df = pl.read_csv('data/lstm_hr_train_clean.csv', schema_overrides=lstm_train_schema)
    lstm_hr_test_clean_df = pl.read_csv('data/lstm_hr_test_clean.csv', schema_overrides=lstm_test_schema)




# NORMALISE DATA AND SAVE AS TENSORS
noisy_participants = ['p01', 'p05', 'p06', 'p21']
participants_in_both = ['p01', 'p02', 'p04', 'p05', 'p06', 'p10', 'p11', 'p12']

participants_to_remove = ['p_num_p03', 'p_num_p15', 'p_num_p16', 'p_num_p18', 'p_num_p19','p_num_p21', 'p_num_p22', 'p_num_p24'] # The idea is that these are unknown participants so no point encoding them

if normalise:
    columns_to_normalise = ["time", "insulin", "carbs", "hr", "steps", "cals"]

    train_data_to_normalise = lstm_hr_train_clean_df.select(columns_to_normalise).to_numpy()
    
    scaler = MinMaxScaler()
    train_normalised_data = scaler.fit_transform(train_data_to_normalise)

    lstm_hr_train_clean_df = lstm_hr_train_clean_df.with_columns(
        [pl.Series(columns_to_normalise[i], train_normalised_data[:, i]) for i in range(len(columns_to_normalise))]
    )

    test_data_to_normalise = lstm_hr_test_clean_df.select(columns_to_normalise).to_numpy()
    test_normalised_data = scaler.transform(test_data_to_normalise)

    lstm_hr_test_clean_df = lstm_hr_test_clean_df.with_columns(
        [pl.Series(columns_to_normalise[i], test_normalised_data[:, i]) for i in range(len(columns_to_normalise))]
    )

    train_bg_data = lstm_hr_train_clean_df.select('bg').to_numpy()
    train_bg_h_data = lstm_hr_train_clean_df.select('bg+1:00').to_numpy()
    test_bg_data = lstm_hr_test_clean_df.select(['bg']).to_numpy()

    bg_scaler = MinMaxScaler()
    bg_scaler.fit(train_bg_data)

    bg_train_normalised = bg_scaler.transform(train_bg_data)
    bg_h_train_normalised = bg_scaler.transform(train_bg_h_data)
    bg_test_normalised = bg_scaler.transform(test_bg_data)
    

    lstm_hr_train_clean_df = lstm_hr_train_clean_df.with_columns(
        [
            pl.Series('bg', bg_train_normalised.flatten()),
            pl.Series('bg+1:00', bg_h_train_normalised.flatten())
        ]
    )

    lstm_hr_test_clean_df = lstm_hr_test_clean_df.with_columns(
        pl.Series('bg', bg_test_normalised.flatten()) 
    )
    
    lstm_hr_train_clean_df = lstm_hr_train_clean_df.with_columns(
        (pl.col("id").str.split('_').list[0].is_in(noisy_participants)).cast(pl.Int8).alias("Noisy Group")
    )   
    lstm_hr_train_clean_df = lstm_hr_train_clean_df.drop([col for col in participants_to_remove if col in lstm_hr_train_clean_df.columns])
    
    lstm_hr_test_clean_df = lstm_hr_test_clean_df.with_columns(
    (pl.col("id").str.split('_').list[0].is_in(noisy_participants)).cast(pl.Int8).alias("Noisy Group")
    )
    lstm_hr_test_clean_df = lstm_hr_test_clean_df.drop([col for col in participants_to_remove if col in lstm_hr_test_clean_df.columns])
    print(lstm_hr_test_clean_df)
    print(lstm_hr_train_clean_df)
    
    lstm_hr_test_clean_df = lstm_hr_test_clean_df.with_columns(
        (
        (
        pl.col("id")
        .str.split("_")
        .list[0]
        .str.split('p')
        .list[1]
        .cast(pl.Int32)
        )
        +
        (
        (pl.col("id")
        .str.split("_")
        .list[1])
        .cast(pl.Float64) / 1e5
        )
        )
        
        .alias("id_float")) 
    print(lstm_hr_test_clean_df)
    
    lstm_train_by_id = lstm_hr_train_clean_df.group_by('id').agg(pl.all())
    lstm_test_by_id = lstm_hr_test_clean_df.group_by('id').agg(pl.all())
    
    lstm_test_df = lstm_test_by_id.drop(['id'])
    lstm_train_df = lstm_train_by_id.drop(['id'])        
        
        
    lstm_train_np = np.stack([
        np.array(lstm_train_df[col].to_list(), 
                dtype=bool if lstm_train_df[col].dtype == pl.Boolean else np.float32) 
        for col in lstm_train_df.columns
    ])
    lstm_train_tensor = torch.from_numpy(lstm_train_np)
    
    lstm_test_np = np.stack([
        np.array(lstm_test_df[col].to_list(), 
                dtype=bool if lstm_test_df[col].dtype == pl.Boolean else np.float32) 
        for col in lstm_test_df.columns
    ])
    lstm_test_tensor = torch.from_numpy(lstm_test_np)
    
    
    
    
    t_lstm_train_tensor = lstm_train_tensor[-2,:,0]
    t_lstm_train_tensor = t_lstm_train_tensor.reshape(-1,1)
    
    X_lstm_train_tensor = torch.cat((lstm_train_tensor[:-2, :, :], lstm_train_tensor[-1:, :, :]), dim=0)
    X_lstm_train_tensor = X_lstm_train_tensor.permute(1,2,0)
    
    print(X_lstm_train_tensor.shape, t_lstm_train_tensor.shape)
    
    X_lstm_test_tensor = lstm_test_tensor.permute(1,2,0)
    
    print(X_lstm_test_tensor.shape)
    
    
    # in form tensor_data[sequence_point, batch_number, feature_index]
    torch.save(X_lstm_train_tensor,'data/X_lstm_train.pt')
    torch.save(t_lstm_train_tensor,'data/t_lstm_train.pt')
    torch.save(X_lstm_test_tensor,'data/X_lstm_test.pt')