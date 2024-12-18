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
load_lstm = True
interpolate_small_gaps = True
imputate_hr = True
load_lstm_hr = True
imputate_bg = True
normalise = True

print('Config: \n')
print(f'load_train_test: {load_train_test}')
print(f'generate_lstm: {generate_lstm}')
print(f'load_lstm: {load_lstm}')
print(f'interpolate_small_gaps: {interpolate_small_gaps}')
print(f'imputate_hr: {imputate_hr}')
print(f'load_lstm_hr: {load_lstm_hr}')
print(f'imputate_bg: {imputate_bg}')
print(f'normalise: {normalise}')

# SETUP SCHEMA AND LOAD DATA 
train_participants = ['p_num_p01','p_num_p02','p_num_p03','p_num_p04','p_num_p05','p_num_p06','p_num_p10','p_num_p11','p_num_p12']
test_participants = [ 'p_num_p01', 'p_num_p02', 'p_num_p04', 'p_num_p05', 'p_num_p06', 'p_num_p10', 'p_num_p11', 'p_num_p12', 'p_num_p15', 'p_num_p16', 'p_num_p18', 'p_num_p19', 'p_num_p21', 'p_num_p22', 'p_num_p24']
all_participants = sorted(set(train_participants + test_participants))

"""
test activities not in train activities-------------------
CoreTraining, Cycling, 
p18 does CoreTraining, 154hr, 8ish bg, pretty constant lets call it workout


train activities not in test activities-------------------
Hike, Stairclimber, Strength training, Tennis, Zumba, 

activities in both-------------------
Aerobic Workout, Bike, Dancing, HIIT, Indoor climbing, Outdoor Bike, Run, Running, Spinning, Sport, Swim, Swimming, Walk, Walking, Weights, Workout, Yoga, 

Activities to group together-------------------
Run, Running
Swim, Swimming
Walk, Walking, Hike
Outdoor Bike, Bike
CoreTraining, Workout
Sport, Tennis
Spinning, Cycling
zumba, aerobic workout
Strength training, Weights
Stairclimber, running


"""
activities = ['Aerobic Workout', 'Dancing', 'HIIT', 'Indoor climbing', 'Outdoor Bike', 'Run', 'Spinning', 'Sport', 'Swim', 'Walk', 'Weights', 'Workout', 'Yoga']

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
    'p_num_p21', 'p_num_p22', 'p_num_p24','bg','bg_null', 'insulin', 'carbs', 'hr','hr_null', 'steps', 
    'cals'] + activities

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
    'bg_null': pl.Boolean,
    'insulin': pl.Float64,
    'carbs': pl.Float64,
    'hr': pl.Float64,  
    'hr_null': pl.Boolean,
    'steps': pl.Float64, 
    'cals': pl.Float64,
}
for activity in activities:
    lstm_train_schema[activity] = pl.Boolean
lstm_train_schema['bg+1:00'] = pl.Float64

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
            'bg_null'          : 1 if melter_row[f'bg-{m_time}'][0] is None else 0,
            'insulin'          : melter_row[f'insulin-{m_time}'][0], 
            'carbs'            : melter_row[f'carbs-{m_time}'][0],
            'hr'               : melter_row[f'hr-{m_time}'][0],
            'hr_null'          : 1 if melter_row[f'hr-{m_time}'][0] is None else 0,
            'steps'            : melter_row[f'steps-{m_time}'][0], 
            'cals'             : melter_row[f'cals-{m_time}'][0],
            'Aerobic Workout'  : 1 if (current_activity == 'Aerobic Workout' or current_activity == 'Zumba') else 0,
            'Dancing'          : 1 if current_activity == 'Dancing' else 0,
            'HIIT'             : 1 if current_activity == 'HIIT' else 0,
            'Indoor climbing'  : 1 if current_activity == 'Indoor climbing' else 0,
            'Outdoor Bike'     : 1 if (current_activity == 'Outdoor Bike' or current_activity == 'Bike') else 0,
            'Run'              : 1 if (current_activity == 'Run' or current_activity == 'Running' or current_activity == 'Stairclimber')  else 0,
            'Spinning'         : 1 if (current_activity == 'Spinning' or current_activity == 'Cycling') else 0,
            'Sport'            : 1 if (current_activity == 'Sport' or current_activity == 'Tennis')else 0,
            'Swim'             : 1 if (current_activity == 'Swim' or current_activity  == 'Swimming') else 0,
            'Walk'             : 1 if (current_activity == 'Walk' or current_activity == 'Walking' or current_activity == 'Hike') else 0,
            'Weights'          : 1 if (current_activity == 'Weights' or current_activity == 'Strength training') else 0,
            'Workout'          : 1 if (current_activity == 'Workout' or current_activity == 'CoreTraining') else 0,
            'Yoga'             : 1 if current_activity == 'Yoga' else 0
            
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
            lstm_df = pl.DataFrame(all_rows, schema_overrides=pl.Schema(lstm_train_schema) if is_train else pl.Schema(lstm_test_schema))
            all_rows = []

            if is_first_write:
                lstm_df.write_csv(path_to_csv, include_header=True)
                is_first_write = False
            else:
                with open(path_to_csv, 'a') as f:
                    lstm_df.write_csv(f, include_header=False)
                    
if generate_lstm:  
    print('Generating LSTM data')
    melt_to_lstm_format(train_df, 'data/lstm_train.csv', is_train=True)
    melt_to_lstm_format(test_df, 'data/lstm_test.csv', is_train=False)           


if load_lstm:
    print('Loading LSTM data')
    lstm_train_df = pl.read_csv('data/lstm_train.csv', schema_overrides=pl.Schema(lstm_train_schema))
    lstm_test_df = pl.read_csv('data/lstm_test.csv', schema_overrides=pl.Schema(lstm_test_schema))

if interpolate_small_gaps:
    print('Interpolating small gaps')
    def interpolate_small_gaps(df):
        interpolate_df = df.select(['hr','bg']).to_pandas()
        interpolate_df = interpolate_df.interpolate(method='linear',
                                                    limit = 3,
                                                    limit_direction='both')

        df = df.with_columns(
            pl.Series('hr', interpolate_df['hr']),
            pl.Series('bg', interpolate_df['bg'])
        )
        return df
    
    lstm_train_df = interpolate_small_gaps(lstm_train_df)
    lstm_test_df = interpolate_small_gaps(lstm_test_df)
    
    lstm_train_df.write_csv('data/lstm_sg_train.csv', include_header=True)
    lstm_test_df.write_csv('data/lstm_sg_test.csv', include_header=True)
    
    
    
    
    
    
    

# # Use XGBoost to predict missing hr values
hr_pred_features = ['time', 'p_num_p01', 'p_num_p02', 'p_num_p03', 'p_num_p04', 'p_num_p05', 'p_num_p06',
                    'p_num_p10', 'p_num_p11', 'p_num_p12', 'p_num_p15', 'p_num_p16', 'p_num_p18',
                    'p_num_p19', 'p_num_p21', 'p_num_p22', 'p_num_p24', 'bg', 'insulin', 'carbs',
                    'steps', 'cals', 'Aerobic Workout', 'Dancing', 'HIIT', 'Indoor climbing', 
                    'Outdoor Bike', 'Run', 'Spinning', 'Sport', 'Swim', 'Walk', 'Weights', 'Workout', 'Yoga']

bg_pred_features = ['time', 'p_num_p01', 'p_num_p02', 'p_num_p03', 'p_num_p04', 'p_num_p05', 'p_num_p06',
                    'p_num_p10', 'p_num_p11', 'p_num_p12', 'p_num_p15', 'p_num_p16', 'p_num_p18', 
                    'p_num_p19', 'p_num_p21', 'p_num_p22', 'p_num_p24', 'insulin', 'carbs', 'hr',
                     'steps', 'cals', 'Aerobic Workout', 'Dancing', 'HIIT', 'Indoor climbing',
                      'Outdoor Bike', 'Run', 'Spinning', 'Sport', 'Swim', 'Walk', 'Weights', 'Workout', 'Yoga']

def imputate_values(df, model,model_features, metric, output_path):

    missing_df = df.filter(pl.col(metric).is_null())

    X_missing = missing_df.select(model_features)
    X_missing_np = X_missing.to_numpy()
    X_missing_np = X_scaler.transform(X_missing_np)
    X_missing = xgb.DMatrix(X_missing_np, missing=np.nan)

    predicted = model.predict(X_missing)
    predicted = predicted.reshape(-1,1)
    predicted = t_scaler.inverse_transform(predicted)
    predicted_lst = [hr for hr in predicted.ravel()]

    pd_df = df.to_pandas()

    predicted_lst_copy = pd.Series(predicted_lst)

    missing_mask = pd_df[metric].isnull() 
    missing_count = missing_mask.sum()
    print(f"Number of missing {metric} values: {missing_count}")

    if missing_count > 0:
        pd_df.loc[missing_mask, metric] = predicted_lst_copy[:missing_count].values

    pd_df.to_csv(output_path, index=False)

if imputate_hr:
    print('Imputating HR values')
    lstm_train_df = pl.read_csv('data/lstm_sg_train.csv', schema_overrides=lstm_train_schema)
    lstm_test_df = pl.read_csv('data/lstm_sg_test.csv', schema_overrides=lstm_test_schema)
    
    all_hr_pred_df = pl.concat([lstm_train_df.drop(['id','time_delta','bg_null','hr_null','bg+1:00']), lstm_test_df.drop(['id','time_delta','bg_null','hr_null'])])
    all_hr_pred_df = all_hr_pred_df.filter(pl.col('hr').is_not_null())

    t = all_hr_pred_df['hr'].to_numpy().reshape(-1,1)
    X = all_hr_pred_df.drop('hr').to_numpy()

    X_scaler = MinMaxScaler()
    t_scaler = MinMaxScaler()
    X = X_scaler.fit_transform(X)
    t = t_scaler.fit_transform(t).ravel()
    
    dtrain_full = xgb.DMatrix(X, label=t, missing=np.nan)
    
    best_boost_round = 467
    
    best_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 10,
        'learning_rate': 0.7057888338320615,
        'min_child_weight': 8,
        'reg_lambda': 0.01696500898391533
    }

    hr_pred_model = xgb.train(
        best_params,
        dtrain_full,
        num_boost_round=best_boost_round,
        verbose_eval=True
    )

    imputate_values(lstm_train_df, hr_pred_model,hr_pred_features,'hr', 'data/lstm_hr_train.csv')
    imputate_values(lstm_test_df, hr_pred_model,hr_pred_features,'hr', 'data/lstm_hr_test.csv')

if load_lstm_hr:
    lstm_hr_train_df = pl.read_csv('data/lstm_hr_train.csv', schema_overrides=lstm_train_schema)
    lstm_hr_test_df = pl.read_csv('data/lstm_hr_test.csv', schema_overrides=lstm_test_schema)
    

if imputate_bg:
    print('Imputating BG values')
    
    lstm_hr_train_df = pl.read_csv('data/lstm_hr_train.csv', schema_overrides=lstm_train_schema)
    lstm_hr_test_df = pl.read_csv('data/lstm_hr_test.csv', schema_overrides=lstm_test_schema)
    
    all_hr_pred_df = pl.concat([lstm_hr_train_df.drop(['id','time_delta','bg_null','hr_null','bg+1:00']), lstm_hr_test_df.drop(['id','time_delta','bg_null','hr_null'])])
    all_hr_pred_df = all_hr_pred_df.filter(pl.col('bg').is_not_null())

    t = all_hr_pred_df['bg'].to_numpy().reshape(-1,1)
    X = all_hr_pred_df.drop('bg').to_numpy()

    X_scaler = MinMaxScaler()
    t_scaler = MinMaxScaler()
    X = X_scaler.fit_transform(X)
    t = t_scaler.fit_transform(t).ravel()
    
    dtrain_full = xgb.DMatrix(X, label=t, missing=np.nan)
    
    best_boost_round = 423
    
    best_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 9,
        'learning_rate': 0.9654913454263729,
        'min_child_weight': 7,
        'reg_lambda': 0.06472962504852985
    }

    bg_pred_model = xgb.train(
        best_params,
        dtrain_full,
        num_boost_round=best_boost_round,
        verbose_eval=True
    )

    imputate_values(lstm_hr_train_df, bg_pred_model,bg_pred_features,'bg', 'data/lstm_bg_train.csv')
    imputate_values(lstm_hr_test_df, bg_pred_model,bg_pred_features,'bg', 'data/lstm_bg_test.csv')



# CLEAN BG VALUES - NEED TO FIX THIS
# GROUP 1
# Max number of missing values in a row: 5
# Solution: Interpolate missing values, play with extrapolating initial and final missing values, or maybe not actually: going on a run for 20 minutes is still going to change hr and bg levels but could only last 4 readings Lets use more complex methods for these

# GROUP 2
# Max number of missing values in a row: 40
# Solution: MICE imputation? KNN Imputation? XGBoost Imputation?

# GROUP 3
# Max number of missing values in a row: 70
# Solution: Mask Missing Values Explicitly for the Model? XGBoost? MICE Imputation? KNN Imputation?


noisy_participants = ['p_num_p01', 'p_num_p05', 'p_num_p06', 'p_num_p21']
clean_participants = [p for p in all_participants if p not in noisy_participants]





# NORMALISE DATA AND SAVE AS TENSORS
noisy_participants = ['p01', 'p05', 'p06', 'p21']
participants_in_both = ['p01', 'p02', 'p04', 'p05', 'p06', 'p10', 'p11', 'p12']

participants_to_remove = ['p_num_p03', 'p_num_p15', 'p_num_p16', 'p_num_p18', 'p_num_p19','p_num_p21', 'p_num_p22', 'p_num_p24'] # The idea is that these are unknown participants so no point encoding them

if normalise:
    print('Normalising data')
    lstm_bg_train_df = pl.read_csv('data/lstm_bg_train.csv', schema_overrides=lstm_train_schema)
    lstm_bg_test_df = pl.read_csv('data/lstm_bg_test.csv', schema_overrides=lstm_test_schema)
    
    columns_to_normalise = ["time", "insulin", "carbs", "hr", "steps", "cals"]

    train_data_to_normalise = lstm_bg_train_df.select(columns_to_normalise).to_numpy()
    
    scaler = MinMaxScaler()
    train_normalised_data = scaler.fit_transform(train_data_to_normalise)

    lstm_bg_train_df = lstm_bg_train_df.with_columns(
        [pl.Series(columns_to_normalise[i], train_normalised_data[:, i]) for i in range(len(columns_to_normalise))]
    )

    test_data_to_normalise = lstm_bg_test_df.select(columns_to_normalise).to_numpy()
    test_normalised_data = scaler.transform(test_data_to_normalise)

    lstm_bg_test_df = lstm_bg_test_df.with_columns(
        [pl.Series(columns_to_normalise[i], test_normalised_data[:, i]) for i in range(len(columns_to_normalise))]
    )

    train_bg_data = lstm_bg_train_df.select('bg').to_numpy()
    train_bg_h_data = lstm_bg_train_df.select('bg+1:00').to_numpy()
    test_bg_data = lstm_bg_test_df.select(['bg']).to_numpy()

    bg_scaler = MinMaxScaler()
    bg_scaler.fit(train_bg_data)

    bg_train_normalised = bg_scaler.transform(train_bg_data)
    bg_h_train_normalised = bg_scaler.transform(train_bg_h_data)
    bg_test_normalised = bg_scaler.transform(test_bg_data)
    

    lstm_bg_train_df = lstm_bg_train_df.with_columns(
        [
            pl.Series('bg', bg_train_normalised.flatten()),
            pl.Series('bg+1:00', bg_h_train_normalised.flatten())
        ]
    )

    lstm_bg_test_df = lstm_bg_test_df.with_columns(
        pl.Series('bg', bg_test_normalised.flatten()) 
    )
    
    lstm_bg_train_df = lstm_bg_train_df.with_columns(
        (pl.col("id").str.split('_').list[0].is_in(noisy_participants)).cast(pl.Int8).alias("Noisy Group")
    )   
    lstm_bg_train_df = lstm_bg_train_df.drop([col for col in participants_to_remove if col in lstm_bg_train_df.columns])
    
    lstm_bg_test_df = lstm_bg_test_df.with_columns(
    (pl.col("id").str.split('_').list[0].is_in(noisy_participants)).cast(pl.Int8).alias("Noisy Group")
    )
    lstm_bg_test_df = lstm_bg_test_df.drop([col for col in participants_to_remove if col in lstm_bg_test_df.columns])
    print(lstm_bg_test_df)
    print(lstm_bg_train_df)
    
    lstm_bg_test_df = lstm_bg_test_df.with_columns(
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
    print(lstm_bg_test_df)
    
    lstm_train_by_id = lstm_bg_train_df.group_by('id').agg(pl.all())
    lstm_test_by_id = lstm_bg_test_df.group_by('id').agg(pl.all())
    
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