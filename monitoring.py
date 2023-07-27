import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib
from prefect import task, Flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""

reference_data = pd.read_parquet('data/reference.parquet')

with open('/home/mussie/Music/home projects/nice_one/seving-ml-model-using-fastapi-and-docker/model.pkl', 'rb') as f_in:
    
    model = joblib.load(f_in)

raw_data = pd.read_csv('/home/mussie/Music/home projects/nice_one/seving-ml-model-using-fastapi-and-docker/Bank Customer Churn Prediction.csv')

# Set the start date to March 1, 2023
begin = datetime.datetime(2023, 3, 1, 0, 0)

# Define the number of rows to take as a single day observation
rows_per_day = 10

num_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']
cat_features = []
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task
def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
            conn.execute(create_table_statement)

@task
def calculate_metrics_postgresql(curr, current_date, data):
    # Calculate the prediction for the given data
    predictions = model.predict(data[num_features + cat_features].fillna(0))
    
    # Calculate the report for the given data
    report.run(reference_data=reference_data, current_data=data, column_mapping=column_mapping)
    result = report.as_dict()
    
    # Get the relevant metrics from the report
    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    # Store the metrics and prediction results in the database
    for prediction in predictions:
        curr.execute(
            "insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
            (current_date, prediction_drift, num_drifted_columns, share_missing_values)
        )

@Flow
def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        for i in range(0, len(raw_data), rows_per_day):
            current_date = begin + datetime.timedelta(days=i // rows_per_day)
            current_data = raw_data.iloc[i:i + rows_per_day]

            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, current_date, current_data)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")

if __name__ == '__main__':
    batch_monitoring_backfill()
