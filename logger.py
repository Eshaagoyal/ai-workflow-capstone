import os
import csv
from datetime import datetime

LOG_DIR = os.path.join(".", "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def update_predict_log(country, y_pred, y_proba, query_date, runtime, model_version, test=False):
    """
    Log inference queries with timestamp, country, runtime, and model version.
    """
    prefix = "test-" if test else ""
    logfile = os.path.join(LOG_DIR, f"{prefix}predict-log.csv")
    write_header = not os.path.exists(logfile)
    
    with open(logfile, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'country', 'y_pred', 'y_proba', 'query_date', 'runtime', 'model_version'])
        writer.writerow([datetime.now().isoformat(), country, y_pred, y_proba, query_date, runtime, model_version])

def update_train_log(tag, date_range, eval_test, runtime, model_version, test=False):
    """
    Log training runs with performance metrics and execution time.
    """
    prefix = "test-" if test else ""
    logfile = os.path.join(LOG_DIR, f"{prefix}train-log.csv")
    write_header = not os.path.exists(logfile)
    
    with open(logfile, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'tag', 'date_range', 'rmse', 'runtime', 'model_version'])
        writer.writerow([datetime.now().isoformat(), tag, date_range, eval_test, runtime, model_version])