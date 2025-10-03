import mlflow, os

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow.sklearn
from mlflow.models import infer_signature

# Parameters
N_ESTIMATORS = 100
MAX_DEPTH = 6
MAX_FEATURES = 3
# Run name
RUN_NAME = "run-01"

# Set an experiment name, unique and case-sensitive
# It will create a new experiment if the experiment with given doesn't exist
mlruns_path = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")



exp_name = "Diabetes Experiments"
bucket = os.getenv("MLFLOW_S3_BUCKET", "").strip()
existing = mlflow.get_experiment_by_name(exp_name)
if existing is None:
    # if you don’t set bucket, artifacts will land under the file backend folder
    artifact_location = None
    exp_id = mlflow.create_experiment(exp_name, artifact_location=artifact_location)
    exp = mlflow.get_experiment(exp_id)
else:
    exp = existing

# Start RUN
mlflow.start_run(run_name= RUN_NAME,                      # specify name of the run
                 experiment_id= exp.experiment_id)        # experiment id under which to create the current run
                 
# Log parameters
mlflow.log_param("n_estimators", N_ESTIMATORS)
mlflow.log_param("max_depth", MAX_DEPTH)
mlflow.log_param("max_features", MAX_FEATURES)

# Load dataset
db = load_diabetes()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, max_features=MAX_FEATURES)
rf.fit(X_train, y_train)

pred_train = rf.predict(X_train)
predictions = rf.predict(X_test)

signature = infer_signature(X_train, pred_train)
mlflow.sklearn.log_model(
    sk_model=rf,
    name="trained_model",
    signature=signature,
    input_example=X_train[:2]
)

# correct order: y_true, y_pred
mlflow.log_metric("training_r2_score", r2_score(y_train, pred_train))
mlflow.log_metric("testing_r2_score", r2_score(y_test, predictions))
mlflow.log_metric("training_mse", mean_squared_error(y_train, pred_train))
mlflow.log_metric("testing_mse", mean_squared_error(y_test, predictions))

mlflow.end_run()


