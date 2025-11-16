import mlflow

def setup_mlflow():
    mlflow.set_experiment("cifar10_classification")
    print("MLflow experiment setup completed")

if __name__ == "__main__":
    setup_mlflow()
