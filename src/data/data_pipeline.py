from .loader import CIFAR10DataLoader
from .cleaner import DataCleaner
from .validator import DataValidator


class DataPipeline:
    def __init__(self):
        self.loader=CIFAR10DataLoader()
        self.cleaner = DataCleaner()
        self.validator = DataValidator()
        
        
    def run_pipeline(self):
        print('Starting data pipeline')
        
        
        #load
        (x_train,y_train),(x_test,y_test)=self.loader.load_data()
        print('Data loaded successfully')
        
        
        #validate 
        self.validator.validate_shapes(x_train,y_train,x_test,y_test)
        self.validator.check_data_range(x_train)
        print('Data validation passed!')
        
        #clean
        x_train_clean = self.cleaner.normalize_images(x_train)
        x_test_clean = self.cleaner.normalize_images(x_test)
        y_train_clean = self.cleaner.one_hot_encode_labels(y_train)
        y_test_clean = self.cleaner.one_hot_encode_labels(y_test)
        print("Data cleaning completed")
        
        return (x_train_clean,y_train_clean),(x_test_clean,y_test_clean)
    
    
if __name__ == "__main__":
    pipeline = DataPipeline()
    (x_train, y_train), (x_test, y_test) = pipeline.run_pipeline()
    print(f"Training data: {x_train.shape}, Training labels: {y_train.shape}")