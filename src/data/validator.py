class DataValidator:
    def __init__(self):
        pass
    
    def validate_shapes(self, x_train, y_train, x_test, y_test):
        assert x_train.shape == (50000, 32, 32, 3), "Invalid training data shape"
        assert y_train.shape == (50000, 1), "Invalid training labels shape"
        assert x_test.shape == (10000, 32, 32, 3), "Invalid test data shape"
        assert y_test.shape == (10000, 1), "Invalid test labels shape"
        print("All data shapes are valid")
    
    def check_data_range(self, images):
        assert images.min() >= 0 and images.max() <= 255, "Image pixel values out of range"
        print("Data range validation passed")