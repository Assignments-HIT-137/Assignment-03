# Demonstrating OOP concepts

# Inheritance & Method Overriding
class BaseModel:
    def run(self, data):
        raise NotImplementedError("Subclass must implement run()")

class CustomModel(BaseModel):
    def run(self, data):
        return f"Processed {data} using CustomModel"

# Encapsulation
class SecureData:
    def __init__(self, value):
        self.__secret = value   # private variable

    def get_secret(self):
        return self.__secret

# Polymorphism
def process_model(model, data):
    return model.run(data)

# Decorator Example
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Running {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logger
def decorated_function(x):
    return x * 2
