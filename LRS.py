'''
Basic Wrapper function explanation

# Create function
def example_func(a,b):
    return a+b

# Takes in the function as a parameter
def wrapper_for_func(func):
    print("Received function")
    my_function = func # Assign the function to a variable
    print(my_function) # This will print the function we used as a parameter

# wrapper_for_func(example_func) This is how it would look/work. All it would return is "I have a function" not the actual function we put in as an argument
'''

'''
Decorator function explanation

Create the function
def add(a, b):
    return a + b

Create the wrapper. Essentially all this does is runs the inner function
def wrapper(func): Requires the function as a parameter 

    def inner(a, b): Parameters for the function
        print("Before")

        result = func(a, b) Assign the function to a variable

        print("After")

        return result // This returns the variable, result() would return the called the function 

    return inner // Returns the function but does not call it
    
wrapped_func = wrapper(add) Adds the function to a variable 
wrapped_func(3,4) this would call the actual function
'''

# Wrapper to pass as a parameter
def wrapper(initial_lr, min_lr):
    "Returns the inner function"
    def lr_decay_schedule(epoch):
        new_lr = initial_lr * (0.5 ** epoch)
        final_lr = max(new_lr, min_lr)
        print(f"Epoch {epoch + 1}: Learning rate is {final_lr}")
        return final_lr
    return lr_decay_schedule # Returns the function but does not call it