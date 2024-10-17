import jax
import jax.numpy as jnp

def find_key_with_randint(target_value, target_iteration, minval, maxval, actions, max_attempts=10000):
    key = jax.random.PRNGKey(0)  # Start with a random seed
    for i in range(max_attempts):
        subkey = jax.random.split(key, num=1)[0]
        # Simulate `target_iteration` random draws
        indices = [jax.random.randint(subkey + (actions * 15), shape=(), minval=minval, maxval=maxval) for _ in range(target_iteration)]
        
        # Check if the value in the target_iteration matches the target
        if indices[-1] == target_value:
            print(f"Key found after {i+1} attempts")
            return subkey
        key = jax.random.split(key, num=2)[0]  # Split key for the next iteration
    print("Key not found")
    return None


    
    
if __name__ == "__main__":
    # Example: find a key where 9 is selected in the 5th iteration
    target_value = 9
    target_iteration = 5
    minval = 0
    maxval = 10  # Assuming len(nodes_final)-1 is 10
    actions = 15  # Value of self.number_of_actions

    key = find_key_with_randint(target_value, target_iteration, minval, maxval, actions)

   # Modify the check to see if key is not None
    if key is not None:
        print("Key:", key)
    else:
        print("Key not found")