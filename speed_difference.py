import numpy as np
import jax
import jax.numpy as jnp
import time

# Set up the size of the matrices
matrix_size = 1000

# Create two random matrices in NumPy
np_matrix1 = np.random.rand(matrix_size, matrix_size)
np_matrix2 = np.random.rand(matrix_size, matrix_size)

# Create two random matrices in JAX (JAX uses the same random API as NumPy, but operates on its own backend)
jax_matrix1 = jnp.array(np_matrix1)
jax_matrix2 = jnp.array(np_matrix2)

### NumPy Timing ###
start_time_np = time.time()
np_result = np.dot(np_matrix1, np_matrix2)
end_time_np = time.time()

numpy_time = end_time_np - start_time_np
print(f"NumPy time for matrix multiplication: {numpy_time:.6f} seconds")

### JAX Timing ###
# JAX operations are lazy; they are not executed immediately. We use .block_until_ready() to ensure it runs.
start_time_jax = time.time()
jax_result = jnp.dot(jax_matrix1, jax_matrix2)
jax_result.block_until_ready()  # Make sure the operation completes
end_time_jax = time.time()

jax_time = end_time_jax - start_time_jax
print(f"JAX time for matrix multiplication: {jax_time:.6f} seconds")

# Print the speedup factor
speedup = numpy_time / jax_time
print(f"JAX is approximately {speedup:.2f}x faster than NumPy")