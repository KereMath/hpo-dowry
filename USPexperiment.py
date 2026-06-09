import itertools
import numpy as np
import time

# Function to simulate the Secretary Problem with a threshold rule
def secretary_problem_with_threshold(perm, k, threshold):
    n = len(perm)

    # Observation phase: just look at the first k candidates, never pick them.
    # After the observation phase, select the first candidate whose score
    # reaches (>=) the threshold. This is where `threshold` actually drives
    # the decision, so different threshold values yield different results.
    for i in range(k, n):
        if perm[i] >= threshold:
            return perm[i]  # Select the first candidate that meets the threshold

    # If no candidate met the threshold, we are forced to take the last one
    return perm[-1]

# Parameters
n = 10  # Number of candidates
candidates = list(range(1, n + 1))  # Candidates scored from 1 to 20
perms = list(itertools.permutations(candidates))  # All permutations of candidates

# Array to store results for different thresholds and observation sizes
threshold_results = []

# Test multiple threshold values
for threshold in range(1, n + 1):  # Try different threshold values (1 to n)
    scores = []
    
    # Measure the time taken for each threshold
    start_time = time.time()  # Start the timer
    
    # Iterate over each permutation
    for perm in perms:
        # Use 1/e rule for observation size
        k = int(n / np.e)
        score = secretary_problem_with_threshold(perm, k, threshold)
        scores.append(score)
    
    # Calculate average score for this threshold
    average_score = np.mean(scores)
    
    # End the timer
    end_time = time.time()
    
    # Calculate the total time taken for this threshold
    time_taken = end_time - start_time
    
    # Store the threshold, average score, and time taken
    threshold_results.append((threshold, average_score, time_taken))

# Find the best threshold that maximizes the average score
best_threshold, best_avg_score, _ = max(threshold_results, key=lambda x: x[1])

# Output results
print(f"Best Threshold: {best_threshold}")
print(f"Maximized Average Score: {best_avg_score}")

# To see all threshold values, their average scores, and time taken
for threshold, avg, time_taken in threshold_results:
    print(f"Threshold: {threshold}, Average Score: {avg}, Time Taken: {time_taken:.4f} seconds")
