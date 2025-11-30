# Training Configuration for Fast Iteration
# Adjust these parameters to balance speed vs accuracy

# Dataset sampling ratios (1.0 = full dataset)
TRAIN_SUBSET_RATIO = 0.3  # Use 30% for fastest training, increase for better accuracy
TEST_SUBSET_RATIO = 0.5   # Use 50% for faster validation

# Training parameters
BATCH_SIZE = 256          # Larger batch size for better GPU utilization
NUM_EPOCHS = 10           # Reduced for faster iteration
LEARNING_RATE = 1e-3

# Early stopping thresholds
TARGET_ACCURACY = 0.85    # Stop when reaching this accuracy
PATIENCE_EPOCHS = 3       # Stop if no improvement for this many epochs
MIN_IMPROVEMENT = 0.005   # Minimum improvement to consider progress

# Performance settings
PIN_MEMORY = True         # Faster data transfer to GPU
NON_BLOCKING = True       # Async data transfer

# For production training, use these settings:
# TRAIN_SUBSET_RATIO = 1.0
# TEST_SUBSET_RATIO = 1.0  
# NUM_EPOCHS = 35
# TARGET_ACCURACY = 0.96