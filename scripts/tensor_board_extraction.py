import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Path to your event file
event_path = 'lightning_logs/version_6'  # Adjust this to your actual path
ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()

# Extract scalars
train_loss = ea.Scalars('train_loss')  # Adjust tag as needed
val_loss = ea.Scalars('val_loss')

# Get steps and values
train_steps = [x.step for x in train_loss]
train_values = [x.value for x in train_loss]
val_steps = [x.step for x in val_loss]
val_values = [x.value for x in val_loss]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_values, label='Train Loss')
plt.plot(val_steps, val_values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Presentation/loss_plot.png')  # Save as static image
plt.show()
