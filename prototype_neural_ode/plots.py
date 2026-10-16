import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib

# Load the CSV file
losses_df = pd.read_csv('losses_icbf.csv')
# Check if the 'Epoch' column already exists
if 'Epoch' not in losses_df.columns:
    # Add an 'Epoch' column starting from 1 up to the number of rows
    losses_df['Epoch'] = range(1, len(losses_df) + 1)

# Save the modified DataFrame to a new CSV file

# Remove the first row from the DataFrame
losses_df = losses_df.iloc[1:]
losses_df = losses_df[(losses_df['Loss'] != 'Loss') & (losses_df['Epoch'] != 'Epoch')]
modified_csv_path = 'loss/losses_icbf.csv'  # Specify your desired file path
losses_df.to_csv(modified_csv_path, index=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(losses_df['Loss'], label='ICBF Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Convert to TikZ
tikz_code = tikzplotlib.get_tikz_code()
with open('plot_tikz_code.tex', 'w') as file:
    file.write(tikz_code)