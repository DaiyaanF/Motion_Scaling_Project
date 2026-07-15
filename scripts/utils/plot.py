import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # optional: for clarity when computing median


def plot_min_distances_with_total_clutch(distance_csv, clutch_csv, save_path=None):
    """
    Reads distance events from a CSV file and total clutch event count from another CSV,
    then creates a plot displaying:
        - PSM1 and PSM2 minimum distances versus time (primary y-axis)
        - A text annotation showing the total clutch event count,
          as well as the average and median for both PSM distances.
    
    Optionally saves the plot to a file if 'save_path' is provided.
    
    Parameters:
        distance_csv (str): Path to the CSV file containing distance data.
            Expected columns: 'time', 'psm1_distance', 'psm2_distance'
        clutch_csv (str): Path to the CSV file containing clutch event data.
            Expected columns: 'event_time', 'clutch_count'
            It is assumed that the last row's 'clutch_count' represents the total count.
        save_path (str, optional): File path to save the plotted graph (e.g. 'output.png').
            If None, the graph is not saved.
    """
    # Read the CSV files
    distance_df = pd.read_csv(distance_csv)
    clutch_df = pd.read_csv(clutch_csv)
    
    # Convert columns to numpy arrays
    time_array = distance_df['time'].to_numpy()
    psm1_array = distance_df['psm1_distance'].to_numpy()
    psm2_array = distance_df['psm2_distance'].to_numpy()
    
    # Retrieve the total clutch count from the last row of the clutch CSV
    total_clutch = clutch_df['clutch_count'].iloc[-1] if not clutch_df.empty else 0
    
    # Compute average and median for both PSM distances
    psm1_avg = psm1_array.mean()
    psm1_med = np.median(psm1_array)
    psm2_avg = psm2_array.mean()
    psm2_med = np.median(psm2_array)
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot the distances for both PSMs
    ax1.plot(time_array, psm1_array, 'r-', label='PSM1 Distance')
    ax1.plot(time_array, psm2_array, 'b-', label='PSM2 Distance')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (m)')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Create a multiline annotation with clutch count, average, and median values
    stats_text = (
        f"Total Clutch Count: {total_clutch}\n"
        f"PSM1 Avg: {psm1_avg:.4f}, Median: {psm1_med:.4f}\n"
        f"PSM2 Avg: {psm2_avg:.4f}, Median: {psm2_med:.4f}"
    )
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
             ha='right', va='top', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.5))
    
    plt.title('Min Distances (PSM1 & PSM2) with Total Clutch Count and Statistics')
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)
        print("Graph saved to:", save_path)
    
    plt.show()