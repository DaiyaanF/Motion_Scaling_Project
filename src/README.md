# Motion Scaling Trajectory Tracking Experiment

## Active Scripts

| Script | Purpose |
|--------|---------|
| `bezier_collector.py` | Main experiment |
| `read_bezier_bag_v2.py` | Bag file reader |
| `bezier_trajectory_cubic.py` | Bezier curve generation (dependency) |
| `frame_transform_manager.py` | Coordinate transforms (dependency) |
| `dvrk_gui_simple.py` | GUI version (optional) |

## Quick Start

### 1. Start AMBF Simulation
```bash
./ambf_simulator --launch_file <the_launch_file_you_choose>
```

### 2. Run Experiment
```bash
cd ~/intel_motion_scaling/src

# Single scale factor
python bezier_collector.py --participant P001 --segments 4 --fixed "1.0"

# Multiple scale factors (one iteration each)
python bezier_collector.py --participant P001 --segments 4 --fixed "0.5,1.0,2.0"
```

### 3. Analyze Data
```bash
# Basic analysis
python read_bezier_bag_v2.py ~/intel_motion_scaling/bezier_data/P001/sliding_<timestamp>.bag --analyze --csv

# With plots
python read_bezier_bag_v2.py <bag_path> --analyze --csv --plot

# Extract video
python read_bezier_bag_v2.py <bag_path> --video --video-side stereo
```

## Experiment Flow

1. **Touch DOT** - Move PSM to start position (green dot)
2. **Track trajectory** - Follow the sliding window of points
3. **Complete iteration** - Reach end of trajectory
4. **Move away, touch DOT again** - Start next iteration (if multiple)
5. **Auto-shutdown** - Experiment ends after all iterations

## Data Output

Bag files saved to: `~/intel_motion_scaling/bezier_data/<participant_id>/`

Contents:
- `/session_metadata` - Experiment configuration
- `/bezier_events` - Start/complete events
- `/bezier_comprehensive` - 30Hz tracking data (position, velocity, progress)
- `/camera_left`, `/camera_right` - Stereo video at 10Hz

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--segments` | 4 | Number of Bezier curve segments |
| `--lookahead` | 100 | Points visible ahead |
| `--lookbehind` | 10 | Points visible behind |
| `--proximity-threshold` | 0.003 | Tracking distance threshold (meters) |
