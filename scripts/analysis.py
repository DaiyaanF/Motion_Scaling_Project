import argparse
import os
from pathlib import Path

from utils.read_bag import sanitize_topic_name, read_and_save_mp4, read_dist, save_dist_to_csv, read_clutch, save_clutch_to_csv
from utils.plot import plot_min_distances_with_total_clutch

def change_base_name(original_path, base_name):
    path = Path(original_path)
    parts = list(path.parts)
    parts[-1] = base_name
    # save under videos
    return str(Path(*parts))

def read_and_save_videos(bag_file, video_topics, video_outputs, fps):
    if video_outputs:
        if len(video_outputs) != len(video_topics):
            raise ValueError("The number of output filenames must match the number of topics for videos.")
    else:
        video_folder = change_base_name(original_path=bag_file, base_name="videos")
        os.makedirs(video_folder, exist_ok=True)
        video_outputs = []
        for v_topic in video_topics:
            v_sanitized = sanitize_topic_name(v_topic)
            # Generate an output filename (e.g., "output_camera_image_raw.mp4")
            video_output = os.path.join(video_folder, "output_{}.mp4".format(v_sanitized))
            video_outputs.append(video_output)

    # Process each topic separately
    for video_topic, video_output_file in zip(video_topics, video_outputs):
        read_and_save_mp4(bag_file, video_topic, video_output_file, fps)
        print(f"Video output file: {video_output_file} for topic: {video_topic}")

def read_and_save_dist(bag_file, dist_topic, dist_outputs):
    if dist_outputs:
        if len(dist_outputs) != 1:
            raise ValueError("The number of output filenames must match the number of topics for distances.")
    else:
        analysis_folder = change_base_name(original_path=bag_file, base_name="analysis")
        os.makedirs(analysis_folder, exist_ok=True)
        # save under distances
        dist_file = os.path.join(analysis_folder, "min_dist.csv")

    save_dist_to_csv(bag_path=bag_file, csv_path=dist_file, topic=dist_topic)
    print(f"Distance output file: {dist_file} for topic: {dist_topic}")
    return dist_file

def read_and_save_clutch(bag_file, clutch_topic, clutch_outputs):
    if clutch_outputs:
        if len(clutch_outputs) != 1:
            raise ValueError("The number of output filenames must match the number of topics for clutch.")
    else:
        analysis_folder = change_base_name(original_path=bag_file, base_name="analysis")
        os.makedirs(analysis_folder, exist_ok=True)
        # save under clutch
        clutch_file = os.path.join(analysis_folder, "clutch.csv")

    save_clutch_to_csv(bag_path=bag_file, csv_path=clutch_file, topic=clutch_topic)
    print(f"Clutch output file: {clutch_file} for topic: {clutch_topic}")
    return clutch_file

def plot_dist_clutch(dist_csv, clutch_csv, plot_outputs):
    if plot_outputs:
        if len(plot_outputs) != 1:
            raise ValueError("The number of output filenames must match the number of trial.")
    else:
        # save under distances
        plot_file = change_base_name(original_path=dist_csv, base_name="dist_clutch.png")
    plot_min_distances_with_total_clutch(distance_csv=dist_csv, clutch_csv=clutch_csv, save_path=plot_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert image messages in a ROS bag to MP4 videos. "
                    "For each specified topic, a separate video is created."
    )
    parser.add_argument("-b", "--bag_file", required=True, help="Path to the input ROS bag file.")
    parser.add_argument("-v", "--videos", default=False, action="store_true",
                        help="Flag to indicate whether we want.")
    parser.add_argument("-d", "--distances", default=False, action="store_true",
                        help="Flag to indicate whether we want to extract distances.")
    parser.add_argument("-c", "--clutch", default=False, action="store_true",
                        help="Flag to indicate whether we want to extract clutching.")
    parser.add_argument("-p", "--plot", default=False, action="store_true",
                        help="Flag to indicate whether we want to save the plotting for distances with the total clutching number.")
    # parser.add_argument("-vt", "--video_topics", nargs="+",
    #                     help="List of topics to extract images from (e.g., /camera/image_raw /camera/front).")
    # parser.add_argument("-vf", "--video_folder", required=False,
    #                     help="Folder for Output MP4 file paths. ")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Frame rate for the output video (default: 15.0).")
    parser.add_argument("-vo", "--video_output", nargs="+",
                        help="List of output MP4 file paths, one for each topic. "
                             "If not provided, output filenames are generated using the topic names.")
    # parser.add_argument("-dt", "--dist_topic",
    #                     help="List of topics to extract distances data.")
    parser.add_argument("-do", "--dist_output", nargs="+",
                        help="List of output csvs for distances data. "
                             "If not provided, output filenames are generated using the topic names.")
    parser.add_argument("-co", "--clutch_output", nargs="+",
                        help="List of output csvs for clutch data. "
                             "If not provided, output filenames are generated using the topic names.")
    parser.add_argument("-po", "--plot_output", nargs="+",
                    help="List of output pngs for plotting. "
                            "If not provided, output filenames are generated using the default names.")
    args = parser.parse_args()

    video_topics = ['/cameraL/video', '/cameraR/video']
    dist_topic = '/min_distances'
    clutch_topic = '/footpedal/clutch'
    
    # read and save the videos
    if args.videos:
        print("Video conversion is enabled.")
        # Check if outputs were provided. If not, generate output filenames based on topic names.
        video_outputs = args.video_output
        read_and_save_videos(bag_file=args.bag_file, video_topics=video_topics, video_outputs=video_outputs, fps=args.fps)
    
    # raed and save the distances
    if args.distances:
        print("Distance extraction is enabled.")
        # Check if outputs were provided. If not, generate output filenames based on topic names.
        dist_outputs = args.dist_output
        dist_csv_path = read_and_save_dist(bag_file=args.bag_file, dist_topic=dist_topic, dist_outputs=dist_outputs)

    # read and save the clutch
    if args.clutch:
        print("Clutching extraction is enabled.")
        # Check if outputs were provided. If not, generate output filenames based on topic names.
        clutch_outputs = args.clutch_output
        clutch_csv_path = read_and_save_clutch(bag_file=args.bag_file, clutch_topic=clutch_topic, clutch_outputs=clutch_outputs)
    
    if args.distances and args.clutch and args.plot:
        print("Plotting is enabled.")        
        plot_outputs = args.plot_output
        plot_dist_clutch(dist_csv=dist_csv_path, clutch_csv=clutch_csv_path, plot_outputs=plot_outputs)