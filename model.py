import os
import numpy as np
import matplotlib.pyplot as plt
from utils import read_csv, plot, polylines2svg
from regularization import fit_line, fit_circle
from symmetry_detection import detect_symmetry
from curve_completion import complete_curve

def process_file(input_file, output_dir):
    # Read CSV file
    paths_XYs = read_csv(input_file)
    
    # Initialize a list to store processed data
    output_data = []
    
    # Process each path
    for path_index, path in enumerate(paths_XYs):
        for points_index, points in enumerate(path):
            # Fit line
            m, c = fit_line(points)
            print(f'Line fit: y = {m}x + {c}')
            
            # Fit circle
            xc, yc, R = fit_circle(points)
            print(f'Circle fit: center=({xc}, {yc}), radius={R}')
            
            # Detect symmetry
            symmetry = detect_symmetry(points)
            print(f'Symmetry detected: {symmetry}')
            
            # Complete curve
            completed_points = complete_curve(points, occlusion_type='connected')
            print(f'Completed curve points: {completed_points[:5]}...')  # Print a subset for brevity
            
            # Prepare data for saving
            path_id = path_index + 1
            for pt in completed_points:
                output_data.append([path_id, points_index + 1, pt[0], pt[1]])

    # Convert the list to a numpy array for saving
    output_array = np.array(output_data)
    
    # Create output CSV file path
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_csv_path = os.path.join(output_dir, f'{base_name}_sol.csv')
    
    # Save the processed data to a new CSV file
    np.savetxt(output_csv_path, output_array, delimiter=',', fmt='%d,%d,%f,%f')
    
    # Create SVG and PNG file paths
    output_svg_path = os.path.join(output_dir, f'{base_name}.svg')
    output_png_path = os.path.join(output_dir, f'{base_name}.png')
    
    # Plot and save the paths in both SVG and PNG formats
    plot(paths_XYs, output_svg_path, output_png_path)
    polylines2svg(paths_XYs, output_svg_path)

if __name__ == "__main__":
    # Define input files and output directory
    input_files = ['frag0.csv', 'frag1.csv', 'frag2.csv','isolated.csv','occlusion1.csv']
    output_dir = 'output_files'
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each input file
    for input_file in input_files:
        print(f'Processing {input_file}...')
        process_file(input_file, output_dir)
        print(f'Finished processing {input_file}.')


  
