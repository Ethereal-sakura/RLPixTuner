import os
import json
import argparse

def find_image_pairs(target_dir, output_name="image_pairs"):
    # List to hold the image pairs
    image_pairs = []
    supported_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    files = os.listdir(target_dir)
    file_set = set(files)

    # Scan the target directory for input/target pairs
    for filename in sorted(files):
        base_name, ext = os.path.splitext(filename)
        if ext.lower() not in supported_exts:
            continue
        if not base_name.endswith('-Input'):
            continue
        target_base = base_name.replace('-Input', '-Target')
        target_path = None
        for candidate_ext in supported_exts:
            candidate = target_base + candidate_ext
            if candidate in file_set:
                target_path = os.path.join(target_dir, candidate)
                break
        if target_path is None:
            continue
        input_path = os.path.join(target_dir, filename)
        image_pairs.append([input_path, target_path])

    # Write the image pairs to a JSON file
    output_file = os.path.join(target_dir, output_name)
    with open(output_file, 'w') as json_file:
        json.dump(image_pairs, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Generate image pairs from a directory containing .tif files.')
    parser.add_argument('directory', type=str, help='The directory to scan for input/target pairs')
    parser.add_argument('--output_name', type=str, default='image_pairs', help='Output file name')

    args = parser.parse_args()

    if os.path.isdir(args.directory):
        find_image_pairs(args.directory, output_name=args.output_name)
    else:
        print(f"Error: {args.directory} is not a valid directory.")

if __name__ == '__main__':
    main()