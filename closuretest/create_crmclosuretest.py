import os
import re

# Alpha to theoryid mapping
alpha_mapping = {
    0.114: 831, 0.115: 832, 0.116: 833, 0.117: 834,
    0.118: 717, 0.119: 835, 0.120: 836, 0.121: 837,
    0.122: 838, 0.123: 839, 0.124: 840, 0.125: 841
}

# Input and output directories
input_dir = "/data/theorie/rstegeman/fits"  # Directory containing original files
output_dir = "/data/theorie/rstegeman/fits/generated_files"
os.makedirs(output_dir, exist_ok=True)

# Process each xxx from 001 to 025
for i in range(1, 26):
    original_filename = f"250306-{i:03d}-rs-closuretest-alphas-0118.yml"
    input_path = os.path.join(input_dir, original_filename)

    if not os.path.exists(input_path):
        print(f"Skipping missing file: {original_filename}")
        continue

    # Read original file content
    with open(input_path, "r") as f:
        content = f.readlines()

    # Modify only the `theoryid:` line
    for alpha, new_theoryid in alpha_mapping.items():
        if new_theoryid == 717:  # Skip 0.118 since it's already correct
            continue

        new_content = []
        for line in content:
            if re.match(r"^\s*theoryid:\s+\d+", line):  # Match only `theoryid: NNN`
                new_content.append(re.sub(r"\d+", str(new_theoryid), line, count=1))
            else:
                new_content.append(line)

        # Save the modified file
        alpha_str = f"{int(alpha*1000):04d}"  # Convert to format e.g., 0114 for 0.114
        new_filename = f"250306-{i:03d}-rs-closuretest-alphas-{alpha_str}.yml"
        output_path = os.path.join(output_dir, new_filename)
        with open(output_path, "w") as out_f:
            out_f.writelines(new_content)

print("File generation complete.")
