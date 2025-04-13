import os
import re
import yaml
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <runcard>")
    sys.exit(1)

runcard = sys.argv[1]

with open(runcard) as f:
    theoryid = yaml.safe_load(f)['theory']['theoryid']

with open("/user/roystege/data/github/nnpdf/validphys2/src/validphys/scalevariations/scalevariationtheoryids.yaml") as f:
    scalevariationtheoryids = yaml.safe_load(f)['scale_variations_for']

variations=[i for i in scalevariationtheoryids if i['theoryid'] == theoryid][0]['variations']

alphas_vals = [0.114, 0.115, 0.116, 0.117, 0.118, 0.119, 0.120, 0.121, 0.122, 0.123, 0.124, 0.125]
alphas_mapping = {alphas :variations[f"({alphas:.3f})"] for alphas in alphas_vals}

# Input and output directories
input_dir = "/data/theorie/rstegeman/fits"  # Directory containing original files
output_dir = "/data/theorie/rstegeman/fits/generated_files"
os.makedirs(output_dir, exist_ok=True)

# Extract base name from runcard
base_name = os.path.basename(runcard)
base_name_no_alpha = re.sub(r'alphas_\d{4}', 'alphas_{alpha_str}', base_name)

# Read original file content
input_path = os.path.join(input_dir, base_name)
with open(input_path, "r") as f:
    content = f.readlines()

# Modify only the `theoryid:` line
for alpha, new_theoryid in alphas_mapping.items():
    new_content = []
    for line in content:
        if re.match(r"^\s*theoryid:\s+", line):  # Match `theoryid:`
            new_content.append(re.sub(r"theoryid:\s+.*", f"theoryid: {new_theoryid}", line))
        else:
            new_content.append(line)

    # Save the modified file
    alpha_str = f"{int(alpha*1000):04d}"  # Convert to format e.g., 0114 for 0.114
    new_filename = base_name_no_alpha.format(alpha_str=alpha_str)
    output_path = os.path.join(output_dir, new_filename)
    with open(output_path, "w") as out_f:
        out_f.writelines(new_content)

print("File generation complete.")
