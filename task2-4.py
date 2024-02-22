import matplotlib.pyplot as plt
import numpy as np

with open("yeoviltondata.txt", "r") as file:
    lines = file.readlines()

yearly_data = {}

for line in lines[7:]:
    parts = line.split()
    year = float(parts[0])
    values = []
    for val in parts[1:]:
        if val == '---':
            values.append(None)  # 将'---'视为缺失值
        else:
            values.append(float(val))
    if year in yearly_data:
        yearly_data[year].append(values)
    else:
        yearly_data[year] = [values]

print(yearly_data)

new_yearly = []
for year, data_list in yearly_data.items():
    x = round(sum([entry[1] for entry in data_list if entry[1] is not None])/len(yearly_data[year]), 6)
    y = round(sum([entry[2] for entry in data_list if entry[2] is not None])/len(yearly_data[year]), 6)
    z = round(sum([entry[4] for entry in data_list if entry[4] is not None])/len(yearly_data[year]), 6)
    new_yearly.append([year, x, y, z])

print(yearly_data)

with open("task_2_solution.txt", "r") as file:
     lines2 = file.readlines()

solution = []
for i in lines2:
    p = i.split()
    float_data = [float(value) for value in p]
    solution.append(float_data)

print(solution)


def compare(list1, list2):
    if list1 == list2:
       print("CORRECT!")


compare(solution, new_yearly)


def dna_process(filename):
    with open(filename, 'r') as file:
        d = file.readlines()
    value = ''
    for L in d:
        value += L[:-2]

    return value


def count_cg_content(dna):
    letter_counts = {}
    for n in dna:
        if n in letter_counts:
            letter_counts[n] += 1
        else:
            letter_counts[n] = 1
    print(letter_counts)
    cg_content = (letter_counts['C'] + letter_counts['G'])*100/len(dna)
    print(f"The CG content is {cg_content}%")

    return letter_counts, cg_content


def complement_sequence(dna):

    char_map = str.maketrans({'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'})
    complement = dna.translate(char_map)

    return complement


def find_islands_print(value, size, cg_threshold):
    num_fragments = len(value) - size + 1
    islands = []
    island_positions = []
    cg_contents = []

    for q in range(num_fragments):
        start = q
        end = q + size
        group = value[start:end]
        cg_count = group.count('C') + group.count('G')
        cg_content = cg_count / size
        if cg_content >= cg_threshold:
            islands.append(group)
            island_positions.append((start, end))
            cg_contents.append(cg_content)
            print(f"The CG content of island at position {start}-{end} is {cg_content}%")

    return islands, island_positions, cg_contents


def find_islands_plt(value, size):
    num_fragments = len(value) - size + 1
    contents = []

    for p in range(num_fragments):
        group = value[p:p + size]
        cg_count = group.count('C') + group.count('G')
        cg_content = cg_count / size
        contents.append(cg_content)

    return contents


Thresdole = 0.5
dna_name = ['DNA_lambdavirus.fasta', 'redmt_DNA_algae.fasta', 'mtDNA_homosapiens.fasta']
color_of_lines = ['green', 'red', 'blue']

for i, j in zip(dna_name, color_of_lines):
    plt.plot(range(len(find_islands_plt(dna_process(i), 200))),
             find_islands_plt(dna_process(i), 200), label='CG Content', linewidth=0.6, color=j)

    above_threshold = np.array(find_islands_plt(dna_process(i), 200)) >= Thresdole

    cg_contents_nan = np.array(find_islands_plt(dna_process(i), 200))

    cg_contents_nan[~above_threshold] = np.nan

    plt.plot(range(len(find_islands_plt(dna_process(i), 200))), cg_contents_nan,
             linestyle='-', color='brown', linewidth=0.6, label='High CG Content')
    # Islands above the threshold are marked with brown lines
    
    # find_islands_print(dna_process(i), 200, 0.5)

plt.axhline(y=Thresdole, color='green', linestyle='--', label='Threshold')
plt.xlabel('Position in DNA Sequence')
plt.ylabel('CG Content (%)')
plt.title('CG Content of DNA Sequence')
plt.legend()
plt.show()
