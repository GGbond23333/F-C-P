import matplotlib.pyplot as plt


genetic_data = 'cgtacgttgacgtgcgtacgtgcgaggggtatacgta'
base_counts = [genetic_data.count(val) for val in 'acgt']
plt.bar(['A', 'C', 'G', 'T'], base_counts)
plt.show()
