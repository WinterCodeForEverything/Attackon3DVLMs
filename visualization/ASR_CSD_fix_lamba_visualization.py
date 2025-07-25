import matplotlib.pyplot as plt

# l = [5, 2, 1, 0.5, 0.2, 0.1]
# line1 = {
#     'color': 'blue',
#     'name': 'GPT4Point_ModelNet40_vision_attack',
#     'CSD': [0.72, 0.94, 1.11, 1.50, 1.95, 2.28],
#     'ASR': [26.5, 29.6, 44.8, 83.5, 90.2, 92.9],
# }

# line2 = {
#     'color': 'red',
#     'name': 'PointLLM_ModelNet40_vision_attack',
#     'CSD': [0.41, 0.79, 1.17, 1.68, 2.16, 2.44],
#     'ASR': [12.1, 14.8, 24.5, 49.5, 77.3, 85.4],
# }

l = [1, 0.5, 0.2, 0.1, 0.05, 0.01]
line1 = {
    'color': 'blue',
    'name': 'GPT4Point_ModelNet40',
    'CSD': [0.11, 0.68, 1.08, 1.30, 1.68, 2],
    'ASR': [5.5, 8.6, 13.4, 16.0, 17.1, 17.9],
}

line2 = {
    'color': 'red',
    'name': 'PointLLM_ModelNet40',
    'CSD': [0.13, 0.27, 0.52, 0.73, 0.96 ],
    'ASR': [4.6, 3.6, 5.7, 5.6, 4.9 ],
}




CSD1, ASR1 = line1['CSD'], line1['ASR']
CSD2, ASR2 = line2['CSD'], line2['ASR']
color1, color2 = line1['color'], line2['color']
name1, name2 = line1['name'], line2['name']

plt.figure(figsize=(8, 6))

# Plot line1
plt.plot(CSD1, ASR1, marker='o', linestyle='-', color=color1, label=name1)
for x, y, label in zip(CSD1, ASR1, l):
    plt.text(x, y, f'l={label}', fontsize=9, ha='right', va='bottom', color=color1)

# Plot line2
plt.plot(CSD2, ASR2, marker='s', linestyle='--', color=color2, label=name2)
for x, y, label in zip(CSD2, ASR2, l):
    plt.text(x, y, f'l={label}', fontsize=9, ha='left', va='top', color=color2)

plt.xlabel('CSD')
plt.ylabel('ASR (%)')
plt.title('ASR vs CSD for targeted Vision Attacks')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('ASR_CSD_targeted_vision_attack.jpg', format='jpg')
plt.show()
