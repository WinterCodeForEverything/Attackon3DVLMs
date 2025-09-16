import numpy as np
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt

# l1 = [5, 2, 1, 0.5, 0.2, 0.1]
line1 = {
    'color': 'blue',
    'name': 'vision attack on GPT4Point',
    'CSD': [1.73, 2.38, 3.14, 3.57],
    'AGS': [5.53, 14.42, 17.82, 20.71],
}

line2 = {
    'color': 'blue',
    'name': 'vision attack on PointLLM',
    'CSD': [1.36, 1.75, 3.32, 3.72, 4.23],
    'AGS': [4.35, 5.5, 7.14, 7.38, 9.2],
}

# l2 = [20, 10, 5, 2, 1, 0.1]
line3 = {
    'color': 'red',
    'name': 'caption attack on GPT4Point',
    'CSD': [1.28, 1.65, 2.26, 2.97, 3.82], 
    'AGS': [5.49, 8.98, 16.7, 19.88, 28.6],
}

line4 = {
    'color': 'red',
    'name': 'caption attack on PointLLM',
    'CSD': [2.05, 3.25, 4.48, 5.11 ],
    'AGS': [5,  7.33, 9.15, 11.13],
}


def smooth_curve(x, y, points=6):
    x = np.array(x)
    y = np.array(y)
    if len(x) < 3:
        return x, y  # Not enough points to smooth
    x_new = np.linspace(x.min(), x.max(), points)
    spline = make_interp_spline(x, y, k=2 if len(x) == 3 else 3)
    y_new = spline(x_new)
    return x_new, y_new



CSD1, AGS1 = line1['CSD'], line1['AGS']
CSD2, AGS2 = line2['CSD'], line2['AGS']
color1, color2 = line1['color'], line2['color']
name1, name2 = line1['name'], line2['name']


CSD1_smooth, AGS1_smooth = smooth_curve(line1['CSD'], line1['AGS'])
CSD2_smooth, AGS2_smooth = smooth_curve(line2['CSD'], line2['AGS'])



plt.figure(figsize=(8, 6))

# plt.rcParams.update({'legend.loc': 'upper left'})
plt.rcParams.update({'legend.fontsize': 14, 'axes.labelsize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18})

# Plot line1
plt.plot(CSD1_smooth, AGS1_smooth, marker='o', linestyle='-', color=color1, label=name1)
# for x, y in zip(CSD1_smooth, AGS1_smooth):
#     plt.text(x, y, f"{x},{y}", fontsize=9, ha='right', va='bottom', color=color1) # f'l={label}',

# Plot line2
plt.plot(CSD2_smooth, AGS2_smooth, marker='s', linestyle='--', color=color2, label=name2)
# for x, y in zip(CSD2_smooth, AGS2_smooth):
#     plt.text(x, y, f'{x},{y}', fontsize=9, ha='left', va='top', color=color2) #f'l={label}',
    
# Prepare data for caption attack lines
CSD3, AGS3 = line3['CSD'], line3['AGS']
CSD4, AGS4 = line4['CSD'], line4['AGS']
color3, color4 = line3['color'], line4['color']
name3, name4 = line3['name'], line4['name']

CSD3_smooth, AGS3_smooth = smooth_curve(line3['CSD'], line3['AGS'])
CSD4_smooth, AGS4_smooth = smooth_curve(line4['CSD'], line4['AGS'])




# Plot line3 (caption attack, blue, dotted)
plt.plot(CSD3_smooth, AGS3_smooth, marker='o', linestyle='-', color=color3, label=name3)
# for x, y in zip(CSD3_smooth, AGS3_smooth):
#     plt.text(x, y, f'{x:.2f},{y:.2f}', fontsize=9, ha='right', va='top', color=color3)

# Plot line4 (caption attack, red, dash-dot)
plt.plot(CSD4_smooth, AGS4_smooth, marker='s', linestyle='--', color=color4, label=name4)
# for x, y in zip(CSD4_smooth, AGS4_smooth):
#     plt.text(x, y, f'{x:.2f},{y:.2f}', fontsize=9, ha='left', va='bottom', color=color4)

plt.xlabel('CSD')
plt.ylabel('AGS')
# plt.title('AGS vs CSD for targeted Vision Attacks')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('AGS_CSD_targeted_attack_on_Objaverse_smooth_captioning.jpg', format='jpg')
plt.show()
