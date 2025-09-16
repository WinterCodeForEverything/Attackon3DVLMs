import numpy as np
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt

l1 = [5, 2, 1, 0.5, 0.2, 0.1]
line1 = {
    'color': 'blue',
    'name': 'vision attack on GPT4Point',
    'CSD': [1.89, 2.05, 2.33, 3.07, 3.53], 
    'AGS': [28.06, 23.44, 10.53, 2.74, 0.75],  
}

line2 = {
    'color': 'blue',
    'name': 'vision attack on PointLLM',
    'CSD': [1.49, 1.76, 2.57, 3.55, 4.34], 
    'AGS': [45.53, 39.53, 24.63, 18.75, 15.38],
}

# l2 = [20, 10, 5, 2, 1, 0.1]
line3 = {
    'color': 'red',
    'name': 'caption attack on GPT4Point',
    'CSD': [1.09, 1.46, 1.85, 2.85, 3.62],
    'AGS': [36.68, 21.86, 15.53, 11.93, 8.7],
}

line4 = {
    'color': 'red',
    'name': 'caption attack on PointLLM',
    'CSD': [ 2.28, 3.15, 4.02, 4.83 ],
    'AGS': [ 40.25, 28.65, 25.43, 18.78],
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

# plt.rcParams.update({'legend.fontsize': 16, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
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
# for x, y in zip(CSD3, AGS3):
#     plt.text(x, y, f'{x},{y}', fontsize=9, ha='right', va='top', color=color3)

# Plot line4 (caption attack, red, dash-dot)
plt.plot(CSD4_smooth, AGS4_smooth, marker='s', linestyle='--', color=color4, label=name4)
# for x, y in zip(CSD4, AGS4):
#     plt.text(x, y, f'{x},{y}', fontsize=9, ha='left', va='bottom', color=color4)

plt.xlabel('CSD')
plt.ylabel('AGS')
# plt.title('AGS vs CSD for targeted Vision Attacks')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('AGS_CSD_untargeted_attack_on_Objaverse_smooth_captioning.jpg', format='jpg')
plt.show()
