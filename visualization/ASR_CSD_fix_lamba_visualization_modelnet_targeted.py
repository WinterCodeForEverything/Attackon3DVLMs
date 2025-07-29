import numpy as np
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt

# l1 = [5, 2, 1, 0.5, 0.2, 0.1]
line1 = {
    'color': 'blue',
    'name': 'vision attack on GPT4Point',
    'CSD': [ 0.11, 0.68, 1.08, 1.30, 1.68, 2 ],
    'ASR': [ 5.5, 8.6, 13.4, 16.0, 17.1, 17.9 ],
}

line2 = {
    'color': 'blue',
    'name': 'vision attack on PointLLM',
    'CSD': [ 0.2, 0.47, 0.82, 0.97, 1.43, 1.74 ], #1.57, 1.68,
    'ASR': [ 4.4, 5.2, 6.8, 8.2, 12.6, 14.1], #9.9, 11.1,
}

# # l2 = [20, 10, 5, 2, 1, 0.1]
line3 = {
    'color': 'red',
    'name': 'caption attack on GPT4Point',
    'CSD': [0.35,  1.05, 1.29, 1.63, 2.09], #0.75,
    'ASR': [6.1,  10.3, 12.7, 15.4, 17.1], #7.0,
}

line4 = {
    'color': 'red',
    'name': 'caption attack on PointLLM',
    'CSD': [0.85, 1.77, 2.11, 2.33, 2.68 ], #1.21,
    'ASR': [3.9, 5.5, 8.2, 10.8, 12.1 ], # 4.0,
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



# Fit a smooth curve to the data using make_interp_spline
CSD1 = np.array(line1['CSD'])
ASR1 = np.array(line1['ASR'])
CSD1_smooth = np.linspace(CSD1.min(), CSD1.max(), 200)
spline1 = make_interp_spline(CSD1, ASR1, k=2 if len(CSD1) == 3 else 3)
ASR1_smooth = spline1(CSD1_smooth)

CSD1, ASR1 = line1['CSD'], line1['ASR']
CSD2, ASR2 = line2['CSD'], line2['ASR']
color1, color2 = line1['color'], line2['color']
name1, name2 = line1['name'], line2['name']

CSD1_smooth, ASR1_smooth = smooth_curve(line1['CSD'], line1['ASR'])
CSD2_smooth, ASR2_smooth = smooth_curve(line2['CSD'], line2['ASR'])

plt.figure(figsize=(8, 6))

# Plot line1
plt.plot(CSD1_smooth, ASR1_smooth, marker='o', linestyle='-', color=color1, label=name1)
# for x, y in zip(CSD1, ASR1):
#     plt.text(x, y, None, fontsize=9, ha='right', va='bottom', color=color1) # f'l={label}',

# Plot line2
plt.plot(CSD2_smooth, ASR2_smooth, marker='s', linestyle='--', color=color2, label=name2)
# for x, y in zip(CSD2_smooth, ASR2_smooth):
#     plt.text(x, y, f'{x},{y}', fontsize=9, ha='left', va='top', color=color2) #f'l={label}',
    
# Prepare data for caption attack lines
CSD3, ASR3 = line3['CSD'], line3['ASR']
CSD4, ASR4 = line4['CSD'], line4['ASR']
color3, color4 = line3['color'], line4['color']
name3, name4 = line3['name'], line4['name']


CSD3_smooth, ASR3_smooth = smooth_curve(line3['CSD'], line3['ASR'])
CSD4_smooth, ASR4_smooth = smooth_curve(line4['CSD'], line4['ASR'])

# Plot line3 (caption attack, blue, dotted)
plt.plot(CSD3_smooth, ASR3_smooth, marker='o', linestyle='-', color=color3, label=name3)
# for x, y in zip(CSD3, ASR3):
#     plt.text(x, y, f'{x},{y}', fontsize=9, ha='right', va='top', color=color3)

# Plot line4 (caption attack, red, dash-dot)
plt.plot(CSD4_smooth, ASR4_smooth, marker='s', linestyle='--', color=color4, label=name4)
# for x, y in zip(CSD4_smooth, ASR4_smooth):
#     plt.text(x, y, f'{x},{y}', fontsize=9, ha='left', va='bottom', color=color4)

plt.xlabel('CSD')
plt.ylabel('ASR (%)')
# plt.title('ASR vs CSD for targeted Vision Attacks')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('ASR_CSD_targeted_attack_on_Modelnet40_smooth.jpg', format='jpg')
plt.show()
