import matplotlib.pyplot as plt
import numpy as np

# 读取文件内容
file_path = './result_stain.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# 提取 Relative Noisy Scores 和 Relative Recover Scores
relative_noisy_scores = []
relative_recover_scores = []

for line in lines:
    if line.startswith('Relative Noisy Scores:'):
        scores = line.split(':')[1].strip().split(',')
        relative_noisy_scores = [float(score.strip()) for score in scores if score.strip()]
    elif line.startswith('Relative Recover Scores:'):
        scores = line.split(':')[1].strip().split(',')
        relative_recover_scores = [float(score.strip()) for score in scores if score.strip()]

# 绘制直方图
bins = np.arange(0, 1.05, 0.05)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)

# 绘制 Relative Noisy Scores 的直方图
ax1.hist(relative_noisy_scores, bins=bins, alpha=0.7, label='Noisy RMS', edgecolor='black', color='blue')
ax1.set_title('Histogram of Noisy RMS')
ax1.set_ylabel('Frequency')
ax1.legend(loc='upper right')
ax1.grid(True)

# 绘制 Relative Recover Scores 的直方图
ax2.hist(relative_recover_scores, bins=bins, alpha=0.7, label='Recover RMS', edgecolor='black', color='red')
ax2.set_title('Histogram of Recover RMS')
ax2.set_xlabel('Score')
ax2.set_ylabel('Frequency')
ax2.legend(loc='upper right')
ax2.grid(True)

# 调整布局并保存图表为 res.png
plt.tight_layout()
plt.savefig('./res_stain.png')
plt.show()
