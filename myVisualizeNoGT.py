import matplotlib.pyplot as plt
import numpy as np
import time

predicted_result_dir = './result/'
gradient_color = True

def plot_route(out, c_out='r'):
	x_idx = 3
	y_idx = 5

	x = [v for v in out[:, x_idx]]
	y = [v for v in out[:, y_idx]]
	plt.plot(x, y, color=c_out, label='DeepVO')
	plt.gca().set_aspect('equal', adjustable='datalim')


# Load in GT and predicted pose
# video_list = ['00', '02', '08', '09']
# video_list += ['01', '04', '05', '06', '07', '10']
video_list = ['campus1','campus2','ntu', 'room'] # ['00', '01', '02']


for video in video_list:
	print('='*50)
	print('Video {}'.format(video))

	pose_result_path = '{}out_{}.txt'.format(predicted_result_dir, video)
	with open(pose_result_path) as f_out:
		out = [l.split('\n')[0] for l in f_out.readlines()]
		for i, line in enumerate(out):
			out[i] = [float(v) for v in line.split(',')]
		out = np.array(out)


	if gradient_color:
		# plot gradient color
		step = 200
		plt.clf()
		plt.scatter([out[0][3]], [out[0][5]], label='sequence start', marker='s', color='k')
		for st in range(0, len(out), step):
			end = st + step
			g = max(0.2, st/len(out))
			c_out = (1, g, 0)
			plot_route(out[st:end], c_out)
			if st == 0:
				plt.legend()
			plt.title('Video {}'.format(video))
			save_name = '{}route_{}_gradient.png'.format(predicted_result_dir, video)
		plt.savefig(save_name)
