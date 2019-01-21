import os
import numpy as np
import argparse
import random
import tensorflow as tf

from data_generator import DataGenerator, DataGeneratorCUB

from time import time


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num", help="Choose the number of GPU.", type=str, default="0", choices=["0", "1"])
parser.add_argument('-t', '--test', action='store_true', default=False, help='set for test, otherwise train')
parser.add_argument('--backbone', type=str, default="0", help="Choose the backbone network.")
args = parser.parse_args()

if args.backbone == "0":
	print("Loading backbone: Conv_4 (origin)...")
	from maml import MAML ## default backbone: Conv_4
else:
	print("Loading backbone: Resnet10...")
	from maml_resnet10 import MAML


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num


def train(model, saver, sess, save_weights_dirpath=None, max_iters=60000):
	"""

	:param model:
	:param saver:
	:param sess:
	:return:
	"""
	# write graph to tensorboard
	# tb = tf.summary.FileWriter(os.path.join('logs', 'mini'), sess.graph)
	prelosses, postlosses, preaccs, postaccs = [], [], [], []
	best_acc = 0
	st = time()
	current_st = time()
	# train for meta_iteartion epoches
	for iteration in range(max_iters): ## 600000
		# this is the main op
		ops = [model.meta_op]

		# add summary and print op
		if iteration % 200 == 0:
			ops.extend([model.summ_op,
			            model.query_losses[0], model.query_losses[-1],
			            model.query_accs[0], model.query_accs[-1]])

		# run all ops
		result = sess.run(ops)

		# summary
		if iteration % 200 == 0:
			# summ_op
			# tb.add_summary(result[1], iteration)
			# query_losses[0]
			prelosses.append(result[2])
			# query_losses[-1]
			postlosses.append(result[3])
			# query_accs[0]
			preaccs.append(result[4])
			# query_accs[-1]
			postaccs.append(result[5])

			print(iteration, '\tloss:', np.mean(prelosses), '=>', np.mean(postlosses),
			      '\t\tacc:', np.mean(preaccs), '=>', np.mean(postaccs))
			prelosses, postlosses, preaccs, postaccs = [], [], [], []

		# evaluation
		if iteration % 2000 == 0:

			# DO NOT write as a = b = [], in that case a=b
			# DO NOT use train variable as we have train func already.
			acc1s, acc2s = [], []
			# sample 20 times to get more accurate statistics.
			for _ in range(200):
				acc1, acc2 = sess.run([model.test_query_accs[0],
				                        model.test_query_accs[-1]])
				acc1s.append(acc1)
				acc2s.append(acc2)

			current_et = time()
			elapsed_time = current_et-current_st
			current_st = current_et
			acc = np.mean(acc2s)
			print('>>>>\t\tValidation accs: ', np.mean(acc1s), acc, 'best:', best_acc, '\t\t<<<<')
			print("===========  Elapsed time: {:.2f}  ===========".format(elapsed_time))

			# if acc - best_acc > 0.05 or acc > 0.4:
			if acc - best_acc > 0.007 or acc > 0.4:
				saver.save(sess, os.path.join(save_weights_dirpath, 'mini.mdl'))
				best_acc = acc
				print('saved into ckpt:', acc)
	total_elapsed_time = time()-st

	print("Total elapsed time: {}".format(total_elapsed_time))

def test(model, sess, save_result2dfile=None):

	np.random.seed(1)
	random.seed(1)

	# repeat test accuracy for 600 times
	test_accs = []
	for i in range(600):
		if i % 100 == 1:
			print(i)
		# extend return None!!!
		ops = [model.test_support_acc]
		ops.extend(model.test_query_accs)
		result = sess.run(ops)
		test_accs.append(result)

	# [600, K+1]
	test_accs = np.array(test_accs)
	# [K+1]
	means = np.mean(test_accs, 0)
	stds = np.std(test_accs, 0)
	ci95 = 1.96 * stds / np.sqrt(600)
	
	dirname = "/".join(save_result2dfile.split("/")[:-1])
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	message = ''
	message += '[support_t0, query_t0 - \t\t\tK] \n'
	message += 'mean: {} \n'.format(means)
	message += 'stds: {} \n'.format(stds)
	message += 'ci95: {}'.format(ci95)
	with open(save_result2dfile, "w") as file:
		file.write(message)

	print('[support_t0, query_t0 - \t\t\tK] ')
	print('mean:', means)
	print('stds:', stds)
	print('ci95:', ci95)



def main(shot=1, workers=1, 
	meta_lr=1e-3, train_lr=1e-2,
	max_iters=120000,
	data_augment=False, 
	save_result2dfile=None, 
	save_weights_dirpath=None, 
	dataset="CUB"):
	training = not args.test
	kshot = shot 
	kquery = 15
	nway = 5
	meta_batchsz = workers#4
	K = 5 #  train update steps


	# kshot + kquery images per category, nway categories, meta_batchsz tasks.
	if dataset == "CUB":
		db = DataGeneratorCUB(nway, kshot, kquery, meta_batchsz, 20000)	
	else:
		db = DataGenerator(nway, kshot, kquery, meta_batchsz, 200000)
	

	if  training:  # only construct training model if needed
		# get the tensor
		# image_tensor: [4, 80, 84*84*3]
		# label_tensor: [4, 80, 5]
		image_tensor, label_tensor = db.make_data_tensor(training=True, augment=data_augment)
		# print(image_tensor.shape)
		# NOTICE: the image order in 80 images should like this now:
		# [label2, label1, label3, label0, label4, and then repeat by 15 times, namely one task]
		# support_x : [4, 1*5, 84*84*3]
		# query_x   : [4, 15*5, 84*84*3]
		# support_y : [4, 5, 5]
		# query_y   : [4, 15*5, 5]
		support_x = tf.slice(image_tensor, [0, 0, 0, 0, 0], [-1,  nway *  kshot, -1, -1, -1], name='support_x')
		query_x = tf.slice(image_tensor, [0,  nway *  kshot, 0, 0, 0], [-1, -1, -1, -1, -1], name='query_x')
		support_y = tf.slice(label_tensor, [0, 0, 0], [-1,  nway *  kshot, -1], name='support_y')
		query_y = tf.slice(label_tensor, [0,  nway *  kshot, 0], [-1, -1, -1], name='query_y')

	# construct test tensors.
	image_tensor, label_tensor = db.make_data_tensor(training=False, augment=False)
	support_x_test = tf.slice(image_tensor, [0, 0, 0, 0, 0], [-1,  nway *  kshot, -1, -1, -1], name='support_x_test')
	query_x_test = tf.slice(image_tensor, [0,  nway *  kshot, 0, 0, 0], [-1, -1, -1, -1, -1],  name='query_x_test')
	support_y_test = tf.slice(label_tensor, [0, 0, 0], [-1,  nway *  kshot, -1],  name='support_y_test')
	query_y_test = tf.slice(label_tensor, [0,  nway *  kshot, 0], [-1, -1, -1],  name='query_y_test')


	# 1. construct MAML model
	if dataset == "CUB":
		model = MAML(128, 3, 5, meta_lr=meta_lr, train_lr=train_lr)
	else:
		model = MAML(84, 3, 5, meta_lr=meta_lr, train_lr=train_lr)
	

	# construct metatrain_ and metaval_
	if  training:
		model.build(support_x, support_y, query_x, query_y, K, meta_batchsz, mode='train')
		model.build(support_x_test, support_y_test, query_x_test, query_y_test, K, meta_batchsz, mode='eval')
	else:
		model.build(support_x_test, support_y_test, query_x_test, query_y_test, K + 5, meta_batchsz, mode='test')
	model.summ_op = tf.summary.merge_all()

	all_vars = filter(lambda x: 'meta_optim' not in x.name, tf.trainable_variables())
	for p in all_vars:
		print(p)


	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)
	# tf.global_variables() to save moving_mean and moving variance of batch norm
	# tf.trainable_variables()  NOT include moving_mean and moving_variance.
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

	# initialize, under interative session
	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

	if os.path.exists(os.path.join(save_weights_dirpath, 'checkpoint')):
		# always load ckpt both train and test.
		model_file = tf.train.latest_checkpoint(save_weights_dirpath)
		print("Restoring model weights from ", model_file)
		saver.restore(sess, model_file)


	if training:
		train(model, saver, sess, save_weights_dirpath=save_weights_dirpath, max_iters=max_iters)
	else:
		test(model, sess, save_result2dfile=save_result2dfile)


if __name__ == "__main__":
	# main(save_result2dfile="./results/miniImagenet/5-way_5-shot_50-shot.txt",
	# 	save_weights_dirpath='ckpt_CUB_5_shot')

	# main(save_result2dfile="./results/CUB/5-way_5-shot_1-work_5-shot.txt",
	# 	save_weights_dirpath='ckpt_CUB_5_shot_1_workers')

	
	### TEST
	# main(shot=1, workers=4,
	#  	save_result2dfile="./results/miniImagenet/5-way_5-shot_1-shot.txt",
	# 	save_weights_dirpath='ckpt_5-shot', dataset="miniImagenet")
	


	# main(shot=1, workers=1,
	# 	save_result2dfile="./results/miniImagenet/5-way_1-shot_1-work_1-shot.txt",
	# 	save_weights_dirpath='ckpt_1_shot_1_workers', dataset="miniImagenet")

	# main(shot=5, workers=1,
	# 	save_result2dfile="./results/CUB/5-way_5-shot_1-work_5-shot.txt",
	# 	save_weights_dirpath='ckpt_CUB_5_shot_1_workers', dataset="CUB")




	# main(shot=1, workers=1,
	# 	max_iters=120000,
	# 	data_augment=True,
	# 	save_result2dfile="./results/CUB_new/5-way_1-shot_1-work_1-shot_DA_000.txt",
	# 	save_weights_dirpath='./weights/ckpt_new_CUB_1_shot_DA_000', dataset="CUB")

	main(shot=1, workers=8,
		# meta_lr=3*1e-4, train_lr=3*1e-3,
		max_iters=60000,
		data_augment=False,
		save_result2dfile="./results/miniImagenet_new/5-way_1-shot_8-work_1-shot.txt",
		save_weights_dirpath='./weights/ckpt_new_miniImagenet_1_shot_8_work', dataset="miniImagenet")


