# coding: utf-8
# from data_provider import *
from C3D_model import *
import torchvision
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os 
from torch import save, load
import pickle
import time
import numpy as np
import PIL.Image as Image
import skimage.io as io
from skimage.transform import resize
import h5py
from PIL import Image

def feature_extractor():
	#trainloader = Train_Data_Loader( VIDEO_DIR, resize_w=128, resize_h=171, crop_w = 112, crop_h = 112, nb_frames=16)
	net = C3D(487)
	print('net', net)
	## Loading pretrained model from sports and finetune the last layer
	net.load_state_dict(torch.load('/data1/miayuan/pretrained_models/c3d.pickle'))
	if RUN_GPU : 
		net.cuda(0)
		net.eval()
		print('net', net)
	feature_dim = 4096 if EXTRACTED_LAYER != 5 else 8192

	# read video list from the folder 
	#video_list = os.listdir(VIDEO_DIR)

	# read video list from the txt list
	video_list_file = args.video_list_file
	video_list = open(video_list_file).readlines()
	video_list = [item.strip() for item in video_list]
	print('video_list', video_list)

	gpu_id = args.gpu_id

	if not os.path.isdir(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	f = h5py.File(os.path.join(OUTPUT_DIR, OUTPUT_NAME), 'w')

	# current location
	temp_path = os.path.join(os.getcwd(), 'temp')
	if not os.path.exists(temp_path):
		os.mkdir(temp_path)

	error_fid = open('error.txt', 'w')
	for video_name in video_list: 
		video_path = os.path.join(VIDEO_DIR, video_name)
		print('video_path', video_path)
		frame_path = os.path.join(temp_path, video_name)
		if not os.path.exists(frame_path):
			os.mkdir(frame_path)


		print('Extracting video frames ...')
		# using ffmpeg to extract video frames into a temporary folder
		# example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg
		os.system('ffmpeg -i ' + video_path + ' -q:v 2 -f image2 ' + frame_path + '/image_%5d.jpg')


		print('Extracting features ...')
		total_frames = len(os.listdir(frame_path))
		if total_frames == 0:
			error_fid.write(video_name+'\n')
			print('Fail to extract frames for video: %s'%video_name)
			continue

		valid_frames = total_frames / nb_frames * nb_frames
		n_feat = valid_frames / nb_frames
		n_batch = n_feat / BATCH_SIZE 
		if n_feat - n_batch*BATCH_SIZE > 0:
			n_batch = n_batch + 1
		print('n_frames: %d; n_feat: %d; n_batch: %d'%(total_frames, n_feat, n_batch))
		
		#print 'Total frames: %d'%total_frames 
		#print 'Total validated frames: %d'%valid_frames
		#print 'NB features: %d' %(valid_frames/nb_frames)
		index_w = np.random.randint(resize_w - crop_w) ## crop
		index_h = np.random.randint(resize_h - crop_h) ## crop

		features = []

		for i in range(n_batch-1):
			input_blobs = []
			for j in range(BATCH_SIZE):
				clip = []
				clip = np.array([resize(io.imread(os.path.join(frame_path, 'image_{:05d}.jpg'.format(k))), output_shape=(resize_w, resize_h), preserve_range=True) for k in range((i*BATCH_SIZE+j) * nb_frames+1, min((i*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])
				#print('clip_shape', clip.shape)
				clip = clip[:, index_w: index_w+ crop_w, index_h: index_h+ crop_h, :]
				#print('clip_shape',clip.shape)
				#print('range', range((i*BATCH_SIZE+j) * nb_frames+1, min((i*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1)))	
				input_blobs.append(clip)
			input_blobs = np.array(input_blobs, dtype='float32')
			#print('input_blobs_shape', input_blobs.shape)
			input_blobs = torch.from_numpy(np.float32(input_blobs.transpose(0, 4, 1, 2, 3)))
			input_blobs = Variable(input_blobs).cuda() if RUN_GPU else Variable(input_blobs)
			_, batch_output = net(input_blobs, EXTRACTED_LAYER)	
			batch_feature  = (batch_output.data).cpu()
			features.append(batch_feature)

		# The last batch
		input_blobs = []
		for j in range(n_feat-(n_batch-1)*BATCH_SIZE):
			clip = []
			clip = np.array([resize(io.imread(os.path.join(frame_path, 'image_{:05d}.jpg'.format(k))), output_shape=(resize_w, resize_h), preserve_range=True) for k in range(((n_batch-1)*BATCH_SIZE+j) * nb_frames+1, min(((n_batch-1)*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])

			clip = clip[:, index_w: index_w+ crop_w, index_h: index_h+ crop_h, :]
			#print('range', range(((n_batch-1)*BATCH_SIZE+j) * nb_frames+1, min(((n_batch-1)*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1)))
			input_blobs.append(clip)
		input_blobs = np.array(input_blobs, dtype='float32')
		#print('input_blobs_shape', input_blobs.shape)
		input_blobs = torch.from_numpy(np.float32(input_blobs.transpose(0, 4, 1, 2, 3)))
		input_blobs = Variable(input_blobs).cuda() if RUN_GPU else Variable(input_blobs)
		_, batch_output = net(input_blobs, EXTRACTED_LAYER)
		batch_feature  = (batch_output.data).cpu()
		features.append(batch_feature)

		features = torch.cat(features, 0)
		features = features.numpy()
		print('features', features)
		fgroup = f.create_group(video_name)
		fgroup.create_dataset('c3d_features', data=features)
		fgroup.create_dataset('total_frames', data=np.array(total_frames))
		fgroup.create_dataset('valid_frames', data=np.array(valid_frames))

		print '%s has been processed...'%video_name


		# clear temp frame folders
		try: 
			os.system('rm -rf ' + frame_path)
		except: 
			pass


#		for i in range(valid_frames/nb_frames) :   
#			clip = np.array([resize(io.imread(os.path.join(video_path, 'image_{:05d}.jpg'.format(j))), output_shape=(resize_w, resize_h), preserve_range=True) for j in range(i * nb_frames+1, (i+1) * nb_frames+1)])
#			#clip = np.array([resize(video.get_data(j), output_shape=(resize_w, resize_h), preserve_range=True) for j in range(i * nb_frames, (i+1) * nb_frames)])
#			clip = clip[:, index_w: index_w+ crop_w, index_h: index_h+ crop_h, :]
#			clip = torch.from_numpy(np.float32(clip.transpose(3, 0, 1, 2)))
#			clip = Variable(clip).cuda() if RUN_GPU else Variable(clip)			
#			clip = clip.resize(1, 3, nb_frames, crop_w, crop_h)
#			#print('clip', clip)
#			_, clip_output = net(clip, EXTRACTED_LAYER)
#			#print('clip_output', clip_output)  
#			clip_feature  = (clip_output.data).cpu()  
#			features.append(clip_feature)
#			#features[i] = np.array(clip_feature)
#		features = torch.cat(features, 0)
#		features = features.numpy()
#		print('features', features)       
#		fgroup = f.create_group(video_name)
#		fgroup.create_dataset('c3d_features', data=features)
#		fgroup.create_dataset('total_frames', data=np.array(total_frames))
#		fgroup.create_dataset('valid_frames', data=np.array(valid_frames))
#	
#		#with open(os.path.join(OUTPUT_DIR, video_name[:-4]), 'wb') as f :
#		#	pickle.dump( features, f )
#		print '%s has been processed...'%video_name


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	print '******--------- Extract C3D features ------*******'
	parser.add_argument('-o', '--OUTPUT_DIR', dest='OUTPUT_DIR', type=str, default='./output_frm/', help='Output file name')
	parser.add_argument('-l', '--EXTRACTED_LAYER', dest='EXTRACTED_LAYER', type=int, choices=[5, 6, 7], default=6, help='Feature extractor layer')
	parser.add_argument('-i', '--VIDEO_DIR', dest='VIDEO_DIR', type = str, help='Input Video directory')
	parser.add_argument('-gpu', '--gpu', dest='GPU', action = 'store_true', help='Run GPU?')
	parser.add_argument('--OUTPUT_NAME', default='c3d_features.hdf5', help='The output name of the hdf5 features')
	parser.add_argument('-b', '--BATCH_SIZE', default=30, help='the batch size')
	parser.add_argument('-id', '--gpu_id', default=0, type=int)
	parser.add_argument('-p', '--video_list_file', type=str, help='the video name list')

	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	print 'parsed parameters:'

	OUTPUT_DIR = params['OUTPUT_DIR']
	EXTRACTED_LAYER = params['EXTRACTED_LAYER']
	VIDEO_DIR = params['VIDEO_DIR']
	RUN_GPU = params['GPU']
	OUTPUT_NAME = params['OUTPUT_NAME']
	BATCH_SIZE = params['BATCH_SIZE']
	crop_w = 112
	resize_w = 128
	crop_h = 112
	resize_h = 171
	nb_frames = 16	
	feature_extractor()


