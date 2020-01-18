
import tensorflow as tf

def wasserstein_d_loss(target,output):
	d_x = - output[0][0]
	d_g_z = output[0][0]
	return (d_x * target[0][0]) + (d_g_z * target[0][1])

def wasserstein_g_loss(target,output):
	return - output[0][0]