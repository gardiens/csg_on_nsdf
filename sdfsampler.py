import torch
import numpy as np

class DataGenerator:
	def __init__(self, low_domain, high_domain, device='cpu'):
		self.HI = high_domain
		self.LO = low_domain
		self.device = device
		self.dim = len(high_domain)
	
	def and_input_eval(self, pts, input_func):
		return pts, input_func(pts)

	def stratified_data_generator(self, model, input_func, NUM_PTS=1000):
		rootN = round(NUM_PTS**(1/self.dim))
		
		strat = [make_stratified_space(rootN, self.LO[i], self.HI[i], device=self.device) for i in range(self.dim)]
		lins, Dgs = zip(*strat)
		pts = make_stratified_samples(lins, Dgs, device=self.device)

		indices = torch.randperm(pts.shape[0], device=pts.device)[:NUM_PTS]
		return pts[indices]
	
	def importance_sampling_boundary(self, model, input_func, NUM_PTS=1000, gaussian_sigma=1e-2, gen_mult=100):
		gen_pts = NUM_PTS * gen_mult
		gen = lambda device: self.stratified_data_generator(model, input_func, NUM_PTS=gen_pts)

		def weight(pts, device):
			return gaussian(input_func(pts), sigma=gaussian_sigma)
			
		return importance_sample_from_function(NUM_PTS, gen, weight, device=self.device)

	def importance_and_stratified(self, model, input_func, NUM_PTS=1000, gaussian_sigma=1e-2, percent_ambient=0.2, gen_mult=100):
		amb_num = round(percent_ambient * NUM_PTS)
		imp_num = NUM_PTS - amb_num

		imp_pts = self.importance_sampling_boundary(model, input_func, NUM_PTS=imp_num, gaussian_sigma=gaussian_sigma, gen_mult=gen_mult)
		if amb_num > 0:
			amb_pts = self.stratified_data_generator(model, input_func, NUM_PTS=amb_num)
		else:
			return imp_pts

		return torch.cat((amb_pts, imp_pts), dim=0)


############################
# Sampling Util. Functions #
############################


"""
General purpose importance sampling function, which will generate N points by iterativly
calling point_generator and using rejection sampling with probabilities generatored by calling
weight_function at the generated points.
"""
def importance_sample_from_function(N, point_generator, weight_function, device='cpu'):
	cur_pts, cur_n = [], 0	
	i = 0
	while cur_n < N:
		i += 1
		pts = point_generator(device=device)
		wghts = weight_function(pts, device=device)
		thresholds = torch.rand((pts.shape[0],1), device=device)

		pts = pts[(thresholds < wghts)[:,0], :]
		cur_pts.append(pts)
		cur_n += pts.shape[0]

	if (i > 50):
		# this warning indicates that either the number of points being generated by point_generator
		# is too low, or the function being importance sampled from is incredbily sparse/sharp
		print(f'Warning!! Importance sampling iterated {i} times.')

	total_pts = torch.cat(cur_pts, dim=0)

	# shuffle before taking top N to avoid bias
	indices = torch.randperm(total_pts.shape[0], device=device)[:N]
	return total_pts[indices]


def gaussian(vals, sigma=1, mu=0):
	return 1/np.sqrt(2*np.pi*sigma) * torch.exp(-(vals-mu)**2/(2*sigma**2))


# generate linspace for stratified sampling along 1 dimension
def make_stratified_space(N, lower, upper, device='cpu'):
	Dg = (upper - lower)/N
	return torch.linspace(lower, upper - Dg, N, device=device), Dg


# given a list of stratified linspaces and grid spacing, sample from the product of those spaces
def make_stratified_samples(lins, Dgs, device='cpu'):
	base_grids = torch.meshgrid(*lins, indexing='ij')        
	pts = []
	for dim in range(len(base_grids)):
	    rand_shift = Dgs[dim] * torch.rand(*base_grids[dim].shape, device=device)
	    pts.append((base_grids[dim] + rand_shift).flatten())
	X = torch.vstack(pts).T
	return X
