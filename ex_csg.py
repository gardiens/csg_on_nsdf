import torch

from sdfpy import circle_sdf, square_sdf, union
from sdfsampler import DataGenerator
from sdfpytorch import DeepSDF, fit
from nsdf_csg_losses import csg_combined_loss
from clearml import Task

#+------------------------------------------------------------------------+#
#|                        ~~~~~~HYPERPARAMS ~~~~~~                        |#
batch_size 			= 50000
epochs 				= 291000 #! Changed w.r.t to intial  
report 				= 1000 
save   				= epochs
step_size 			= 1e-4
samp_sigma 			= 1e-2
samp_amb_percent 	= 0.5
reuse_data_epochs 	= 20
from argparse import ArgumentParser, Namespace
import sys
parser=ArgumentParser(description="Training script parameters")
parser.add_argument("--name", type=str, default = "circle_square_union")
parser.add_argument('--lambda_loss', nargs='+', type=int,default=(15,1,1))

args = parser.parse_args(sys.argv[1:])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#+------------------------------------------------------------------------+#
print("the name is ",args.name)
task = Task.init(project_name='csvg', task_name=args.name)

hyper_params_dict = {"epochs": epochs, "step_size": step_size, "report": report, "save": save, "device": device, "reuse_data_epochs": reuse_data_epochs}
samp_hyper_params_dict = {"NUM_PTS": batch_size, "gaussian_sigma": samp_sigma, "percent_ambient": samp_amb_percent}

# 1. Set up input: Pseudo-SDF for desired CSG problem
circle = lambda pts: circle_sdf(pts, torch.tensor([[0.5,0.5]], device=device), 1.2, device=device) 
square = lambda pts: square_sdf(pts, torch.tensor([[-0.5,-0.5]], device=device), torch.tensor([2,2], device=device), device=device) 
union_approx = union(circle, square)

# 2. Set up data generation function, for creating sample points during training
DG = DataGenerator([-2, -2], [2, 2], device)
data_gen = lambda model: DG.and_input_eval(DG.importance_and_stratified(model, union_approx, **samp_hyper_params_dict), union_approx)

# 3. Intialize the neural network
model = DeepSDF(8, 128, n_dim=2).to(device)

# 4. Define the loss function
print("lambda_loss",args.lambda_loss)
loss_fn = csg_combined_loss(300,args.lambda_loss, dim=2)

# 5. Train! 
test_name = args.name
fit(model, loss_fn, data_gen, test_name, **hyper_params_dict)
 
# 6. After training, visualize the results by running `python3 viz_ex.py circle_square_union` in the command line!