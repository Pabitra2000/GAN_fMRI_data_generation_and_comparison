# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:51:22 2022

@author: Sveekruth

Description: E9205 (MLSP) Project. Has two objectives:
    1. Perform Independent Component Analysis (ICA, d = 10) on a subset (N = 
       339 healthy subjects, D = 360 parcels) of Human Connectome Project (HCP) 
       Resting State Functional Magnetic Resonance Imaging (fMRI) Data:
       (https://osf.io/bqp7m/)
       Do the ICs match those described in the literature?
       (https://doi.org/10.1073/pnas.0601417103)
    2. Create and train a GAN to generate a deepfake version of the above
       data. Does it resemble the real data in D (say, with a heatmap of TxD 
       for a single subject or more)?. Run ICA again. Do the ICs match (order
       not too important) those above?
If time permits, explore additional similarity metrics like RSA between 
parcel functional connectivity etc.                                                                 
"""

# Importing libraries:
import os, time, requests, tarfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk')
from nilearn import plotting, datasets # Necessary for visualization

# Setting directory paths:
pdir = 'E:/IISc/III/E9205/Project/'
os.chdir(pdir)
HCP_DIR = f'{pdir}HCP_DIR/'
if not os.path.isdir(HCP_DIR):
    os.mkdir(HCP_DIR) # Initializing nested directory
data_dir = f'{HCP_DIR}hcp_rest/' #hcp_task/

# Basic parameters:
N_SUBJECTS = 339
N_PARCELS = 360
TR = 0.72  # Time resolution, in sec
HEMIS = ["Right", "Left"] # The parcels are matched across hemispheres with the same order
# Each experiment was repeated multiple times in each subject
N_RUNS_REST = 4
# N_RUNS_TASK = 2
# Time series data are organized by experiment, with each experiment
# having an LR and RL (phase-encode direction) acquistion
BOLD_NAMES = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", \
              "rfMRI_REST2_RL", "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR", \
              "tfMRI_WM_RL", "tfMRI_WM_LR", "tfMRI_EMOTION_RL", \
              "tfMRI_EMOTION_LR", "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR", \
              "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR", "tfMRI_RELATIONAL_RL", \
              "tfMRI_RELATIONAL_LR", "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"]
subjects = range(N_SUBJECTS) # Limit to subset if desired

# Atlas for Visualizations:
with np.load('atlas.npz') as dobj:
    atlas = dict(**dobj) # Has keys "coords", "labels_L", and "labels_R", for use with fsaverage5 (10240 nodes)
# https://figshare.com/articles/HCP-MMP1_0_projected_on_fsaverage/3498446)

# Region information:
regions = np.load(f'{data_dir}regions.npy').T # regions.np created on extracting any one of the above datasets
region_info = dict(name=regions[0].tolist(), network=regions[1], myelin=\
                   regions[2].astype(float))
# regions[0, :][regions[1, :] == "Somatomotor"] # To see which regions are present in a given network
# len(region_info["name"]) # 360 regions, 180 R, 180 L

# Load the data tensor:
if not os.path.exists(f'{data_dir}X.npz'):
    # Downloading required TAR files from OSF.io (if not already present):
    fnames = ["hcp_rest.tgz", "atlas.npz"] # "hcp_task.tgz", "hcp_covariates.tgz",
    urls = ["https://osf.io/bqp7m/download",
            "https://osf.io/j5kuc/download"] # "https://osf.io/s4h8j/download", "https://osf.io/x5p4g/download",
    for fname, url in zip(fnames, urls):
        if not os.path.isfile(fname):
            try:
                r = requests.get(url)
            except requests.ConnectionError:
                print("!!! Failed to download data !!!")
            else:
                if r.status_code != requests.codes.ok:
                    print("!!! Failed to download data !!!")
                else:
                    print(f"Downloading {fname}...")
                    with open(fname, "wb") as fid:
                      fid.write(r.content)
                    print(f"Download {fname} completed!")
    
    # Extracting data to HCP_DIR:
    fnames = ["hcp_rest"] # "hcp_covariates", "hcp_task"
    for fname in fnames:
      path_name = os.path.join(HCP_DIR, fname) # open file
      if not os.path.exists(path_name):
        print(f"Extracting {fname}.tgz...")
        with tarfile.open(f"{fname}.tgz") as fzip:
          fzip.extractall(HCP_DIR)
      else:
        print(f"File {fname}.tgz has already been extracted.")    
        
    """
    # Inspecting the data:
    subject = 0
    bold_run = 1
    bold_path = os.path.join(data_dir, "subjects", str(subject), "timeseries")
    bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
    ts = np.load(os.path.join(bold_path, bold_file)) # D (parcels) x T (timepoints)
    
    ts.shape # 360 x 1200. Since TR = 0.72s, total of 864s.
    ts -= ts.mean(axis=1, keepdims=True)
    sns.heatmap(ts)
    """
    
    # Helper Functions:
    def get_image_ids(name):
      """
      Get the 1-based image indices for runs in a given experiment.
      Args:
        name (str) : Name of experiment ("rest" or name of task) to load
      Returns:
        run_ids (list of int) : Numeric ID for experiment image files
      """
      run_ids = [i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code]
      if not run_ids:
        raise ValueError(f"Found no data for '{name}'")
      return run_ids
    def load_single_timeseries(subject, bold_run, data_dir, remove_mean=True, remove_var=False):
      """
      Load timeseries data for a single subject and single run.
      Args:
        subject (int): 0-based subject ID to load
        bold_run (int): 1-based run index, across all tasks
        dir (str) : data directory
        remove_mean (bool): If True, subtract the parcel-wise mean
      Returns
        ts (n_parcel x n_timepoint array): Array of BOLD data values
      """
      bold_path = os.path.join(data_dir, "subjects", str(subject), "timeseries")
      bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
      ts = np.load(os.path.join(bold_path, bold_file))
      if remove_mean:
        ts -= ts.mean(axis=1, keepdims=True)
      if remove_var:
        ts /= ts.std(axis=1, keepdims=True)
      return ts
    def load_timeseries(subject, name, data_dir, runs=None, concat=True, remove_mean=True, \
                        remove_var = False):
      """
      Load timeseries data for a single subject.
      Args:
        subject (int): 0-based subject ID to load
        name (str) : Name of experiment ("rest" or name of task) to load
        dir (str) : data directory
        run (None or int or list of ints): 0-based run(s) of the task to load,
          or None to load all runs.
        concat (bool) : If True, concatenate multiple runs in time
        remove_mean (bool) : If True, subtract the parcel-wise mean
      Returns
        ts (n_parcel x n_tp array): Array of BOLD data values
      """
      # Get the list relative 0-based index of runs to use
      if runs is None:
        runs = range(N_RUNS_REST) if name == "rest" else range(N_RUNS_TASK)
      elif isinstance(runs, int):
        runs = [runs]
      # Get the first (1-based) run id for this experiment
      offset = get_image_ids(name)[0]
      # Load each run's data
      bold_data = [load_single_timeseries(subject, offset + run, data_dir, remove_mean, remove_var) \
                   for run in runs]
      # Optionally concatenate in time
      if concat:
        bold_data = np.concatenate(bold_data, axis=-1)
      return bold_data
    
    """
    # More inspection:
    foo = load_timeseries(0, "rest", data_dir, remove_mean=True, remove_var=True).T
    foo.shape # Around 1 hour of total rs fMRI per subject
    sns.heatmap(foo)
    """
    
    # Resting State fMRI Data Tensor:
    # All N = 339 subjects data included. All 4 rs fmri runs per subject normalized then concatenated along the T axis.
    rs_fmri_tensor = np.array([load_timeseries(j, "rest", data_dir, runs=None, \
                               concat=True, remove_mean=True, remove_var=True).T \
                               for j in subjects])
    rs_fmri_tensor.shape # NxTxD = 339x4800x360. Centered and scaled.
    np.savez_compressed(f'{data_dir}X.npz', X = rs_fmri_tensor)
else:
    with np.load(f'{data_dir}X.npz') as f:
        # f.files # Only stored array is X
        X = f['X']
        
# Downsampling (IMPORTANT: Further normalization has NOT been performed. This may affect results downstream, though FastICA and Pearson's r are known to perform standardizations):
seed = 0
rng = np.random.default_rng(seed)
index = rng.choice(range(N_SUBJECTS), size=50, replace=False)
# X = X[index.tolist(), np.arange(0, 1200, 3)[:N_PARCELS].tolist(), :] # It refuses to work in a single step :/
X = X[index, :, :] # depth reduction
X = X[:, np.arange(0, 1200, 3)[:N_PARCELS], :] # row reduction to column count

# Load GAN data tensor:
with np.load(f'{data_dir}X_GAN.npz') as f:
    # f.files # Only stored array is X_GAN
    X_GAN = f['X_GAN']

# Helper functions:
t2a = lambda X: X.reshape(-1, X.shape[-1]) # Tensor to Array (unravels depth dimension)
a2t = lambda X, depth: X.reshape(depth, int(X.shape[0]/depth), X.shape[1]) # Array to Tensor

# Group Functional Connectivity (FC):
def group_FC(X):
    """
    Takes tensor X and returns the average correlation coefficient based
    functional connectivity matrix along the depth dimension.
    """
    fc = np.zeros([X.shape[0], X.shape[2], X.shape[2]])
    for j in range(X.shape[0]):
      fc[j, :, :] = np.corrcoef(X[j, :, :].T) # We need the arrays to be DxT, not TxD, as correlation is between parcels
    group_fc = fc.mean(axis=0) # Final group averaged correlation matrix
    return group_fc
X_FC = group_FC(X)
X_GAN_FC = group_FC(X_GAN)
plt.figure()
plt.suptitle("Functional Connectivity in Resting State fMRI")
plt.subplot(1, 2, 1)
plt.title('HCP fMRI Data')
plt.imshow(X_FC, interpolation="none", cmap="bwr", vmin=-1, vmax=1)
# Looks similar to what would be obtained with the full tensor in the tutorial (https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/fMRI/load_hcp.ipynb) 
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title('GAN fMRI Data')
plt.imshow(X_GAN_FC, interpolation="none", cmap="bwr", vmin=-1, vmax=1)
plt.colorbar()
# Connectome View:  
display = plotting.view_connectome(X_FC, atlas["coords"], edge_threshold="99%") #
display.save_as_html(f'{data_dir}HCP_connectome.html') # Open this manually and view in browser. Again, the results show some similarity, such as lack of connectivity in motor areas, and strong connectivity in V1.
display = plotting.view_connectome(X_GAN_FC, atlas["coords"], edge_threshold="99%") #
display.save_as_html(f'{data_dir}GAN_connectome.html') # Open this manually and view in browser. Again, the results show some similarity, such as lack of connectivity in motor areas, and strong connectivity in V1.
# if not os.path.exists(f'{data_dir}connectome.html'):

# Group Independent Component Analysis (ICA):
from sklearn.decomposition import FastICA
"""
# Single subject ICA:
X1 = X[0, :, :] # Data for Subject 0, to check the run time of this algo
ica = FastICA(n_components= 20, max_iter=200, random_state=seed) # Max. components is 360
# X1 = A1@S1
start = time.time()
S1 = ica.fit_transform(X1.T)  # Reconstruct signals/source matrix
stop = time.time()
print(f"Time Elapsed = {(stop-start):.2f}s") # Pretty damn good, 0.78s for a single subject!
A1 = ica.mixing_  # Get estimated mixing matrix

# Checking to see if the reconstruction works
# foo = A1@S1.T
# bar = ica.mean_[:, np.newaxis]
# baz = foo + bar
# np.allclose(baz, X1) # Will be True for n_components = 359, faces numerical issues at 360

# Top 20 components:
ica = FastICA(n_components= 20, max_iter=int(1e6), random_state=seed) # Max. components is 360
# X = A@S, where X is the depthwise flattened version of the tensor X
start = time.time()
S = ica.fit_transform(X.T).T  # Reconstruct signals/source matrix
# np.allclose(S@S.T, np.eye(20)) # Preliminary proof of independence: all ICs are orthonormal, hence uncorrelated. ALso, cov = corr.
stop = time.time()
print(f"Time Elapsed = {(stop-start):.2f}s") # 134s for C = 20

# Checking if top 10 components are in any way related to the top 20 (like in PCA):
ica_ = FastICA(n_components= 10, max_iter=int(1e6), random_state=seed) # Max. components is 360
start = time.time()
S_ = ica_.fit_transform(X.T).T  # Reconstruct signals/source matrix
stop = time.time()
print(f"Time Elapsed = {(stop-start):.2f}s") # 0.82s! for C = 10

softmax = lambda X: np.exp(X)/np.sum(np.exp(X))
c_index = np.apply_along_axis(softmax, 1, (S_@S.T)**2).argmax(axis=1) # View the most likely components from C = 20 for each in C = 10
foo = S[c_index, :]
plt.subplot(1, 3, 1)
sns.heatmap(S_)
plt.subplot(1, 3, 2)
sns.heatmap(foo)
# Each component of S_ shows only a weak correspondence with those found in S, despite unique matching. Likely due to "splitting" of ICs.
theta = np.linalg.inv(S@S.T)@S@S_.T # Best linear model
S_linear = theta.T@S
plt.subplot(1, 3, 2)
sns.heatmap(S_linear) # Indeed, the 10 ICs are just a linear combination of the 20 ICs! Some ICs are more mixed than others.
np.allclose(S_ == S_linear)

# Are the same 10 ICs recovered if we use a different seed?
ica = FastICA(n_components= 10, max_iter=int(1e6), random_state=69) # Max. components is 360
S = ica.fit_transform(X.T).T  # Reconstruct signals/source matrix
foo = S@S_.T # Yes! Just a quick look at the array confirms that
np.allclose(foo@foo.T, np.eye(10)) # Indeed, 'foo' is an orthonormal matrix
"""
X = t2a(X)
X_GAN = t2a(X_GAN)
def group_IC(X, C=10, max_iter=int(1e6), seed=0):
    """
    Returns the Group ICs (components x parcels) for the input array X.
    Standardization performed. Max components is N_PARCELS.
    """
    ica = FastICA(n_components=C, max_iter=max_iter, random_state=seed)
    start = time.time()   
    S = ica.fit_transform(X.T).T  # Reconstruct signals/source matrix
    stop = time.time()
    print(f"Time Elapsed = {(stop-start):.2f}s") # 0.82s! for C = 10
    # Working with normalized ICs as done in the literature
    S_norm = S - S.mean(axis=1, keepdims=True)
    S_norm /= S_norm.std(axis=1, keepdims=True)
    return S_norm
X_IC = group_IC(X)
# ICs = range(len(X_IC)) # If preparing views for all ICs
fsaverage = datasets.fetch_surf_fsaverage() # Has 10242 nodes/voxels. Surface mesh for viewing ICs.
ICs = [2, 7, 8]
for i in ICs: 
    for hemi in ['L', 'R']:
        vox2par = atlas[f"labels_{hemi}"] # Mapping from 10242 voxels to their corresponding 130 parcels for each hemi
        voxels = np.zeros_like(vox2par, dtype=float)
        if hemi == 'L':
            for parcel in range(int(N_PARCELS/2), N_PARCELS):
                voxels[vox2par == parcel] = X_IC[i, parcel] # Color entire parcel with IC values
            display = plotting.view_surf(fsaverage['pial_left'], voxels, black_bg=False, threshold=1, vmax=5) # infl, pial, sphere, white, orig. threshold = 2, black_bg=True
        else:
            for parcel in range(int(N_PARCELS/2)):
                voxels[vox2par == parcel] = X_IC[i, parcel] # Color entire parcel with IC values
            display = plotting.view_surf(fsaverage['pial_right'], voxels, black_bg=False, threshold=1, vmax=5)
        # if not os.path.exists(f'{data_dir}IC{i+1}_{hemi}.html'):
        display.save_as_html(f'{data_dir}IC{i+1}_{hemi}.html') # Open this manually and view in browser.
# We found nice ICs in X at 2(3 if using 1 indexing), 7(8), and 8(9) for the given threshold. They correspond to frontal, occipital, and temporo-parietal networks.
X_GAN_IC = group_IC(X_GAN)
# Looking for IC correspondence between both datasets:
(np.abs(X_IC@X_GAN_IC.T)).argmax(axis=1) # Given orthogonality of ICs, this matrix returns how similar each GAN IC is to the give real IC. Since ICs are just vectors, they may be flipped, so their projection magnitude is considered.
# The most similar ICs in X_GAN to these are accordingly: 3(4), 4(5), 2(3)
# Repeat for X_GAN_IC:
ICs = [3, 4, 2]
for i in ICs: 
    for hemi in ['L', 'R']:
        vox2par = atlas[f"labels_{hemi}"] # Mapping from 10242 voxels to their corresponding 130 parcels for each hemi
        voxels = np.zeros_like(vox2par, dtype=float)
        if hemi == 'L':
            for parcel in range(int(N_PARCELS/2), N_PARCELS):
                voxels[vox2par == parcel] = X_GAN_IC[i, parcel] # Color entire parcel with IC values
            display = plotting.view_surf(fsaverage['pial_left'], voxels, black_bg=False, threshold=1, vmax=5) # infl, pial, sphere, white, orig
        else:
            for parcel in range(int(N_PARCELS/2)):
                voxels[vox2par == parcel] = X_GAN_IC[i, parcel] # Color entire parcel with IC values
            display = plotting.view_surf(fsaverage['pial_right'], voxels, black_bg=False, threshold=1, vmax=5)
        # if not os.path.exists(f'{data_dir}IC{i+1}_{hemi}.html'):
        display.save_as_html(f'{data_dir}GAN_IC{i+1}_{hemi}.html') # Open this manually and view in browser.
plt.show() 
 # # Network visualization (eg. DMN):
 # for parcel in np.argwhere(regions[1, :] == 'Default').squeeze():
 #   voxels[vox2par == parcel] = 1 



