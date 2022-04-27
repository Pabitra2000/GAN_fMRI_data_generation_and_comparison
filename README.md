# GAN_fMRI_data_generation_and_comparison

This project was made for Machine Learning in Signal Processing course offered in IISc in Spring sem. The purpuse is to generate fMRI data from existing fMRI data using Generative Adversial Networks(GANs). The GANs used has been implemented by keras library in python and uses Dense Neural Networks.

The GAN implementation is made by Pabitra Sharma(pabitras@iisc.ac.in)                           
The rs_fmri code is made by Sveekruth Sheshagiri Pai(sveekruths@iisc.ac.in)

# References
1.	https://compneuro.neuromatch.io/projects/fMRI/README.html [Neuromatch Academy projects page – curated HCP Data.]
2.	https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/fMRI/load_hcp.ipynb [Specific Colab notebook adapted for this project.]
3.	http://www.di.fc.ul.pt/~jpn/r/ica/index.html [Notes on ICA by João Neto, adapted from J Nathan Kutz’s lecture on the same topic.]
4.	https://www.youtube.com/watch?v=n95tOxWcQkw [Video tutorial – high level overview of ICA for fMRI data.]
5.	https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC [FSL toolbox - MELODIC software guide. Describes the procedure used for computing Group and Individual ICA with fMRI data.]
6.	http://ric.uthscsa.edu/personalpages/lancaster/SPM_Class/Lecture_18/melodic.pdf [In-depth version of the above.]
7.	Damoiseaux, J. S., Rombouts, S. A., Barkhof, F., Scheltens, P., Stam, C. J., Smith, S. M., & Beckmann, C. F. (2006). Consistent resting-state networks across healthy subjects. Proceedings of the National Academy of Sciences of the United States of America, 103(37), 13848–13853. https://doi.org/10.1073/pnas.0601417103 [Landmark paper describing resting state fMRI ICs/networks.]
8.	https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.htm l#blind-source-separation-using-fastica [FastICA implementation with scikit-learn library for Python. Based on the iterative ICA algorithm by Aapo Hyvärinen.]
9.	https://dartbrains.org/content/RSA.html [Primer on Representation Similarity Analysis (RSA).]
10.	Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis - connecting the branches of systems neuroscience. Frontiers in systems neuroscience, 2, 4. https://doi.org/10.3389/neuro.06.004.2008 [Landmark paper describing RSA.]
