# sparse_encoder_testsuite

This is a test-suite that implements some common well-known sparse recovery techniques for compressed sensing, as well as a 'problem_factory' to create synthethic problems of this kind. Currently, the implementation is focussed on running experiments for support recovery although this might change in future updates. 

Amost others, the following techniques are implemented:
  - OMP (using scipy OMP)
  - Lasso (using scipy LASSO)
  - LAR (using scipy LAR)
  - Cosamp
  - iterative hard-thresholding (basic iteration)
  - preconditioned versions of all methods (preconditioning as in [1])
  - ...

To run experiments, one only needs to use the methods run_single (for a single experiments), run_batch (for numerous repititions of an experiments with the same characteristics (matrix dimensions, signal sparsity, and so on) and run_multiple_batches for numerous repititions and varying characteristics (from the top folder). The problems characteristics itself are defined inside these functions in a python dictionary, whereas descriptions of the keys/the dictionary can be found in files in the 'problem_factory' folder (containing all methods regarding set-up of experiments) and the run_batch files in the first level folder.



Sources:

[1] Jia, Jinzhu, and Karl Rohe. "Preconditioning to comply with the irrepresentable condition." arXiv preprint arXiv:1208.5584 (2012).
