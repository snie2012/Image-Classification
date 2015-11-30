• Libarries needed for running code:
  1. Theano 0.7
  2. CUDA (If GPU acceleration is needed)
  3. PIL
• Main file for running SDAE: TestStackedDAE.py
• For applying GPU calculation, environment setting is needed. So, please run these shell bash scripts, e.g., sh_TestStackedDAE_5.sh.
• Since training images are separated in 5 batches, I've used different bash scripts for loading different number of training images. sh_TestStackedDAE_5.sh loads all 50k training images.
• The serialized dataset for python is used in these SDAE scripts. Please download them from https://www.cs.toronto.edu/~kriz/cifar.html
