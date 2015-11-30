a) Hardware requirements: High performance GPU and at least 1.5G memory of the GPU.
b) Follow the instructions on http://torch.ch/docs/getting-started.html#_ to install lua and torch. If installed successively, run the command 'th' in the terminal will open the console for torch.
c) To run the code on GPU, you have to download and install CUDA from NVDIA’s official website.
d) Use luarocks(package manager for torch) to install package 'nn' and 'cunn'.
e) Run prepareData.lua, trainModel.lua, testModel.lua in order, then the results will be save in 'results/' folder.