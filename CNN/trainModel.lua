require 'torch'
require 'nn'
require 'cunn'

function trainModels(trainbatch)
    local trainData = torch.load('data/processed-data/train_batch' .. (trainbatch) .. '.t7')

    setmetatable(trainData, 
        {__index = function(t, i) 
                        return {t.data[i], t.label[i]} 
                    end}
    )

    function trainData:size() 
        return self.data:size(1) 
    end

    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 8, 7, 7, 1, 1, 3, 3))
    net:add(nn.SpatialMaxPooling(3,3,2,2)) 
    net:add(nn.SpatialConvolution(8, 18, 7, 7, 1, 1, 3, 3))
    net:add(nn.SpatialMaxPooling(3,3,2,2))
    net:add(nn.View(18*7*7))            
    net:add(nn.Linear(18*7*7, 150)) 
    net:add(nn.Linear(150, 100))
    net:add(nn.Linear(100, 10))      
    net:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    local trainer = nn.StochasticGradient(net, criterion)

    net = net:cuda()
    criterion = criterion:cuda()
    trainData.data = trainData.data:cuda()

    trainer.learningRate = 0.001
    trainer.maxIteration = 30
    trainer:train(trainData)
    trainer.learningRate = 0.0006
    trainer.maxIteration = 10
    trainer:train(trainData)
    trainer.learningRate = 0.0001
    trainer.maxIteration = 10
    trainer:train(trainData)

    torch.save('models/model-'..(trainbatch)..'.t7', net)
end

for n=1,5 do
    trainModels(n)
end