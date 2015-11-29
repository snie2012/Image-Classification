require 'torch'
require 'nn'
require 'cunn'

function testModel(model, resultpath, testbatch)
    local testData = torch.load('../data/test_batch_'..(testbatch)..'.t7')
    testData.data = testData.data:cuda()
    io.open(resultpath, 'w+')
    local f = io.open(resultpath, 'a')
    io.input(f)

    local correct = 0
    for i=1,10000 do
        local groundtruth = testData.label[i]
        local prediction = model:forward(testData.data[i])
        local confidences, indices = torch.sort(prediction, true)
        if groundtruth == indices[1] then
            correct = correct + 1
        end
    end

    f:write(correct..' ', 100*correct/10000 .. ' % ')
    f:write('\n')
    io.flush(f)
    io.close(f)
end

for j=1,5 do
    model = torch.load('models/model-'..(j)..'.t7')
    testModel(model, 'results/test-result-'..(j)..'.txt', j)
end
