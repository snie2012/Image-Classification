require 'torch'

function prepareData(batchNum)
    local trsize = 10000*(batchNum+1)
    local tesize = 10000
    local subset = {}

    local trainData = {
      data = torch.Tensor(trsize, 3*32*32),
      label = torch.Tensor(trsize),
      mean = {},
      stdv = {}
    }
    
    for i = 0, batchNum do
       subset = torch.load('data/cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
       trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
       trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    end
    trainData.labels = trainData.labels + 1

    subset = torch.load('data/cifar-10-batches-t7/test_batch.t7', 'ascii')
    testData = {
       data = subset.data:t():double(),
       labels = subset.labels[1]:double(),
    }
    testData.labels = testData.labels + 1

    trainData.data = trainData.data:reshape(trsize,3,32,32)
    testData.data = testData.data:reshape(tesize,3,32,32)

    local mean = {}
    local stdv  = {}
    for j=1,3 do
      mean[j] = trainData.data[{ {}, {j}, {}, {}  }]:mean()
      trainData.data[{ {}, {j}, {}, {}  }]:add(-mean[j])
      testData.data[{ {}, {j}, {}, {}  }]:add(-mean[j])
      stdv[j] = trainData.data[{ {}, {j}, {}, {}  }]:std()
      trainData.data[{ {}, {j}, {}, {}  }]:div(stdv[j])
      testData.data[{ {}, {j}, {}, {}  }]:div(stdv[j])
    end

    torch.save('data/processed-data/train_batch' .. (batchNum+1) .. '.t7', trainData)
    torch.save('data/processed-data/test_batch' .. (batchNum+1) .. '.t7', testData)
end

for n=0,4 do
  prepareData(n)
end
