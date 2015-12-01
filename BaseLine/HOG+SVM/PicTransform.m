% 1.Set the Batch name first: data_batch_1, data_batch_2, data_batch_3, 
%   data_batch_3, data_batch_4, data_batch_5. it will load the batch data,
%   and save the image in the file

% 2.Set the image start number for each batch:
%   data_batch_1: imageNumberStart = 0, it contains pic index (0-9999)
%   data_batch_2: imageNumberStart = 10000, it contains pic index (10000-19999)
%   data_batch_3: imageNumberStart = 20000, it contains pic index (20000-29999)
%   data_batch_4: imageNumberStart = 30000, it contains pic index (30000-39999)
%   data_batch_5: imageNumberStart = 40000, it contains pic index (40000-49999)
%   test_batch: imageNumberStart = 50000, it contains pic index (50000-59999)


BatchName = 'test_batch';
imageNumberStart =50000;

cd (strcat('/Users/linting/Desktop/CSC522/Project/simplecv/data/cifar-10-batches-mat/',BatchName))
Batch = load(strcat(BatchName,'.mat'));


data = Batch.data;
data_labels = Batch.labels;

image_array = zeros(32,32,3);
imageNumber = imageNumberStart;
for row_index = 1:10000
    col_index = 1;
    image_data = data(row_index,:);
    image_label = data_labels(row_index); 
    for i = 1:3
        for j = 1: 32
            for k = 1:32
                image_array(j,k,i)=image_data(col_index);
                col_index = col_index + 1;
            end
        end 
    end

    
    if image_label == 0
        Dir_name = '0';
    elseif image_label == 1
        Dir_name = '1';
    elseif image_label == 2
        Dir_name = '2';
    elseif image_label == 3
        Dir_name = '3';
    elseif image_label == 4
        Dir_name = '4';
    elseif image_label == 5
        Dir_name = '5';
    elseif image_label == 6
        Dir_name = '6';
    elseif image_label == 7
        Dir_name = '7';
    elseif image_label == 8
        Dir_name = '8';
    else
        Dir_name = '9';
    end
    
    % save image
    filePath = '/Users/linting/Desktop/CSC522/Project/simplecv/data/cifar-10-batches-mat/';
    batchname = BatchName; 
    path = strcat(filePath,BatchName, '/', Dir_name, '/', int2str(imageNumber),'.png');
    image_array = uint8(image_array);
    imwrite(image_array,path,'png');
    imageNumber = imageNumber+1;
    
    fprintf ('batchname: %s \n',batchname)
    fprintf ('Dir_name: %s \n',Dir_name)
    fprintf ('imageNumber: %i\n',imageNumber)
    
       
end
               
            
