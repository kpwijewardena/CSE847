% Main training file
fileData = fopen('data.txt','r');
fileLabels = fopen('labels.txt','r');
formatSpec = '%f';
size_all = [4601 Inf];
data = fscanf(fileData,formatSpec, size_all);
intercept = ones(length(data),1);
data_all = [data intercept];
labels = fscanf(fileLabels,formatSpec, size_all);

data_train = data_all(1:2000,:);
label_train = labels(1:2000,:);
data_test = data_all(2001:4601,:);
label_test = labels(2001:4601,:);

weights = logistic_train(data_train, label_train, 1e-5, 1000);

final_val = data_test * weights;

plot(final_val);

count = 0;
for i = 1:size(label_test,1)
    if(final_val(i)>0 && label_test(i) > 0)
        count = count + 1;
    end
    if(final_val(i)<0 && label_test(i) < 0)
        count = count + 1;
    end
end

count_final = (count/length(label_test))*100