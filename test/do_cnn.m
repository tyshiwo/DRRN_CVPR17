function [ outputdata ] = do_cnn( model_1, weights, inputdata )
%%%%% change TXT1  %%%%%
[wid,hei,channels_num] = size(inputdata);
fidin1=fopen(model_1,'r+');
i=0;
while ~feof(fidin1)
    tline=fgetl(fidin1);
    i=i+1;
    newtline{i}=tline;
    if i == 4
        newtline{i}=[tline(1:11) num2str(channels_num)];
    end
    if i == 5
        newtline{i}=[tline(1:11) num2str(hei)];
    end
    if i == 6
        newtline{i}=[tline(1:11) num2str(wid)];
    end
end
fclose(fidin1);
fidin1=fopen(model_1,'w+');
for j=1:i
    fprintf(fidin1,'%s\n',newtline{j});
end
fclose(fidin1);
%%%%%%%%%%%%%%%%%%%%%%%%
net_1 = caffe.Net(model_1, weights, 'test'); % create net and load weights

%img = permute(inputdata,[2, 1, 3]);
res = net_1.forward({inputdata});
outputdata = res{1};
caffe.reset_all();


end

