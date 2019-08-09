%% title:digits recognition mnist
%  author:zj
%  date:09/08/2019
%  discription:
%  struct of the model:    
%       input£º1*28*28
%       conv1£º5*5*1*5,stride=1,padding=0,valid
%       pool1£ºavge_2*2
%       conv2£º5*5*5*3,stride=1,padding=0,valid
%       pool2£ºavge_2*2
%       full£º 48
%       softmax£º10
%%
clear all;clc;

%% Network parameters settings
layer_c1_num=5;
layer_c2_num=3;
layer_f1_num=48;
layer_output_num=10;

learning_rate=1;

%initial parameters of convolution layer 
bias_c1=(2*rand(1,layer_c1_num)-ones(1,layer_c1_num))/sqrt(layer_c1_num);
kernel_c1=init_kernel(1,layer_c1_num);

bias_c2=(2*rand(1,layer_c2_num)-ones(1,layer_c2_num))/sqrt(layer_c2_num);
kernel_c2=init_kernel(layer_c1_num,layer_c2_num);

%initial pooling rule of pooling layer
pooling_a=ones(2,2)/4;

%initial parameters of output layer
weight_output=(2*rand(layer_f1_num,layer_output_num)-ones(layer_f1_num,layer_output_num))/sqrt(layer_f1_num);
bias_output=(2*rand(1,layer_output_num)-ones(1,layer_output_num))/sqrt(layer_output_num);

%% training
disp('Parameters initial finised!');
disp('Start training!');
for iter=1:20
    iter
for n=1:30
    for m=0:9
        %import images
        train_data=imread(strcat('train/',num2str(m),'_',num2str(n),'.bmp'));
        train_data=double(train_data);
        
        %% forward pass
        % conv1+pool1 layer
        for k=1:layer_c1_num
            state_c1(:,:,k)=conv2(train_data,rot90(kernel_c1(:,:,1,k),2),'valid');
            state_c1(:,:,k)=sigmoid(state_c1(:,:,k)+bias_c1(1,k));
            state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a)/4;
        end
        
        % conv2+pool2 layer
        for k=1:layer_c2_num
            state_c2_0=zeros(8,8);
            for k1=1:layer_c1_num
                state_c2_0=state_c2_0+conv2(state_s1(:,:,k1),rot90(kernel_c2(:,:,k),2),'valid');
            end
            state_c2(:,:,k)= state_c2_0;
            state_c2(:,:,k)=sigmoid(state_c2(:,:,k)+bias_c2(1,k));
            state_s2(:,:,k)=pooling(state_c2(:,:,k),pooling_a)/4;
        end
        
        % full layer
        state_f1=state_s2(:)';
        
        % output layer
        for nn=1:layer_output_num
            output(1,nn)=exp(state_f1*weight_output(:,nn)+bias_output(:,nn))/sum(exp(state_f1*weight_output+bias_output));
        end
        
        %% Backward pass
        [kernel_c1,kernel_c2,weight_output,bias_c1,bias_c2,bias_output]= ...
        CNN_upweight(learning_rate,m,train_data,state_c1,state_s1,state_c2,state_s2,state_f1,...
        output,kernel_c1,kernel_c2,weight_output,bias_c1,bias_c2,bias_output);
    end    
end
end
%% Parameters save
save c1 kernel_c1 bias_c1
save c2 kernel_c2 bias_c2
save output weight_output bias_output
disp('Parameters saved successfully,finished training!');

%% testing
LeNet_test;
