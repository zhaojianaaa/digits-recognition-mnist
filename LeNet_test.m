%% matlab implements LeNet
%  zj
%  2019.08.09
% discription:
% test: implements the forward pass    

%%
clear all;clc;

%% parameters setting
layer_c1_num=5;
layer_c2_num=3;
layer_output_num=10;
pooling_a=ones(2,2)/4;

%%
load c1 kernel_c1 bias_c1
load c2 kernel_c2 bias_c2
load output weight_output bias_output
disp('Parameters import finished£¡ Start testing!');
count=0;
for n=1:10
    for m=0:9
        %import images
        train_data=imread(strcat('test/',num2str(m),'_',num2str(n),'.bmp'));
        train_data=double(train_data);
        
        %% forward pass
        % conv1+pool1 layer
        for k=1:layer_c1_num
            state_c1(:,:,k)=conv2(train_data,rot90(kernel_c1(:,:,k),2),'valid');
            state_c1(:,:,k)=sigmoid(state_c1(:,:,k)+bias_c1(1,k));
            state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a)/4;
        end
        
        % conv2+pool2 layer       
        for k=1:layer_c2_num
            state_c2_0=zeros(8,8);
            for k1=1:layer_c1_num
                state_c2_0 = state_c2_0 + conv2(state_s1(:,:,k1),rot90(kernel_c2(:,:,k),2),'valid');
            end
            state_c2(:,:,k)= state_c2_0;
            state_c2(:,:,k)=sigmoid(state_c2(:,:,k)+bias_c2(1,k));
            state_s2(:,:,k)=pooling(state_c2(:,:,k),pooling_a)/4;
        end       

        state_f1=state_s2(:)';
        
        % output layer
        for nn=1:layer_output_num
            output(1,nn)=exp(state_f1*weight_output(:,nn)+bias_output(:,nn))/sum(exp(state_f1*weight_output+bias_output));
        end
        %% Get the results
        [p,classify]=max(output);
        if (classify==m+1)
            count=count+1;
        end
        fprintf('%d --> %d \n',m,classify-1);
    end
end
        fprintf('recog rate:');
        count/90