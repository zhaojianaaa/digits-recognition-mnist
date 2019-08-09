function [kernel_c1,kernel_c2,weight_output,bias_c1,bias_c2,bias_output]=CNN_upweight(yita,...
    classify,train_data,state_c1,state_s1,state_c2,state_s2,state_f1,...
                                    output,kernel_c1,kernel_c2,weight_output,bias_c1,bias_c2,bias_output)
%% Get parameters of network
layer_c1_num=size(state_c1,3);
layer_c2_num=size(state_c2,3);
layer_output_num=size(output,2);

%% Error of the output layer
label=zeros(1,layer_output_num);
label(1,classify+1)=1;
delta_layer_output=output-label;

%% Gradient of weights of output layer
for n=1:layer_output_num
    delta_weight_output_temp(:,n)=delta_layer_output(1,n)*state_f1';
end

%% Error from output layer back to full layer
delta_layer_full=delta_layer_output*weight_output'.*state_f1;

%% Error from full layer back to pool layer2
%delta_layer_chihau2=reshape(delta_layer_full,4,4,3).*state_s2;
delta_layer_chihau2=reshape(delta_layer_full,4,4,3);

%% Error from pool layer2 back to conv2 layer2
for n=1:layer_c2_num
    % Error of conv2
    delta_layer_c2(:,:,n)=kron(delta_layer_chihau2(:,:,n),ones(2,2)/4).*(1-sigmoid(state_c2(:,:,n)).*sigmoid(state_c2(:,:,n)));
    % Gridents of conv2 bias
    delta_bias_c2(1,n)=sum(sum(delta_layer_c2(:,:,n)));
end

%% Gridents of conv2 weights
for n=1:layer_c2_num
    for n1=1:layer_c1_num
        delta_kernel_c2_temp(:,:,n1,n)=conv2(state_s1(:,:,n1),rot90(delta_layer_c2(:,:,n),2),'valid');
    end
end

%% Error from conv2 layer2 back to pool layer1
for n=1:layer_c1_num
    state_c2_1=zeros(12,12);
    for n1=1:layer_c2_num
        state_c2_1=state_c2_1+conv2(kernel_c2(:,:,n1),delta_layer_c2(:,:,n1),'full');
    end
    delta_layer_chihua1(:,:,n)=state_c2_1;
    %delta_layer_chihua1(:,:,n)=delta_layer_chihua1(:,:,n).*state_s1(:,:,n);
end

%% Error from pool layer1 back to conv2 layer1
for n=1:layer_c1_num
    delta_layer_c1(:,:,n)=kron(delta_layer_chihua1(:,:,n),ones(2,2)/4).*sigmoid(state_c1(:,:,n)).*(1-sigmoid(state_c1(:,:,n)));
    % Gridents of conv1 bias
    delta_bias_c1(1,n)=sum(sum(delta_layer_c1(:,:,n)));
end

%% Gridents of conv2 weights
for n=1:layer_c1_num
    delta_kernel_c1_temp(:,:,n)=conv2(train_data,rot90(delta_layer_c1(:,:,n),2),'valid');
end

%% Parameters Update 
weight_output=weight_output-yita*delta_weight_output_temp;
bias_output=bias_output-yita*delta_layer_output;
bias_c2=bias_c2-yita*delta_bias_c2/64;
kernel_c2=kernel_c2-yita*delta_kernel_c2_temp;
bias_c1=bias_c1-yita*delta_bias_c1/(24*24);
kernel_c1=kernel_c1-yita*delta_kernel_c1_temp;
end