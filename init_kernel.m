function kernel_c1=init_kernel(layer_num1,layer_num2)
% kernel weights initial
for n=1:layer_num1
    for n1=1:layer_num2
        kernel_c1(:,:,n,n1)=(2*rand(5,5)-ones(5,5))/12;
    end
end
end