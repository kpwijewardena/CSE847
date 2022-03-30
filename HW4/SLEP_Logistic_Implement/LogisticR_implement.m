clear, clc;

% This is an example for running the function LogisticR
%
%  Problem:
%
%  min  f(x,c) = - weight_i * log (p_i) + 1/2 * rsL2 * ||x||_2^2 
%                 + rho * \|x\|_1 
%
%  a_i denotes a training sample,
%      and a_i' corresponds to the i-th row of the data matrix A
%
%  y_i (either 1 or -1) is the response
%     
%  p_i= 1/ (1+ exp(-y_i (x' * a_i + c) ) ) denotes the probability
%
%  weight_i denotes the weight for the i-th sample
%
% For detailed description of the function, please refer to the Manual.
%
%% Related papers
%
% [1]  Jun Liu and Jieping Ye, Efficient Euclidean Projections
%      in Linear Time, ICML 2009.
%
% [2]  Jun Liu and Jieping Ye, Sparse Learning with Efficient Euclidean
%      Projections onto the L1 Ball, Technical Report ASU, 2008.
%
% [3]  Jun Liu, Jianhui Chen, and Jieping Ye, 
%      Large-Scale Sparse Logistic Regression, KDD, 2009.
%
%% ------------   History --------------------
%
% First version on August 10, 2009.
%
% September 5, 2009: adaptive line search is added
%
% For any problem, please contact Jun Liu (j.liu@asu.edu)

cd ..
cd ..

root=cd;
addpath(genpath([root '/SLEP']));
                     % add the functions in the folder SLEP to the path
                   
% change to the original folder
cd Examples/L1;

% ---------------------- generate random data ----------------------
A_file = load('ad_data.mat');        % the data matrix
A = A_file.X_train;
 
y = A_file.y_train;                     % the response

rho1=0.1e-8;
rho2=0.01; 
rho3=0.1;
rho4=0.2;
rho5=0.3;
rho6=0.4;
rho7=0.5;
rho8=0.6;
rho9=0.7;
rho10=0.8;
rho11=0.9;
rho12=1.0;
%----------------------- Set optional items -----------------------
opts=[];

% Termination 
opts.rFlag=1;
opts.tol = 1e-6;
opts.tFlag=4;       % run .maxIter iterations
opts.maxIter=5000;    % maximum number of iterations

%----------------------- Run the code LeastR -----------------------
[x1, c1, funVal1, ValueL1]= LogisticR(A, y, rho1, opts);
[x2, c2, funVal2, ValueL2]= LogisticR(A, y, rho2, opts);
[x3, c3, funVal3, ValueL3]= LogisticR(A, y, rho3, opts);
[x4, c4, funVal4, ValueL4]= LogisticR(A, y, rho4, opts);
[x5, c5, funVal5, ValueL5]= LogisticR(A, y, rho5, opts);
[x6, c6, funVal6, ValueL6]= LogisticR(A, y, rho6, opts);
[x7, c7, funVal7, ValueL7]= LogisticR(A, y, rho7, opts);
[x8, c8, funVal8, ValueL8]= LogisticR(A, y, rho8, opts);
[x9, c9, funVal9, ValueL9]= LogisticR(A, y, rho9, opts);
[x10, c10, funVal10, ValueL10]= LogisticR(A, y, rho10, opts);
[x11, c11, funVal11, ValueL11]= LogisticR(A, y, rho11, opts);
[x12, c12, funVal12, ValueL12]= LogisticR(A, y, rho12, opts);

count = 0;
for i = 1:size(x1,1)
    if(x1(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x2,1)
    if(x2(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x3,1)
    if(x3(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x4,1)
    if(x4(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x5,1)
    if(x5(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x6,1)
    if(x6(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x7,1)
    if(x7(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x8,1)
    if(x8(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x9,1)
    if(x9(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x10,1)
    if(x10(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x11,1)
    if(x11(i)~=0)
        count = count + 1;
    end
end

count

count = 0;
for i = 1:size(x12,1)
    if(x12(i)~=0)
        count = count + 1;
    end
end

count

scores1 = A_file.X_test*x1;
scores2 = A_file.X_test*x2;
scores3 = A_file.X_test*x3;
scores4 = A_file.X_test*x4;
scores5 = A_file.X_test*x5;
scores6 = A_file.X_test*x6;
scores7 = A_file.X_test*x7;
scores8 = A_file.X_test*x8;
scores9 = A_file.X_test*x9;
scores10 = A_file.X_test*x10;
scores11 = A_file.X_test*x11;
scores12 = A_file.X_test*x12;

labels = A_file.y_test;

% figure;
[X1,Y1,T1,AUC1] = perfcurve(labels,scores1,1);
[X2,Y2,T2,AUC2] = perfcurve(labels,scores2,1);
[X3,Y3,T3,AUC3] = perfcurve(labels,scores3,1);
[X4,Y4,T4,AUC4] = perfcurve(labels,scores4,1);
[X5,Y5,T5,AUC5] = perfcurve(labels,scores5,1);
[X6,Y6,T6,AUC6] = perfcurve(labels,scores6,1);
[X7,Y7,T7,AUC7] = perfcurve(labels,scores7,1);
[X8,Y8,T8,AUC8] = perfcurve(labels,scores8,1);
[X9,Y9,T9,AUC9] = perfcurve(labels,scores9,1);
[X10,Y10,T10,AUC10] = perfcurve(labels,scores10,1);
[X11,Y11,T11,AUC11] = perfcurve(labels,scores11,1);
[X12,Y12,T12,AUC12] = perfcurve(labels,scores12,1);

AUC1
AUC2
AUC3
AUC4
AUC5
AUC6
AUC7
AUC8
AUC9
AUC10
AUC11
AUC12