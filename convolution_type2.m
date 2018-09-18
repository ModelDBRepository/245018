function [conv, conv1] = convolution_type2 (conv1, tau_m, eps0, X)
% exponential decay 
% input =       conv1= convolution, decay component 
%               tau_m = decay time constant
%               eps0= multiplying constant, scales the whole convolution  
%               X = spike train vector, if there is a spike the convolution jumps by 1
%output =       conv = total convolution (scaled by eps0)
%               conv1 = convolution, decay component (updated)

conv1 = conv1 + (-conv1)/tau_m + X;
conv = eps0.*(conv1);

end