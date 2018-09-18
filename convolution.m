function [conv, conv_decay, conv_rise] = convolution (conv_decay, conv_rise, tau_m, tau_s, eps0, X, w)
%input =    conv_decay = convolution, decay component
%           conv_rise = convolution, rise component
%           tau_m = decay time constant
%           tau_s = rise time constant
%           eps0 = multiplying constant, scales the whole convolution 
%           X = spike train vector, if there is a spike the convolution jumps by 1
%           w = multiplying constant for X (weights the input)
%
%output =   conv = total convolution 
%           conv_decay = convolution, decay component (updated)
%           conv_rise = convolution, rise component (updated)


%computes a convolution with both rise and decay exponentials

%in no input is given for w, set it all w to one
if nargin < 7
    w = ones(size(X))';
end

conv_decay = conv_decay + (-conv_decay)/tau_m + X.*w'; 
conv_rise = conv_rise + (-conv_rise)/tau_s + X.*w';
conv = eps0*(conv_decay-conv_rise)/(tau_m-tau_s);

end