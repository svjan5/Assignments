function [ out ] = shuffle( in )
%SHUFFLE Summary of this function goes here
%   Detailed explanation goes here
out = in(randperm(size(in,1)),:);

end

