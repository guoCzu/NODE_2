function dsig(z)
    a = sig(z) .* (1.0 .-sig(z));
    return a 
end
#######################
function d2sig(z)
    a = sig(z) .* (1.0 .-sig(z)) .* (1.0 .- 2.0* sig(z) );
    return a 
end#######################
function d3sig(z)
    a = sig(z) .* (1.0 .-sig(z)) .* (1.0 .- 6.0* sig(z) .+ 6.0 * sig(z) .* sig(z) );
    return a 
end