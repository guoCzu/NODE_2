include("Sig.jl")
function feedForward(w_H,b_H,w_out,x0)
    # weighted inputs to hidden layer
    z_H = w_H*x0 + b_H;
    # activation of hidden layer
    a_H = sig(z_H);
    # println("a_H=",a_H)
    # println("w_out=",w_out)
    # weighted input to output layer
    z_out = w_out' * a_H ;
    # println("z_out=",z_out)
    # activation of output layer
    a_out = z_out[1] ;
    # println("a_out= ",a_out)
    # 这里的输出有激活码？MATLAB代码里没有。
    
    return a_H,z_H,a_out,z_out
end