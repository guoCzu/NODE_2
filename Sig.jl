function sig(z)
    # println("z= ",1.0/(1.0 .+ exp.(-z)))
    # for i=1:length(z)
    #     println("z_i=",z[i])
    #     println("1+z_i=",i," ",1/(1+exp(-z[i])))
    # end
   
    # z = 1.0 .+ exp.(-z)
    # z = z'
    # println("z=",z)
    # println("ok")
    a = 1.0 ./(1.0 .+ exp.(-z));
    # println("a= ",a)
    return a
end