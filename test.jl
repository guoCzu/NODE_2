using Plots



 N = 100;

 # Training Points
 x= Array{Float64}(undef,N)
 dx = 1.0/N
 for j = 1:N
     x[j] = (j-1) * dx
 end


# y(x) = 0.1*exp.(-2* x) .* (29 .+ 49 *exp.(4*x) .- 48*exp.(2*x) .* cos.(x))
y(x) = exp(-2* x) * (2 + 4 *exp(4* x) - 3* exp(2* x) * cos(2* x))


p0 = plot(1)
p2 = plot!(p0,x,y.(x),color="red",label = "analysis solution",legend=true)
    