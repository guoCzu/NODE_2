include("Forward.jl")
include("Sig.jl")
include("Dsig.jl")
function backPropagate(H,w_H,b_H, w_out,n,x,f,pf,IC,eta,droprate,epoch)
    # trial solutions
    y_trial = 0;
    
    # trial solution derivatives
    dytrial_dx = 0;
    
    # feedforward
    # activations
    a_H = zeros(H,1);
    # weighted inputs
    z_H = zeros(H,1);
    # activations
    a_out = 0;
    # weighted input
    z_out = 0;
    
    # output layer error
    err_out = 0;
    # hidden layer error
    err_wH = zeros(H,1);
    err_bH = zeros(H,1)
    err_wO = zeros(H,1)
    
    # grad of output layer weights
    dw_out = zeros(H,1);
    # grad of hidden layer weights
    dw_H = zeros(H,1);
    # grad of hidden layer biases
    db_H = zeros(H,1);

    # gradient of Network wrt error
    grad_N = 0;

    # Cost Function Gradient
    # grad_C(dy_t,y_t,x) =  2*(dy_t - f(y_t,x))*(1-pf(x)*x);
    # grad_C1(dy_t,y_t,x) = 2*(dy_t - f(y_t,x))
    grad_C1(dytrial_dx,y_trial,x) = 1-pf(x)*x
    grad_C2(dytrial_dx,y_trial,x) = x
    grad_C0(dytrial_dx,y_trial,x) = 2*(dytrial_dx - f(y_trial,x))


    # drop learning rate by half every 20 epochs
    # eta = eta*(1/2)^floor(epoch/droprate);
    # eta = eta * 1.0/2^(floor((epoch+1)/2000))

    # loop over 
    for i = 1:n # x 的100个点。
        # feedforward 
        # current
        a_H,z_H,a_out,z_out = feedForward(w_H,b_H,w_out,x[i]);
        
        # trial solutions
        # y_t_m = IC + x[i]*(a_out-h);
        y_t = IC + x[i]*a_out; # a_out 即为文献中的 Ψ(xi)=a_out 
        # y_t_p = IC + x[i]*(a_out+h);
        
        # trial solution derivative
        dytrial_dx = a_out + x[i] * sum(w_out.*w_H.*dsig(z_H));
        # w_out.*w_H.*dsig(z_H) 3项均为 10 * 1 向量， 
        # gradient of network wrt output
        grad_N0 = grad_C0(dytrial_dx,y_trial,x[i]);
        grad_N1 = grad_C1(dytrial_dx,y_trial,x[i]);
        grad_N2 = grad_C2(dytrial_dx,y_trial,x[i])
        
        # error of layers
        # output layer
        err_wO1 = grad_N0 * grad_N1 * sig(z_H)
        err_wO2 = grad_N0 * grad_N2 * (dsig(z_H) .* w_H)
        err_wO = err_wO1 .+ err_wO2        
        # err_out = grad_N*dsig(z_out);
        # err_out = grad_N *sig(z_H)
        # err_out = err_out + x[i]* dsig(z_H) .* w_H
        # hidden layer
        err_wH1 = grad_N0 * grad_N1 * x[i] * w_out.*dsig(z_H)
        err_wH2 = grad_N0 * grad_N2 * (w_out .* dsig(z_H) .+ x[i] * w_out .* d2sig(z_H) .* w_H)
        err_wH = err_wH1 .+ err_wH2
        ###
        err_bH1 = grad_N0 * grad_N1 * w_out.*dsig(z_H)
        err_bH2 = grad_N0 * grad_N2 * (w_out .* d2sig(z_H) .* w_H)
        err_bH = err_bH1 .+ err_bH2
        # err_H = (w_out*err_out).*dsig(z_H);
        # err_H = grad_N * (w_out.*dsig(z_H))  ;
        # err_H = err_H + (grad_N1*w_out).*dsig(z_H)  .*(1 .-x[i]*(w_H.-2 * w_H .*sig(z_H))  ) 
        # err_wH = err_H + grad_N1*(w_out.*dsig(z_H) .+ x[i] * w_out .*  d2sig(z_H) .* w_H)
        # err_bH = err_H + grad_N1*(w_out .*  d2sig(z_H) .* w_H)
        #   grad_N1 * x[i]*(1-x[i]*(1-2*sig(z_H)) .* w_H)

        # gradients of network parameters
        # update llhlearning rate
        
        # output layer weights
        # dw_out = a_H*err_out;
        dw_out = err_wO;
        # hidden layer bias
        db_H = err_bH;
        # hidden layer weights
        dw_H = err_wH;
        
        # gradient descent
        # output layer weights
        w_out = w_out - eta*dw_out;
        # hidden layer bias
        b_H = b_H - eta*db_H;
        # hidden layer weights
        w_H = w_H - eta*dw_H;
    end
    return w_H,b_H,w_out
end