include("Forward.jl")
include("Sig.jl")
include("Dsig.jl")
function backPropagate(H,w_H,b_H, w_out,n,x,f,a,b,df_dy,A,B,eta,droprate,epoch)
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
    # grad_N = 0;

    # Cost Function Gradient
    # E1_0(x,a_out,da_out_dx,d2a_out_dx2,y_trial) = a(x)*B + b(x)*A + x* b(x) *B # A,B 是初始值
    # E1_1(x,a_out,da_out_dx,d2a_out_dx2,y_trial) = (2+2*x*a(x)+x^2*b(x))*a_out
    # E1_2(x,a_out,da_out_dx,d2a_out_dx2,y_trial) = (4*x+x^2*a(x))* da_out_dx
    # E1_3(x,a_out,da_out_dx,d2a_out_dx2,y_trial) = x^2 * d2a_out_dx2
    # E1_4(x,a_out,da_out_dx,d2a_out_dx2,y_trial) = -f(x,y_trial)
    # grad_C1(dytrial_dx,y_trial,x) = 1-pf(x)*x



    # drop learning rate by half every 20 epochs
    # eta = eta*(1/2)^floor(epoch/droprate);
    eta = eta * 1.0/2^(floor((epoch+1)/5000))

    # loop over 
    for i = 1:n # x 的100个点。
        # feedforward 
        # current
        a_H,z_H,a_out,z_out = feedForward(w_H,b_H,w_out,x[i]);
        
        da_out_dx = w_out' * (dsig(z_H) .* w_H) 
        d2a_out_dx2 = w_out' * (d2sig(z_H) .* w_H .* w_H)
        # trial solutions
        # y_t_m = IC + x[i]*(a_out-h);
        # y_t = IC + x[i]*a_out; # a_out 即为文献中的 Ψ(xi)=a_out 
        # y_t_p = IC + x[i]*(a_out+h);
        y_trial = A + B * x[i] + x[i]^2 * a_out 
        # trial solution derivative
        dytrial_dx = B + 2*x[i]*a_out + x[i]^2 * da_out_dx;
        # trial solution 2 derivative
        d2ytrial_dx2 = 2 * a_out + 4*x[i] * da_out_dx + x[i]^2 * d2a_out_dx2;
        # w_out.*w_H.*dsig(z_H) 3项均为 10 * 1 向量， 
        da_out_dwH = x[i] * w_out .* dsig(z_H)
        da_outP_dwH =  w_out .* (dsig(z_H) .+ x[i] * d2sig(z_H) .* w_H)    # a_outP = da_out_dx
        da_out2P_dwH = w_out .* (2 * d2sig(z_H) .* w_H .+ x[i]* d3sig(z_H) .* w_H .* w_H)
        da_out_dwO = sig(z_H)
        da_outP_dwO = dsig(z_H) .* w_H
        da_out2P_dwO = d2sig(z_H) .* w_H .* w_H
        da_out_dbH = w_out.*dsig(z_H)
        da_outP_dbH = (w_out .* d2sig(z_H) .* w_H)
        da_out2P_dbH = w_out .* d3sig(z_H) .* w_H .* w_H
        # gradient of network wrt output
        dE1_da_out = 2+2*x[i]*a(x[i])+x[i]^2*b(x[i])-x[i]^2*df_dy(x[i],y_trial)
        dE1_da_outP = 4*x[i]+x[i]^2*a(x[i])
        dE1_da_out2P = x[i]^2
        # dE1_df = df_dy(x[i],y_trial)*x[i]^2
        dE1 = [dE1_da_out, dE1_da_outP,dE1_da_out2P]
        da_out_dwO = [da_out_dwO,da_outP_dwO,da_out2P_dwO  ] # dp 代表 parameter的倒数
        da_out_dwH = [da_out_dwH,da_outP_dwH,da_out2P_dwH  ]
        da_out_dbH = [da_out_dbH,da_outP_dbH,da_out2P_dbH  ]

        # E1 =      E1_0(x[i],a_out,da_out_dx,d2a_out_dx2,y_trial)
        # E1 = E1 + E1_1(x[i],a_out,da_out_dx,d2a_out_dx2,y_trial)
        # E1 = E1 + E1_2(x[i],a_out,da_out_dx,d2a_out_dx2,y_trial) 
        # E1 = E1 + E1_3(x[i],a_out,da_out_dx,d2a_out_dx2,y_trial)
        # E1 = E1 + E1_4(x[i],a_out,da_out_dx,d2a_out_dx2,y_trial)
        E1 = d2ytrial_dx2 + a(x[i]) * dytrial_dx + b(x[i]) * y_trial - f(x[i],y_trial)
        # error of layers
        # output layer
        # err_wO1 = grad_N0 * grad_N1 * sig(z_H)
        # err_wO2 = grad_N0 * grad_N2 * (dsig(z_H) .* w_H)
        # err_wO = err_wO1 .+ err_wO2 
        kk=3
        err_wO = 2 * E1 *  dE1[1:kk]' *   da_out_dwO[1:kk]
        # err_out = grad_N*dsig(z_out);
        # err_out = grad_N *sig(z_H)
        # err_out = err_out + x[i]* dsig(z_H) .* w_H
        # hidden layer
        # err_wH1 = grad_N0 * grad_N1 * x[i] * w_out.*dsig(z_H)
        # err_wH2 = grad_N0 * grad_N2 * (w_out .* dsig(z_H) .+ x[i] * w_out .* d2sig(z_H) .* w_H)
        # err_wH = err_wH1 .+ err_wH2
        err_wH = 2 * E1 *  dE1[1:kk]' *   da_out_dwH[1:kk] 
        ###
        # err_bH1 = grad_N0 * grad_N
        err_bH = 2 * E1 *  dE1[1:kk]' *   da_out_dbH[1:kk]


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