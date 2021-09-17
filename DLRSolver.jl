__precompile__
include("quadrature.jl")
include("Basis.jl")
include("TimeSolver.jl")

using ProgressMeter
using LinearAlgebra
using Einsum

struct DLRSolver
    # spatial grid of cell interfaces
    x;

    # quadrature
    q::Quadrature;

    # DLRSolver settings
    settings::Settings;

    # spatial basis functions
    basis::Basis;

    # low-rank solution matrices
    X::Array{Float64,2}
    W::Array{Float64,2}
    S::Array{Float64,2}

    # preallocated matrices for Rhs
    A::Array{Float64,3}
    B::Array{Float64,3}
    Y::Array{Float64,3}
    Y1::Array{Float64,2}
    ACons::Array{Float64,3}

    fluxS::Array{Float64,2}
    fluxL::Array{Float64,2}

    yL::Array{Float64,2}
    yS::Array{Float64,2}
    yK::Array{Float64,2}

    # Dirichlet BCs
    uL::Array{Float64,1}
    uR::Array{Float64,1}

    # time solver
    rkUpdate::TimeSolver

    # tridiagonal stencil matrices
    L1I::Tridiagonal{Float64, Vector{Float64}};
    L2::Tridiagonal{Float64, Vector{Float64}};

    # constructor
    function DLRSolver(settings)
        x = settings.x;
        r = settings.r;
        q = Quadrature(settings.Nq,"Gauss");
        basis = Basis(q,settings);

        # note that these are actually the hat variables
        X = zeros(settings.Nx,r)
        S = zeros(r,r)
        W = zeros(settings.N,r) 

        A = zeros(r,r,r);
        B = zeros(r,r,settings.Nq^2);
        Y = zeros(r,r,r);
        Y1 = zeros(r,r);

        fluxS = zeros(r,r);
        fluxL = zeros(r,settings.N^2);
        yL = zeros(r,settings.N^2);
        yS = zeros(r,r);
        yK = zeros(settings.Nx,r);

        uL = zeros(settings.Nq^2);
        uR = zeros(settings.Nq^2);

        ACons = zeros(settings.NCons,settings.N^2,settings.N^2);

        rkUpdate = TimeSolver(settings);

        onesBC = ones(settings.Nx);
        onesBC[1] = 0.0; onesBC[end] = 0.0;
        onesBCL = ones(settings.Nx-1);
        onesBCL[end] = 0.0;
        onesBCR = ones(settings.Nx-1);
        onesBCR[1] = 0.0;

        L1I = 0.5*Tridiagonal(onesBCL,-2*onesBC,onesBCR)/settings.dt;
        L2 = 0.25*Tridiagonal(-onesBCL,zeros(settings.Nx),onesBCR)/settings.dx;

        new(x,q,settings,basis,X,S,W,A,B,Y,Y1,ACons,fluxS,fluxL,yL,yS,yK,uL,uR,rkUpdate,L1I,L2);
    end
end

function projector(u::Array{Float64,1},a::Array{Float64,1})
    factor = u'a/(u'u);
    return  factor .* u;
end

function qrGramSchmidt(A::Array{Float64,2})
    XNew,S = qr(A);
    XNew = Matrix(XNew);
    S = Matrix(S);
    r = size(S,2);
    X = XNew[:,1:r]
    S = S[1:r,1:r]
    return X,S;
    N = size(A,1);
    r = size(A,2);

    Q = zeros(N,r);
    R = zeros(r,r);

    for k = 1:r
        Q[:,k] .= A[:,k];
        for j = 1:k-1
            Q[:,k] .-= projector(Q[:,j],A[:,k]);
        end
    end

    for k = 1:r
        Q[:,k] ./= sqrt(Q[:,k]'Q[:,k]); # normalize
    end

    for k = 1:r
        for j = r:-1:k
            R[k,j] = Q[:,k]'A[:,j];
        end
    end

    return Q,R;

end

function RhsK(obj::DLRSolver,K::Array{Float64,2},W::Array{Float64,2})
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq;
    N = obj.settings.N;
    fXi = 0.25;
    dt = obj.settings.dt;
    flux = zeros(r);
    dx = obj.settings.dx;

    WQuad = EvalAtQuad(obj.basis,W)';

    # Compute A_{i,j,m} = E[W_i W_j W_m]
    WQuad = EvalAtQuad(obj.basis,W)';
    for i = 1:r
        for j = 1:r
            for m = 1:r
                obj.A[i,j,m] = Integral(obj.q,WQuad[i,:].*WQuad[j,:].*WQuad[m,:].*fXi);
            end
        end
    end

    for j = 2:(Nx-1)
        for p = 1:r
            flux[p] = 0.0;
            for l = 1:r
                for m = 1:r
                    flux[p] += 1/4/dx * (K[j+1,l]*K[j+1,m]-K[j-1,l]*K[j-1,m])*obj.A[l,m,p];
                end
            end
        end
        for p = 1:r
            obj.yK[j,p] = 1/2/dt .* (K[j+1,p]-2*K[j,p]+K[j-1,p]) - flux[p];
        end
    end
    return obj.yK;
end

function RhsS(obj::DLRSolver,X::Array{Float64,2},W::Array{Float64,2},L::Array{Float64,2})
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2;
    fXi = 0.25;
    dt = obj.settings.dt;
    # Compute A_{i,j,m} = E[L_i L_j phi_m]
    LQuad = EvalAtQuad(obj.basis,L)';
    WQuad = EvalAtQuad(obj.basis,W)';
    for i = 1:r
        for j = 1:r
            for m = 1:r
                obj.A[i,j,m] = Integral(obj.q,LQuad[i,:].*LQuad[j,:].*WQuad[m,:].*fXi);
            end
        end
    end

    for p = 1:r
        for l = 1:r
            obj.Y1[p,l] = 0.0;
            for j = 2:(Nx-1)
                if obj.settings.stabilization != 1
                    if obj.settings.stabilization == 2
                        obj.Y1[p,l] += -1/2/dt * X[j,p]*(X[j+1,l]-2*X[j,l]+X[j-1,l]);# DLR first, discretize second; stable version for projector splitting
                    else
                        obj.Y1[p,l] += 1/2/dt * X[j,p]*(X[j+1,l]-2*X[j,l]+X[j-1,l]); # discretize first, DLR second
                    end
                end
            end
        end
    end

    for p = 1:r
        for m = 1:r
            for l = 1:r
                obj.Y[m,l,p] = 0;
                for j = 2:(Nx-1) 
                    obj.Y[m,l,p] += 1/4/obj.settings.dx * X[j,p]*(X[j+1,l]*X[j+1,m]-X[j-1,l]*X[j-1,m]);
                end
            end
        end
    end

    for q = 1:r
        for l = 1:r
            obj.fluxS[q,l] = 0.0;
            for p = 1:r
                for m = 1:r
                    obj.fluxS[q,l] += obj.Y[m,p,q]*obj.A[m,p,l];
                end
            end
        end
        for l = 1:r
            obj.yS[q,l] = 0.0
            for m = 1:r
                for i = 1:N
                    obj.yS[q,l] += obj.Y1[q,m]*L[i,m]*W[i,l];
                end
            end
            obj.yS[q,l] -= obj.fluxS[q,l];
        end
    end
    return obj.yS;

end

function RhsL(obj::DLRSolver,X::Array{Float64,2},L::Array{Float64,2},recompute::Bool=true)
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2;
    fXi = 0.25;
    # Compute A_{i,j,m} = E[L_i L_j phi_m]
    LQuad = EvalAtQuad(obj.basis,L)';
    for i = 1:r
        for j = 1:r
            for m = 1:N
                obj.B[i,j,m] = Integral(obj.q,LQuad[i,:].*LQuad[j,:].*obj.basis.PhiQuad[:,m].*fXi);
            end
        end
    end

    if recompute
        for p = 1:r
            for m = 1:r
                for l = 1:r
                    obj.Y[m,l,p] = 0;
                    for j = 2:(Nx-1) 
                        obj.Y[m,l,p] += 1/4/obj.settings.dx * X[j,p]*(X[j+1,l]*X[j+1,m]-X[j-1,l]*X[j-1,m]);
                    end
                end
            end
        end

        for p = 1:r
            for l = 1:r
                obj.Y1[p,l] = 0.0;
                for j = 2:(Nx-1)
                    if obj.settings.stabilization != 1
                        obj.Y1[p,l] += 1/2/obj.settings.dt * X[j,p]*(X[j+1,l]-2*X[j,l]+X[j-1,l]);
                    end
                end
            end
        end
    end
    
    for p = 1:r
        for i = 1:N
            obj.fluxL[p,i] = 0.0;
            for l = 1:r
                for m = 1:r
                    obj.fluxL[p,i] += obj.Y[m,l,p]*obj.B[m,l,i];
                end
            end
        end
    end

    for p = 1:r
        for i = 1:N
            obj.yL[p,i] = 0.0;
            for l = 1:r
                obj.yL[p,i] += obj.Y1[p,l]*L[i,l];
            end
            obj.yL[p,i] -= obj.fluxL[p,i];
        end
    end

    return obj.yL;
end

function SetupIC(obj::DLRSolver)
    u = zeros(obj.settings.N*obj.settings.N,obj.settings.Nx);
    uCons = zeros(obj.settings.Nx,obj.settings.NCons);
    uVals = zeros(obj.settings.Nq^2)
    for j = 1:obj.settings.Nx
        for q = 1:obj.settings.Nq
            for k = 1:obj.settings.Nq
                uVals[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[j],obj.q.xi[k],obj.q.xi[q])[1];
            end
        end
        u[:,j] = ComputeMomentsDLR(obj.basis,uVals*0.25);
        uCons[j,:] = ComputeMomentsCons(obj.basis,uVals*0.25);
    end
    return u,uCons;
end

function FactorIC(obj::DLRSolver)
    r = obj.settings.r;
    # Set up initial condition
    u, uCons = SetupIC(obj);
    u = Array(u');

    # Low-rank approx of init data:
    X,S,W = svd(u); 

    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Array(Diagonal(S));
    S = S[1:r, 1:r]; 
    return X,S,W,uCons;
end

# matrix projector-splitting integrator
function SolveBackward(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points

    fXi = 0.25;

    # Set up initial condition
    u,uCons = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u'); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    K1 = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N,r);
    L1 = zeros(N,r);
    S1 = zeros(r,r);
    
    Nt = Integer(round(tEnd/dt));

    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    #@gif 
    for n = 1:Nt

        ###### K-step ######
        K .= X*S;

        WQuad = EvalAtQuad(obj.basis,W)';

        # impose BCs
        for i = 1:r
            K[1,i] = Integral(obj.q,obj.uL.*WQuad[i,:].*fXi);
            K[end,i] = Integral(obj.q,obj.uR.*WQuad[i,:].*fXi);
        end
        
        if obj.settings.rkType == "Heun"
            K1 .= K .+ dt*RhsK(obj,K,W);
            K1 .= K1 .+ dt*RhsK(obj,K1,W);
            K .= 0.5.*(K.+K1);
        elseif obj.settings.rkType == "Euler"
            K .= K .+ dt*RhsK(obj,K,W);
        elseif obj.settings.rkType == "SSP"
            K .= UpdateK(obj.rkUpdate,obj,K,W);
        end

        X,S = qr(K); # optimize by choosing XFull, SFull
        X = X[:, 1:obj.settings.r]; 
        S = S[1:obj.settings.r, 1:obj.settings.r];

        ###### S-step ######

        L .= W*S';        
        if obj.settings.rkType == "Heun"
            S1 .= S .- dt.*RhsS(obj,X,W,L);
            L .= W*S';
            S1 .= S1 .- dt.*RhsS(obj,X,W,L);
            S .= 0.5.*(S.+S1);
        elseif obj.settings.rkType == "Euler"
            S .= S .- dt.*RhsS(obj,X,W,L);
        elseif obj.settings.rkType == "SSP"
            S .= UpdateS(obj.rkUpdate,obj,X,S,W);
        end

        ###### L-step ######
        L .= W*S';

        if obj.settings.rkType == "Heun"
            L1 .= L .+ dt*RhsL(obj,X,L,false)';
            L1 .= L1 .+ dt*RhsL(obj,X,L1,false)';
            L .= 0.5.*(L.+L1);
        elseif obj.settings.rkType == "Euler"
            L .= L .+ dt*RhsL(obj,X,L,false)';
        elseif obj.settings.rkType == "SSP"
            L .= UpdateL(obj.rkUpdate,obj,X,L,false);
        end
                
        W,S = qr(L);
        #W,S = qr(L);
        W = W[:, 1:obj.settings.r];
        S = S[1:obj.settings.r, 1:obj.settings.r];

        S .= S';

        # apply filter step
        Filter(obj,W);
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t, X,S,W;
end

# unconventional integrator
function SolveForward(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points

    fXi = 0.25;

    # Set up initial condition
    u,uCons = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u'); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Array(Diagonal(S));
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    K1 = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N,r);
    L1 = zeros(N,r);
    S1 = zeros(r,r);

    PhiQuad = obj.basis.PhiQuad;
    
    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    #@gif 
    for n = 1:Nt

        ################## K-step ##################
        K .= X*S;

        # impose BCs
        WQuad = EvalAtQuad(obj.basis,W)';
        for i = 1:r
            K[1,i] = Integral(obj.q,obj.uL.*WQuad[i,:].*fXi);
            K[end,i] = Integral(obj.q,obj.uR.*WQuad[i,:].*fXi);
        end
        
        if obj.settings.rkType == "Heun"
            K1 .= K .+ dt*RhsK(obj,K,W);
            K1 .= K1 .+ dt*RhsK(obj,K1,W);
            K .= 0.5.*(K.+K1);
        elseif obj.settings.rkType == "Euler"
            K .= K .+ dt*RhsK(obj,K,W);
        elseif obj.settings.rkType == "SSP"
            K .= UpdateK(obj.rkUpdate,obj,K,W);
        end

        XNew,STmp = qrGramSchmidt(K);

        MUp = XNew' * X;

        ################## L-step ##################
        L .= W*S';

        if obj.settings.rkType == "Heun"
            L1 .= L .+ dt*RhsL(obj,X,L)';
            L1 .= L1 .+ dt*RhsL(obj,X,L1)';
            L .= 0.5.*(L.+L1);
        elseif obj.settings.rkType == "Euler"
            L .= L .+ dt*RhsL(obj,X,L)';
        elseif obj.settings.rkType == "SSP"
            L .= UpdateL(obj.rkUpdate,obj,X,L);
        end
                
        WNew,STmp = qrGramSchmidt(L);

        NUp = WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        L = W*S';

        if obj.settings.rkType == "Heun"
            S1 .= S .+ dt.*RhsS(obj,X,W,L);
            L .= W*S';
            S1 .= S1 .+ dt.*RhsS(obj,X,W,L);
            S .= 0.5.*(S.+S1);
        elseif obj.settings.rkType == "Euler"
            S .= S .+ dt.*RhsS(obj,X,W,L);
        elseif obj.settings.rkType == "SSP"
            S .= UpdateS(obj.rkUpdate,obj,X,S,W,false);
        end

        # apply filter step
        Filter(obj,W);
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    #println(norm(uHat))

    # return end time and solution
    return t, X,S,W;
end

function RhsNodal(obj::DLRSolver,uQ::Array{Float64,2})
    rhs = zeros(size(uQ))
    Nx = obj.settings.Nx;
    for j = 2:(Nx-1)
        for k = 1:obj.settings.Nq^2
            rhs[j,k] = 0.5*(uQ[j+1,k]-2*uQ[j,k]+uQ[j-1,k])/obj.settings.dt - 0.25/obj.settings.dx*(uQ[j+1,k]^2-uQ[j-1,k]^2);
        end
    end
    return rhs;
end

function RhsNodalSplit(obj::DLRSolver,uQ::Array{Float64,2})
    rhs = zeros(size(uQ))
    Nx = obj.settings.Nx;


    for j = 2:(Nx-1)
        for k = 1:obj.settings.Nq^2
            rhs[j,k] = 0.5*(uQ[j+1,k]-2*uQ[j,k]+uQ[j-1,k])/obj.settings.dt - 0.25/obj.settings.dx*(uQ[j+1,k]^2-uQ[j-1,k]^2);
        end
    end
    return rhs;
end

function SolveNaiveSplit(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points

    fXi = 0.25;

    # save basis
    PhiQuad = Array(obj.basis.PhiQuad');
    PhiQuadCons = Array(obj.basis.PhiQuadCons');
    PhiQuadW = Array(obj.basis.PhiQuadW');
    PhiQuadWCons = Array(obj.basis.PhiQuadWCons');

    # Set up initial condition
    u, uCons = SetupIC(obj);
    u = Array(u');
    uQ = u*PhiQuad;
    #uNew = u;

    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    for n = 1:Nt

        uQ = u*PhiQuad + uCons*PhiQuadCons;

        println(maximum(dt*RhsNodal(obj,uQ)))

        #uQ = uQ .+ dt*RhsNodal(obj,uQ);

        uCons = uCons .+ dt*RhsNodalSplit(obj,uQ)*fXi*PhiQuadWCons;

        u = u .+ dt*RhsNodalSplit(obj,uQ)*fXi*PhiQuadW;

        next!(prog) # update progress bar

        t = t+dt;
    end

    # compute moments
    uQ = u*PhiQuad + uCons*PhiQuadCons;
    u = uQ*Array(obj.basis.PhiQuadWFull')*fXi;

    # return end time and solution
    return t, u;
    #return t, uQ*fXi*PhiQuadW;
end

function SolveNaiveSplitUnconventionalIntegrator(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points
    r = obj.settings.r;

    fXi = 0.25;

    # save basis
    PhiQuad = Array(obj.basis.PhiQuad');
    PhiQuadCons = Array(obj.basis.PhiQuadCons');
    PhiQuadW = Array(obj.basis.PhiQuadW');
    PhiQuadWCons = Array(obj.basis.PhiQuadWCons');

    # Set up initial condition
    u, uCons = SetupIC(obj);
    u = Array(u');
    uQ = u*PhiQuad;

    # Low-rank approx of init data:
    X,S,W = svd(u); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Array(Diagonal(S));
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    K1 = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N,r);
    L1 = zeros(N,r);
    S1 = zeros(r,r);

    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    for n = 1:Nt

        # reconstruct solution (inefficient)
        uQ = X*S*W'*PhiQuad + uCons*PhiQuadCons;

        # update conservative part (inefficient)
        uCons = uCons .+ dt*RhsNodalSplit(obj,uQ)*fXi*PhiQuadWCons;


        ###### K-step ######
        K .= X*S;
        WQuadW = PhiQuadW*W; # compute W_{ki}*w_k , where W \in R^{Nq x r}

        #K .= K .+ dt*Integral(obj.q,F(obj,K*EvalAtQuadDLR(obj.basis,W))*EvalAtQuadDLR(obj.basis,W).*fXi);
        K .= K .+ dt*RhsNodalSplit(obj,uQ)*WQuadW.*fXi;

        XNew,STmp = qrGramSchmidt(K); # optimize bei choosing XFull, SFull

        MUp = XNew' * X;

        ###### L-step ######
        L = W*S';

        #L .= L .+ dt*(X'*F(obj,X*EvalAtQuadDLR(obj.basis,L)))';
        L .= L .+ dt*(X'*RhsNodalSplit(obj,uQ)*PhiQuadW.*fXi)';
                
        WNew,STmp = qrGramSchmidt(L);

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        uQ = X*S*W'*PhiQuad + uCons*PhiQuadCons;
        WQuadW = PhiQuadW*W;

        S .= S .+ dt.*X'*RhsNodalSplit(obj,uQ)*WQuadW.*fXi;
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    # compute moments
    uQ = X*S*W'*PhiQuad + uCons*PhiQuadCons;
    u = uQ*Array(obj.basis.PhiQuadWFull')*fXi;

    # return end time and solution
    return t, u;
end

function SolveSplitUnconventionalIntegrator(obj::DLRSolver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2},uCons::Array{Float64,2})
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    N = obj.settings.N^2; # here, N is the number of quadrature points
    r = obj.settings.r;
    NCons = obj.settings.NCons
    w = obj.q.wTens;

    fXi = 0.25;

    # save basis
    PhiQuad = Array(obj.basis.PhiQuad');
    PhiQuadCons = Array(obj.basis.PhiQuadCons');
    PhiQuadW = Array(obj.basis.PhiQuadW');

    K = zeros(Nx,r);
    L = zeros(N,r);

    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)

    Y = zeros(r,r,r);
    YCons = zeros(NCons,NCons,r);
    YCross = zeros(NCons,r,r);

    Y1 = zeros(r,r,r);
    Y1Cons = zeros(NCons,NCons,r);
    Y1Cross = zeros(NCons,r,r);
    Y1ConsFull = zeros(NCons,NCons,NCons);

    fluxK1 = zeros(Nx,r)
    fluxK2 = zeros(Nx,r)
    fluxK3 = zeros(Nx,r)

    fluxL1 = zeros(r,N^2)
    fluxL2 = zeros(r,N^2)
    fluxL3 = zeros(r,N^2)

    fluxS1 = zeros(r,N^2)
    fluxS2 = zeros(r,N^2)
    fluxS3 = zeros(r,N^2)

    fluxCons1 = zeros(Nx,NCons)
    fluxCons2 = zeros(Nx,NCons)
    fluxCons3 = zeros(Nx,NCons)
    
    # time loop
    for n = 1:Nt

        # reconstruct solution (inefficient)
        WQuad = PhiQuad'*W; # compute W_{ki}*w_k , where W \in R^{Nq x r}
        K .= X*S;

        # update conservative part (inefficient)
        @einsum Y1Cross[l,m,i] := WQuad[k,m]*WQuad[k,i]*PhiQuadCons[l,k]*w[k].*fXi
        @einsum Y1Cons[l,m,i] := PhiQuadCons[m,k]*WQuad[k,i]*PhiQuadCons[l,k]*w[k].*fXi
        @einsum Y1ConsFull[l,m,i] := PhiQuadCons[m,k]*PhiQuadCons[i,k]*PhiQuadCons[l,k]*w[k].*fXi

        @einsum fluxCons1[j,l] := K[j,m]*Y1Cross[l,m,i]*K[j,i];
        @einsum fluxCons2[j,l] := K[j,i]*Y1Cons[l,m,i]*uCons[j,m];
        @einsum fluxCons3[j,l] := uCons[j,m]*Y1ConsFull[l,m,i]*uCons[j,i];

        uConsNew = uCons .+ dt*(obj.L1I*uCons .- obj.L2*(fluxCons1+2.0*fluxCons2+fluxCons3));

        ###### K-step ######
        @einsum Y[l,m,i] := WQuad[k,l]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi
        @einsum YCross[l,m,i] := PhiQuadCons[l,k]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi
        @einsum YCons[l,m,i] := PhiQuadCons[l,k]*PhiQuadCons[m,k]*WQuad[k,i]*w[k].*fXi

        # value uQ^2*WQuadW.*fXi
        @einsum fluxK1[j,i] := K[j,l]*Y[l,m,i]*K[j,m]
        @einsum fluxK2[j,i] := uCons[j,l]*YCross[l,m,i]*K[j,m]
        @einsum fluxK3[j,i] := uCons[j,l]*YCons[l,m,i]*uCons[j,m]

        K .= K .+ dt*(obj.L1I*K .- obj.L2*(fluxK1.+2*fluxK2.+fluxK3));

        XNew,STmp = qrGramSchmidt(K); # optimize bei choosing XFull, SFull

        MUp = XNew' * X;

        ###### L-step ######
        L = W*S';
        LQuad = PhiQuad'*L;

        XL1X = X'*obj.L1I*X;
        XL2 = X'*obj.L2

        @einsum Y[l,m,i] := XL2[i,j]*X[j,l]*X[j,m];
        @einsum YCross[l,m,i] := XL2[i,j]*uCons[j,l]*X[j,m];
        @einsum YCons[l,m,i] := XL2[i,j]*uCons[j,l]*uCons[j,m];

        @einsum fluxL1[i,k] := Y[l,m,i]*LQuad[k,l]*LQuad[k,m]
        @einsum fluxL2[i,k] := YCross[l,m,i]*PhiQuadCons[l,k]*LQuad[k,m]
        @einsum fluxL3[i,k] := YCons[l,m,i]*PhiQuadCons[l,k]*PhiQuadCons[m,k]

        L .= L .+ dt*(XL1X*L' .- (fluxL1 .+ 2*fluxL2 .+ fluxL3)*PhiQuadW.*fXi)';
                
        WNew,STmp = qrGramSchmidt(L);

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        WQuad = PhiQuad'*W;
        K .= X*S;

        @einsum Y[l,m,i] := WQuad[k,l]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi
        @einsum YCross[l,m,i] := PhiQuadCons[l,k]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi
        @einsum YCons[l,m,i] := PhiQuadCons[l,k]*PhiQuadCons[m,k]*WQuad[k,i]*w[k].*fXi

        XL2 = X'*obj.L2
        @einsum Y1[l,m,i] := XL2[i,j]*K[j,l]*K[j,m];
        @einsum Y1Cross[l,m,i] := XL2[i,j]*uConsNew[j,l]*K[j,m];
        @einsum Y1Cons[l,m,i] := XL2[i,j]*uConsNew[j,l]*uConsNew[j,m];

        @einsum fluxS1[j,i] := Y[l,m,i]*Y1[l,m,j];
        @einsum fluxS2[j,i] := YCross[l,m,i]*Y1Cross[l,m,j];
        @einsum fluxS3[j,i] := YCons[l,m,i]*Y1Cons[l,m,j];

        S .= S .+ dt.*(XL1X*S .- (fluxS1.+2*fluxS2.+fluxS3));
        
        uCons .= uConsNew
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t, X,S,W,uCons;
end

function SolveSplitProjectorSplittingIntegrator(obj::DLRSolver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2},uCons::Array{Float64,2})
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    N = obj.settings.N^2; # here, N is the number of quadrature points
    r = obj.settings.r;
    NCons = obj.settings.NCons
    w = obj.q.wTens;

    fXi = 0.25;

    # save basis
    PhiQuad = Array(obj.basis.PhiQuad');
    PhiQuadCons = Array(obj.basis.PhiQuadCons');
    PhiQuadW = Array(obj.basis.PhiQuadW');

    K = zeros(Nx,r);
    L = zeros(N,r);

    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)

    Y = zeros(r,r,r);
    YCons = zeros(NCons,NCons,r);
    YCross = zeros(NCons,r,r);

    Y1 = zeros(r,r,r);
    Y1Cons = zeros(NCons,NCons,r);
    Y1Cross = zeros(NCons,r,r);
    Y1ConsFull = zeros(NCons,NCons,NCons);

    fluxK1 = zeros(Nx,r)
    fluxK2 = zeros(Nx,r)
    fluxK3 = zeros(Nx,r)

    fluxL1 = zeros(r,N^2)
    fluxL2 = zeros(r,N^2)
    fluxL3 = zeros(r,N^2)

    fluxS1 = zeros(r,N^2)
    fluxS2 = zeros(r,N^2)
    fluxS3 = zeros(r,N^2)

    fluxCons1 = zeros(Nx,NCons)
    fluxCons2 = zeros(Nx,NCons)
    fluxCons3 = zeros(Nx,NCons)
    
    # time loop
    for n = 1:Nt

        # reconstruct solution (inefficient)
        WQuad = PhiQuad'*W; # compute W_{ki}*w_k , where W \in R^{Nq x r}
        K .= X*S;

        # update conservative part (inefficient)
        @einsum Y1Cross[l,m,i] := WQuad[k,m]*WQuad[k,i]*PhiQuadCons[l,k]*w[k].*fXi
        @einsum Y1Cons[l,m,i] := PhiQuadCons[m,k]*WQuad[k,i]*PhiQuadCons[l,k]*w[k].*fXi
        @einsum Y1ConsFull[l,m,i] := PhiQuadCons[m,k]*PhiQuadCons[i,k]*PhiQuadCons[l,k]*w[k].*fXi

        @einsum fluxCons1[j,l] := K[j,m]*Y1Cross[l,m,i]*K[j,i];
        @einsum fluxCons2[j,l] := K[j,i]*Y1Cons[l,m,i]*uCons[j,m];
        @einsum fluxCons3[j,l] := uCons[j,m]*Y1ConsFull[l,m,i]*uCons[j,i];

        uConsNew = uCons .+ dt*(obj.L1I*uCons .- obj.L2*(fluxCons1+2.0*fluxCons2+fluxCons3));

        ###### K-step ######
        @einsum Y[l,m,i] := WQuad[k,l]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi
        @einsum YCross[l,m,i] := PhiQuadCons[l,k]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi
        @einsum YCons[l,m,i] := PhiQuadCons[l,k]*PhiQuadCons[m,k]*WQuad[k,i]*w[k].*fXi

        # value uQ^2*WQuadW.*fXi
        @einsum fluxK1[j,i] := K[j,l]*Y[l,m,i]*K[j,m]
        @einsum fluxK2[j,i] := uCons[j,l]*YCross[l,m,i]*K[j,m]
        @einsum fluxK3[j,i] := uCons[j,l]*YCons[l,m,i]*uCons[j,m]

        K .= K .+ dt*(obj.L1I*K .- obj.L2*(fluxK1.+2*fluxK2.+fluxK3));

        X,S = qrGramSchmidt(K); # optimize bei choosing XFull, SFull

        ################## S-step ##################

        @einsum Y[l,m,i] := WQuad[k,l]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi
        @einsum YCross[l,m,i] := PhiQuadCons[l,k]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi
        @einsum YCons[l,m,i] := PhiQuadCons[l,k]*PhiQuadCons[m,k]*WQuad[k,i]*w[k].*fXi

        XL1X = X'*obj.L1I*X;
        XL2 = X'*obj.L2
        @einsum Y1[l,m,i] := XL2[i,j]*K[j,l]*K[j,m];
        @einsum Y1Cross[l,m,i] := XL2[i,j]*uConsNew[j,l]*K[j,m];
        @einsum Y1Cons[l,m,i] := XL2[i,j]*uConsNew[j,l]*uConsNew[j,m];

        @einsum fluxS1[j,i] := Y[l,m,i]*Y1[l,m,j];
        @einsum fluxS2[j,i] := YCross[l,m,i]*Y1Cross[l,m,j];
        @einsum fluxS3[j,i] := YCons[l,m,i]*Y1Cons[l,m,j];

        S .= S .- dt.*(XL1X*S .- (fluxS1.+2*fluxS2.+fluxS3));

        ###### L-step ######
        L = W*S';
        LQuad = PhiQuad'*L;

        @einsum Y[l,m,i] := XL2[i,j]*X[j,l]*X[j,m];
        @einsum YCross[l,m,i] := XL2[i,j]*uCons[j,l]*X[j,m];
        @einsum YCons[l,m,i] := XL2[i,j]*uCons[j,l]*uCons[j,m];

        @einsum fluxL1[i,k] := Y[l,m,i]*LQuad[k,l]*LQuad[k,m]
        @einsum fluxL2[i,k] := YCross[l,m,i]*PhiQuadCons[l,k]*LQuad[k,m]
        @einsum fluxL3[i,k] := YCons[l,m,i]*PhiQuadCons[l,k]*PhiQuadCons[m,k]

        L .= L .+ dt*(XL1X*L' .- (fluxL1 .+ 2*fluxL2 .+ fluxL3)*PhiQuadW.*fXi)';
                
        W,S = qrGramSchmidt(L);
        S .= S';
        
        uCons .= uConsNew
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t, X,S,W,uCons;
end

function SolveUnconventionalIntegrator(obj::DLRSolver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    N = obj.settings.N^2; # here, N is the number of quadrature points
    r = obj.settings.r;
    w = obj.q.wTens;

    fXi = 0.25;

    K = zeros(Nx,r);
    L = zeros(N,r);

    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)

    Y = zeros(r,r,r);

    Y1 = zeros(r,r,r);

    fluxK1 = zeros(Nx,r)

    fluxL1 = zeros(r,N^2)

    fluxS1 = zeros(r,N^2)
    
    # time loop
    for n = 1:Nt

        WQuad = obj.basis.PhiQuad*W; # compute W_{ki}*w_k , where W \in R^{Nq x r}
        K .= X*S;

        ###### K-step ######
        @einsum Y[l,m,i] := WQuad[k,l]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi

        # value uQ^2*WQuadW.*fXi
        @einsum fluxK1[j,i] := K[j,l]*Y[l,m,i]*K[j,m]

        K .= K .+ dt*(obj.L1I*K .- obj.L2*fluxK1);

        XNew,STmp = qrGramSchmidt(K); # optimize bei choosing XFull, SFull

        MUp = XNew' * X;

        ###### L-step ######
        L = W*S';
        LQuad = obj.basis.PhiQuad*L;

        XL1X = X'*obj.L1I*X;
        XL2 = X'*obj.L2

        @einsum Y[l,m,i] := XL2[i,j]*X[j,l]*X[j,m];

        @einsum fluxL1[i,k] := Y[l,m,i]*LQuad[k,l]*LQuad[k,m]

        L .= L .+ dt*(XL1X*L' .- fluxL1*obj.basis.PhiQuadW'.*fXi)';
                
        WNew,STmp = qrGramSchmidt(L);

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        WQuad = obj.basis.PhiQuad*W;
        K .= X*S;

        @einsum Y[l,m,i] := WQuad[k,l]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi

        XL2 = X'*obj.L2
        @einsum Y1[l,m,i] := XL2[i,j]*K[j,l]*K[j,m];

        @einsum fluxS1[j,i] := Y[l,m,i]*Y1[l,m,j];

        S .= S .+ dt.*(XL1X*S .- fluxS1);
                
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t, X,S,W;
end

function SolveProjectorSplittingIntegrator(obj::DLRSolver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    N = obj.settings.N^2; # here, N is the number of quadrature points
    r = obj.settings.r;
    w = obj.q.wTens;

    fXi = 0.25;

    K = zeros(Nx,r);
    L = zeros(N,r);

    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)

    Y = zeros(r,r,r);

    Y1 = zeros(r,r,r);

    fluxK1 = zeros(Nx,r)

    fluxL1 = zeros(r,N^2)

    fluxS1 = zeros(r,N^2)
    
    # time loop
    for n = 1:Nt

        WQuad = obj.basis.PhiQuad*W; # compute W_{ki}*w_k , where W \in R^{Nq x r}
        K .= X*S;

        # impose BCs
        for i = 1:r
            K[1,i] = Integral(obj.q,obj.uL.*WQuad[:,i].*fXi);
            K[end,i] = Integral(obj.q,obj.uR.*WQuad[:,i].*fXi);
        end

        ###### K-step ######
        @einsum Y[l,m,i] := WQuad[k,l]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi

        # value uQ^2*WQuadW.*fXi
        @einsum fluxK1[j,i] := K[j,l]*Y[l,m,i]*K[j,m]

        K .= K .+ dt*(obj.L1I*K .- obj.L2*fluxK1);

        X,S = qrGramSchmidt(K); # optimize bei choosing XFull, SFull

        ################## S-step ##################

        @einsum Y[l,m,i] := WQuad[k,l]*WQuad[k,m]*WQuad[k,i]*w[k].*fXi

        XL1X = X'*obj.L1I*X;
        XL2 = X'*obj.L2
        @einsum Y1[l,m,i] := XL2[i,j]*K[j,l]*K[j,m];

        @einsum fluxS1[j,i] := Y[l,m,i]*Y1[l,m,j];

        S .= S .- dt.*(XL1X*S .- fluxS1);

        ###### L-step ######
        L = W*S';
        LQuad = obj.basis.PhiQuad*L;

        @einsum Y[l,m,i] := XL2[i,j]*X[j,l]*X[j,m];

        @einsum fluxL1[i,k] := Y[l,m,i]*LQuad[k,l]*LQuad[k,m]

        L .= L .+ dt*(XL1X*L' .- fluxL1*obj.basis.PhiQuadW'.*fXi)';
                
        W,S = qrGramSchmidt(L);
        S .= S';
                
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t, X,S,W;
end

function Filter(obj::DLRSolver,W::Array{Float64,2})
    lambda = obj.settings.lambda
    N = obj.settings.N;
    if obj.settings.filterType == "L2"
        for i = 1:N
            for l = 1:N
                W[(l-1)*N+i,:] .= W[(l-1)*N+i,:]/(1+lambda*(i-1)^2*i^2+lambda*(l-1)^2*l^2);
            end
        end
    elseif obj.settings.filterType == "EXP"
        epsilonM = eps(Float64);
        c = log( epsilonM );
        for i = 1:size(W,1)
            eta = i/(obj.settings.N+1)
            W[i,:] .= W[i,:]*exp( c * eta^obj.settings.filterOrder )^(lambda*obj.settings.dt);
        end
    end
end

# SSP Update function for K-step
function UpdateK(obj::TimeSolver,solver::DLRSolver,K::Array{Float64,2},W::Array{Float64,2})
    NCells = solver.settings.Nx;
    obj.KRK[1,:,:] .= K;
    for s = 1:obj.rkStages
        obj.Krhs[s,:,:] .= RhsK(solver,obj.KRK[s,:,:],W);;
        obj.KRK[s+1,:,:] .= zeros(NCells,solver.settings.r);
        for j = 1:s
            obj.KRK[s+1,:,:] = obj.KRK[s+1,:,:]+obj.alpha[s,j].*obj.KRK[j,:,:]+obj.dt*obj.beta[s,j].*obj.Krhs[j,:,:];
        end
    end

    return obj.KRK[obj.rkStages+1,:,:];
end

# SSP Update function for S-step
function UpdateS(obj::TimeSolver,solver::DLRSolver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2},backward::Bool=true)
    obj.SRK[1,:,:] .= S;
    for s = 1:obj.rkStages
        obj.Srhs[s,:,:] .= RhsS(solver,X,W,W*obj.SRK[s,:,:]');
        if backward # if projector splitting integrator is used, sign in rhsS must be changed
            obj.Srhs[s,:,:] .= -obj.Srhs[s,:,:];
        end
        obj.SRK[s+1,:,:] .= zeros(solver.settings.r,solver.settings.r);
        for j = 1:s
            obj.SRK[s+1,:,:] = obj.SRK[s+1,:,:]+obj.alpha[s,j].*obj.SRK[j,:,:]+obj.dt*obj.beta[s,j].*obj.Srhs[j,:,:];
        end
    end

    return obj.SRK[obj.rkStages+1,:,:];
end

# SSP Update function for L-step
function UpdateL(obj::TimeSolver,solver::DLRSolver,X::Array{Float64,2},L::Array{Float64,2},recompute::Bool=true)
    N = solver.settings.N^2;
    obj.LRK[1,:,:] .= L;
    for s = 1:obj.rkStages
        obj.Lrhs[s,:,:] .= RhsL(solver,X,obj.LRK[s,:,:],recompute)';
        obj.LRK[s+1,:,:] .= zeros(N,solver.settings.r);
        for j = 1:s
            obj.LRK[s+1,:,:] = obj.LRK[s+1,:,:]+obj.alpha[s,j].*obj.LRK[j,:,:]+obj.dt*obj.beta[s,j].*obj.Lrhs[j,:,:];
        end
    end

    return obj.LRK[obj.rkStages+1,:,:];
end